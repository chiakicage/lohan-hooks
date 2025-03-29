from typing import (
    Dict,
    List,
    Tuple,
    Any,
    Union,
    Optional,
    Callable,
)

# import time
import weakref
from dataclasses import dataclass
import torch
from torch import nn
from torch.autograd import Variable

from lohan_hooks.model_info import ModelInfo
from lohan_hooks.common import TrainingState, StorageStatus
from lohan_hooks.logger import create_logger
from lohan_hooks.parameter import RawParam, LoHanParamGroup
from lohan_hooks.cpu_manager import LoHanCPUParamManager, CPUManagerWrapper

logger = create_logger(__file__)


@dataclass
class FetchInfo:
    stamp: int
    fetch_size: int
    param_group: LoHanParamGroup

    def __str__(self) -> str:
        return f"FetchInfo({self.stamp}, {self.fetch_size}, {self.param_group.name})"

    def __repr__(self) -> str:
        return str(self)


class LoHanComputeManager:
    def __init__(
        self,
        model: nn.Module,
        model_info: ModelInfo,
        cpu_mgr: LoHanCPUParamManager,
        rank: int,
        device: torch.device,
        memory_budget: float = 4e9,
    ) -> None:
        self.model = model
        self.model_name = model_info.prefix
        self.leaf_modules = model_info.leaf_modules
        self.cpu_mgr = CPUManagerWrapper(cpu_mgr, model_info.cpu_param_infos)
        self.rank = rank
        self.device = device
        self.stamp = 0
        torch.cuda.set_device(self.device)

        self.fetch_stream = torch.cuda.Stream()
        self.offload_stream = torch.cuda.Stream()

        # Module State
        self.module_states: Dict[str, LoHanModuleState] = {}

        # Hooks
        self.callback_stack: List[Tuple[int, Callable]] = []

        # Prefetch
        self.memory_budget = memory_budget  # 4GB
        self.current_memory = self.memory_budget
        self.forward_fetch_infos: List[FetchInfo] = []
        self.backward_fetch_infos: List[FetchInfo] = []
        self.locked_forward_infos: bool = False
        self.locked_backward_infos: bool = False
        self.active_l: int = 0
        self.active_r: int = 0
        self.offloading_l: int = 0

        self.training_state: TrainingState = TrainingState.IDLE

        self.init_model()

        logger.setLevel("DEBUG")

    def init_model(self) -> None:
        for name, module in self.model.named_modules(
            prefix=self.model_name, remove_duplicate=False
        ):
            state = LoHanModuleState(module, name, self)
            self.module_states[name] = state
        self.module_states[self.model_name].is_root_module = True

    def get_stamp(self) -> int:
        ret = self.stamp
        self.stamp += 1
        return ret

    def push_hook(self, stamp: int, hook: Callable) -> None:
        self.callback_stack.append((stamp, hook))

    def pop_hook(self, stamp: int) -> None:
        while (
            len(self.callback_stack) > 0 and self.callback_stack[-1][0] >= stamp
        ):
            _, hook = self.callback_stack.pop()
            hook()

    def add_fetch_info(
        self,
        stamp: int,
        fetch_size: int,
        param_group: LoHanParamGroup,
        is_backward: bool,
    ) -> None:
        if is_backward:
            assert (
                not self.locked_backward_infos
            ), "Cannot add fetch info after first iter"
            fetch_info = FetchInfo(stamp, fetch_size, param_group)
            self.backward_fetch_infos.append(fetch_info)
        else:
            assert (
                not self.locked_forward_infos
            ), "Cannot add fetch info after first iter"
            fetch_info = FetchInfo(stamp, fetch_size, param_group)
            self.forward_fetch_infos.append(fetch_info)

    def get_fetch_info(self, idx: int, backward: bool) -> FetchInfo:
        if backward:
            return self.backward_fetch_infos[idx]
        else:
            return self.forward_fetch_infos[idx]

    def fetch_infos_len(self, backward: bool) -> int:
        if backward:
            return len(self.backward_fetch_infos)
        else:
            return len(self.forward_fetch_infos)

    def lock_infos(self, backward: bool) -> None:
        if backward:
            self.locked_backward_infos = True
            self.backward_fetch_infos.sort(key=lambda x: -x.stamp)
            logger.debug(f"Backward Len: {len(self.backward_fetch_infos)}")
            logger.debug(f"Backward Infos: {self.backward_fetch_infos}")
        else:
            self.locked_forward_infos = True
            self.forward_fetch_infos.sort(key=lambda x: x.stamp)
            logger.debug(f"Forward Len: {len(self.forward_fetch_infos)}")
            logger.debug(f"Forward Infos: {self.forward_fetch_infos}")

    def reset_prefetch(self) -> None:
        logger.debug(
            f"{self.active_l=}, {self.active_r=}, {self.offloading_l=}"
        )
        assert (
            self.current_memory == self.memory_budget
        ), f"{self.current_memory} {self.memory_budget}"
        self.active_l = 0
        self.active_r = 0
        self.offloading_l = 0

    def prefetch(
        self, is_backward: bool, to_stamp: Optional[int] = None
    ) -> None:
        # Query offloading grads to free memory
        if is_backward and self.offloading_l < self.active_l:
            self.query_offloading_grads()
        # Prefetch new params
        while self.active_r < self.fetch_infos_len(is_backward):
            fetch_info = self.get_fetch_info(self.active_r, is_backward)
            if fetch_info.fetch_size > self.current_memory:
                # No prefetched memory or must prefetch to specific stamp
                if self.active_l == self.active_r or (
                    to_stamp is not None and fetch_info.stamp >= to_stamp
                ):
                    # Cannot prefetch any params, wait for offloading grads
                    assert is_backward and self.offloading_l < self.active_l
                    while fetch_info.fetch_size > self.current_memory:
                        self.query_offloading_grads()
                else:
                    break
            fetch_size = fetch_info.param_group.fetch_params_async(
                with_grads=is_backward
            )
            assert (
                fetch_size == fetch_info.fetch_size
            ), f"Module {fetch_info.param_group.name} fetch size mismatch {fetch_size} != {fetch_info.fetch_size}"
            self.current_memory -= fetch_info.fetch_size
            self.active_r += 1

        if (
            self.active_r < self.fetch_infos_len(is_backward)
            and self.active_l == self.active_r
        ):
            raise RuntimeError("Prefetch failed")

    def offload(self, stamp: int, is_backward: bool) -> None:
        logger.debug(
            f"Offloading at {stamp}, {self.active_l=}, {self.active_r=}, {self.offloading_l=}"
        )
        # Query offloading grads to free memory
        if is_backward and self.offloading_l < self.active_l:
            self.query_offloading_grads()
        if not is_backward:
            # Remove outdated params
            while self.active_l < self.active_r:
                fetch_info = self.get_fetch_info(self.active_l, backward=False)
                if stamp < fetch_info.stamp:
                    break
                offload_size = fetch_info.param_group.offload_params_sync(
                    with_grads=False
                )
                logger.debug(
                    f"Offloading {fetch_info.param_group.name} with {fetch_info.stamp} at {stamp}"
                )
                assert (
                    offload_size == fetch_info.fetch_size
                ), f"Module {fetch_info.param_group.name} offload size mismatch"
                self.current_memory += offload_size
                self.active_l += 1
        else:
            # Remove outdated params and offload grads
            while self.active_l < self.active_r:
                fetch_info = self.get_fetch_info(self.active_l, backward=True)
                if stamp > fetch_info.stamp:
                    break
                offload_param_size = (
                    fetch_info.param_group.offload_params_async(with_grads=True)
                )
                self.current_memory += offload_param_size
                self.active_l += 1
        logger.debug(
            f"Offloading at {stamp}, {self.active_l=}, {self.active_r=}, {self.offloading_l=}"
        )

    def query_offloading_grads(self, wait: bool = False) -> None:
        while self.offloading_l < self.active_l:
            fetch_info = self.get_fetch_info(self.offloading_l, backward=True)
            if not wait and not fetch_info.param_group.query_offload_event():
                break
            offload_grad_size = fetch_info.param_group.wait_offload_params(
                with_grads=True
            )
            self.current_memory += offload_grad_size
            self.offloading_l += 1

    def finish(self) -> None:
        self.cpu_mgr.finish()


class LoHanModuleState:
    def __init__(
        self,
        module: nn.Module,
        module_name: str,
        mgr: LoHanComputeManager,
    ) -> None:
        self.module = module
        self.module_name = module_name
        self.param_group: Optional[LoHanParamGroup] = None
        self.mgr_ref = weakref.ref(mgr)
        self.is_root_module = False
        self.first_forward = True
        self.first_backward = True
        self.enter_stamp: int = -1
        self.exit_stamp: int = -1

        self.register_hooks()

        logger.debug(
            f"LoHan enabled for {self.module_name}: {self.module.__class__.__name__}"
        )

    def register_hooks(self) -> None:
        self.module.register_forward_pre_hook(
            self.pre_forward_hook, with_kwargs=True, prepend=True
        )
        self.module.register_forward_hook(self.post_forward_hook, prepend=False)
        self.module.register_full_backward_pre_hook(
            self.pre_backward_hook, prepend=True
        )
        self.module.register_full_backward_hook(
            self.post_backward_hook, prepend=False
        )

    def init_weights(self) -> None:
        self.module.to_empty(device=self.mgr.device, recurse=False)

        if self.module_name in self.mgr.leaf_modules:
            managed_param_infos = self.mgr.leaf_modules[self.module_name]
            raw_params: List[RawParam] = []
            for param_name, param in self.module.named_parameters(
                prefix=self.module_name, recurse=False
            ):
                assert param_name in managed_param_infos
                param_info = managed_param_infos[param_name]
                raw_params.append(RawParam(info=param_info, param=param))
            self.param_group = LoHanParamGroup(
                name=self.module_name,
                raw_params=raw_params,
                rank=self.mgr.rank,
                device=self.mgr.device,
                fetch_stream=self.mgr.fetch_stream,
                offload_stream=self.mgr.offload_stream,
                cpu_mgr=self.mgr.cpu_mgr,
            )
            self.param_group.init_weights()

    def pre_forward_hook(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> None:
        if self.first_forward:
            self.enter_stamp = self.mgr.get_stamp()

        if self.is_root_module:
            self.mgr.training_state = TrainingState.FORWARD
            if not self.first_forward:
                # Start Prefetch
                self.mgr.prefetch(is_backward=False)

        if self.mgr.training_state == TrainingState.FORWARD:
            logger.debug(
                f"\033[1;32mPre-Forward pass for {self.module_name}: {module.__class__.__name__} #{self.enter_stamp}\033[0m"
            )
            if self.first_forward:
                self.init_weights()
            if self.param_group is not None:
                if self.param_group.status == StorageStatus.OFFLOADED:
                    if self.first_forward:
                        fetch_size = self.param_group.fetch_params_sync()
                        # Add fetch info for the first forward pass
                        self.mgr.add_fetch_info(
                            self.enter_stamp,
                            fetch_size,
                            self.param_group,
                            is_backward=False,
                        )
                    else:
                        raise RuntimeError(
                            f"Invalid Prefetch in {self.module_name}"
                        )
                elif self.param_group.status == StorageStatus.FETCHING:
                    self.param_group.wait_fetch_params()

        else:
            logger.debug(
                f"\033[1;32m-Recompute pass for {self.module_name}: {module.__class__.__name__} #{self.enter_stamp}\033[0m"
            )
            if self.param_group is not None:
                if self.param_group.status != StorageStatus.READY:
                    if self.first_backward:
                        assert (
                            self.param_group.status == StorageStatus.OFFLOADED
                        )
                        fetch_size = self.param_group.fetch_params_sync(
                            with_grads=True
                        )
                        # Add fetch info for the first backward pass
                        self.mgr.add_fetch_info(
                            self.enter_stamp,
                            fetch_size,
                            self.param_group,
                            is_backward=True,
                        )
                    else:
                        assert self.param_group.status == StorageStatus.FETCHING
                        self.param_group.wait_fetch_params()
            self.mgr.query_offloading_grads()

    def post_forward_hook(
        self, module: nn.Module, input: Any, output: Any
    ) -> None:
        if self.first_forward:
            self.exit_stamp = self.mgr.get_stamp()
        # Parameter
        if self.mgr.training_state == TrainingState.FORWARD:
            logger.debug(
                f"\033[1;32mPost-Forward pass for {self.module_name}: {module.__class__.__name__} #{self.exit_stamp}\033[0m"
            )
            if self.param_group is not None:
                torch.cuda.current_stream().synchronize()
                if self.first_forward:
                    self.param_group.offload_params_sync(with_grads=False)
                else:
                    self.mgr.offload(self.exit_stamp, is_backward=False)
                    self.mgr.prefetch(is_backward=False)

            if self.is_root_module:
                torch.cuda.current_stream().synchronize()
                self.mgr.fetch_stream.synchronize()
                self.mgr.offload_stream.synchronize()
                self.mgr.reset_prefetch()
                self.mgr.training_state = TrainingState.IDLE

            if self.is_root_module:
                self.mgr.lock_infos(backward=False)

            if self.first_forward:
                self.first_forward = False

        else:
            logger.debug(
                f"\033[1;32mPost-Recompute pass for {self.module_name}: {module.__class__.__name__} #{self.exit_stamp}\033[0m"
            )
            self.mgr.query_offloading_grads()

    def pre_backward_hook(
        self,
        module: nn.Module,
        grad_output: Union[Tuple[torch.Tensor, ...], torch.Tensor],
    ) -> None:
        self.mgr.pop_hook(self.exit_stamp)
        logger.debug(
            f"\033[1;32mPre-Backward pass for {self.module_name}: {module.__class__.__name__} #{self.exit_stamp}\033[0m"
        )
        if self.is_root_module:
            self.mgr.training_state = TrainingState.BACKWARD
            if not self.first_backward:
                self.mgr.prefetch(is_backward=True)
                logger.debug(
                    f"{self.mgr.forward_fetch_infos} {self.mgr.backward_fetch_infos}"
                )

        self.mgr.query_offloading_grads()
        if not self.first_backward:
            self.mgr.prefetch(is_backward=True, to_stamp=self.enter_stamp)

        # Parameter
        if self.param_group is not None:
            if self.param_group.status != StorageStatus.READY:
                if self.first_backward:
                    assert self.param_group.status == StorageStatus.OFFLOADED
                    fetch_size = self.param_group.fetch_params_sync(
                        with_grads=True
                    )
                    # Add fetch info for the first backward pass
                    self.mgr.add_fetch_info(
                        self.exit_stamp,
                        fetch_size,
                        self.param_group,
                        is_backward=True,
                    )
                else:
                    assert self.param_group.status == StorageStatus.FETCHING
                    self.param_group.wait_fetch_params()

    def post_backward_hook(
        self,
        module: nn.Module,
        grad_input: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        grad_output: Union[Tuple[torch.Tensor, ...], torch.Tensor],
    ) -> None:
        # Hooks
        if all([g is None for g in grad_input]):
            logger.debug(
                f"{module.__class__.__name__} has no grad input, skipping post-backward"
            )
            # Hooks
            self.register_post_backward_hook()
            return
        self.post_backward_hook_internal()

    def post_backward_hook_internal(self):
        self.mgr.pop_hook(self.enter_stamp)
        logger.debug(
            f"\033[1;32mPost-Backward pass for {self.module_name}: {self.module.__class__.__name__} #{self.enter_stamp}\033[0m"
        )
        # Parameter
        if self.param_group is not None:
            torch.cuda.current_stream().synchronize()
            if self.first_backward:
                self.param_group.offload_params_sync(with_grads=True)
            else:
                self.mgr.offload(self.enter_stamp, is_backward=True)

        self.mgr.query_offloading_grads()

        if self.is_root_module:
            torch.cuda.current_stream().synchronize()
            self.mgr.fetch_stream.synchronize()
            self.mgr.offload_stream.synchronize()
            self.mgr.query_offloading_grads(wait=True)
            self.mgr.reset_prefetch()

            if self.first_backward:
                self.mgr.lock_infos(backward=True)
            self.mgr.training_state = TrainingState.IDLE

        if self.first_backward:
            self.first_backward = False

    def register_post_backward_hook(self) -> None:
        if self.is_root_module:
            Variable._execution_engine.queue_callback(
                self.post_backward_hook_internal
            )
        else:
            self.mgr.push_hook(
                self.enter_stamp, self.post_backward_hook_internal
            )

    @property
    def mgr(self) -> LoHanComputeManager:
        mgr = self.mgr_ref()
        assert mgr is not None
        return mgr
