import os
import glob
from typing import Dict, List, Set, Optional, cast, Callable
import ray
import torch
import asyncio
from multiprocessing import shared_memory

from lohan_hooks.parameter.param import ParamInfo
from lohan_hooks.logger import create_logger
from lohan_hooks.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    safetensors_weights_iterator,
)
from lohan_hooks.parameter.cpu_param import SharedTensor, LoHanCPUParam
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

logger = create_logger(__file__)


class MPGroup:
    def __init__(self, group_id: int, ranks: List[int]) -> None:
        self.group_id = group_id
        self.ranks = ranks
        self.inflight_grads: Dict[str, SharedTensor] = {}
        self.grad_hits: Dict[str, int] = {}
        self.grad_locks: Dict[str, asyncio.Lock] = {}

    async def acquire_grad_lock(self, param_name: str) -> None:
        if param_name not in self.grad_locks:
            self.grad_locks[param_name] = asyncio.Lock()
        await self.grad_locks[param_name].acquire()
        logger.debug(f"Acquired {param_name}")

    def release_grad_lock(self, param_name: str) -> None:
        self.grad_locks[param_name].release()
        logger.debug(f"Released {param_name}")

    def commit_grad(self, param_name: str) -> int:
        if param_name not in self.grad_hits:
            self.grad_hits[param_name] = 0
        self.grad_hits[param_name] += 1
        return self.grad_hits[param_name]

    def reset_grad_hits(self, param_name: str) -> None:
        self.grad_hits[param_name] = 0


class DPGroup:
    def __init__(self, group_id: int, mp_groups: List[MPGroup]) -> None:
        self.group_id = group_id
        self.mp_groups = mp_groups
        # self.inflight_grads: Dict[str, Dict[int, SharedTensor]] = {}
        self.grad_group_hits: Dict[str, int] = {}

    def rank_to_mp_group(self, rank: int) -> MPGroup:
        for mp_group in self.mp_groups:
            if rank in mp_group.ranks:
                return mp_group
        raise ValueError(f"Rank {rank} not found in any MP group")

    def commit_grad_group(self, param_name: str) -> int:
        if param_name not in self.grad_group_hits:
            self.grad_group_hits[param_name] = 0
        self.grad_group_hits[param_name] += 1
        return self.grad_group_hits[param_name]

    def reset_grad_group_hits(self, param_name: str) -> None:
        self.grad_group_hits[param_name] = 0


@ray.remote(num_cpus=1)
class LoHanCPUParamManager:
    def __init__(
        self,
        num_workers: int,
        param_infos: List[ParamInfo],
        dp_ranks: Optional[List[List[List[int]]]] = None,
    ) -> None:
        logger.setLevel("DEBUG")
        self.num_workers = num_workers
        self.barrier = asyncio.Barrier(self.num_workers)
        self.param_name_to_cpu_param: Dict[str, LoHanCPUParam] = {}
        if dp_ranks is None:
            self.dp_groups = [
                DPGroup(0, [MPGroup(i, [i]) for i in range(num_workers)])
            ]
        else:
            self.dp_groups = []
            for i, mp_ranks in enumerate(dp_ranks):
                mp_groups = []
                for j, ranks in enumerate(mp_ranks):
                    mp_groups.append(MPGroup(j, ranks))
                self.dp_groups.append(DPGroup(i, mp_groups))
        self.active_grads: Set[str] = set()
        # self.optimizer_commit_queue: Queue[str] = Queue()
        logger.debug(f"LoHanCPUManager: {param_infos}")
        self.init_cpu_param(param_infos)

        self.iter_first = True

    def wait_init(self) -> None:
        pass

    def rank_to_dp_group(self, rank: int) -> DPGroup:
        for dp_group in self.dp_groups:
            for mp_group in dp_group.mp_groups:
                if rank in mp_group.ranks:
                    return dp_group
        raise ValueError(f"Rank {rank} not found in any DP group")

    def _new_shared_tensor(
        self,
        numel: int,
        element_size: int,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> SharedTensor:
        shm = shared_memory.SharedMemory(create=True, size=numel * element_size)
        tensor = SharedTensor(dtype, shape, shm.name)
        return tensor

    def free_shared_tensor(self, tensor: SharedTensor) -> None:
        tensor.shm.close()
        tensor.shm.unlink()

    def new_cpu_param(self, info: ParamInfo) -> LoHanCPUParam:
        if info.dtype != torch.float16 and info.dtype != torch.bfloat16:
            logger.warning(
                f"Parameter {info.name} is neither dtype torch.float16 nor torch.bfloat16"
            )
        fp16_param = self._new_shared_tensor(
            info.numel, info.dtype.itemsize, info.dtype, info.shape
        )
        if info.requires_grad:
            fp32_param = torch.empty(
                info.shape, dtype=torch.float32, device="cpu"
            )
        else:
            fp32_param = None

        cpu_param = LoHanCPUParam(info, fp16_param, [], fp32_param)
        self.param_name_to_cpu_param[info.name] = cpu_param
        return cpu_param

    def init_cpu_param(self, infos: List[ParamInfo]) -> None:
        for info in infos:
            self.new_cpu_param(info)
            logger.debug(
                f"Init CPU Param: {info}, shape: {info.shape}, dtype: {info.dtype}"
            )

    def load_weights(self, hf_folder: str) -> None:
        hf_weights_files = glob.glob(os.path.join(hf_folder, "*.safetensors"))
        hf_weights_files = filter_duplicate_safetensors_files(
            hf_weights_files, hf_folder, SAFE_WEIGHTS_INDEX_NAME
        )
        weight_iterator = safetensors_weights_iterator(hf_weights_files)

        for param_name, weight in weight_iterator:
            assert (
                param_name in self.param_name_to_cpu_param
            ), f"param_name: {param_name} not found"
            cpu_param = self.param_name_to_cpu_param[param_name]
            assert (
                cpu_param.fp16_param.data.shape == weight.shape
            ), f"shape mismatch: {cpu_param.fp16_param.data.shape} != {weight.shape}"
            assert (
                cpu_param.fp16_param.data.dtype == weight.dtype
            ), f"dtype mismatch: {cpu_param.fp16_param.data.dtype} != {weight.dtype}"
            cpu_param.fp16_param.data.copy_(weight.data)
            if cpu_param.fp32_param is not None:
                cpu_param.fp32_param.copy_(weight.data.to(torch.float32))

    def init_weights(
        self, param_init_methods: Dict[str, Callable[[torch.Tensor], None]]
    ):
        for key, init_method in param_init_methods.items():
            assert key in self.param_name_to_cpu_param
            cpu_param = self.param_name_to_cpu_param[key]
            init_method(cpu_param.fp16_param.data)
            if cpu_param.fp32_param is not None:
                cpu_param.fp32_param.copy_(
                    cpu_param.fp16_param.data.to(torch.float32)
                )

    def get_fp16_param(self, param_name: str) -> SharedTensor:
        assert (
            param_name in self.param_name_to_cpu_param
        ), f"{param_name} not found"
        return self.param_name_to_cpu_param[param_name].fp16_param

    def get_fp16_grad(self, param_name: str, rank: int) -> SharedTensor:
        logger.debug(f"Getting FP16 Grad: {param_name}, Rank: {rank}")
        assert (
            param_name in self.param_name_to_cpu_param
        ), f"{param_name} not found"
        cpu_param = self.param_name_to_cpu_param[param_name]
        assert cpu_param.info.requires_grad
        dp_group = self.rank_to_dp_group(rank)
        mp_group = dp_group.rank_to_mp_group(rank)
        if param_name not in mp_group.inflight_grads:
            fp16_grad = self._new_shared_tensor(
                cpu_param.info.numel,
                2,
                cpu_param.info.dtype,
                cpu_param.info.shape,
            )
            mp_group.inflight_grads[param_name] = fp16_grad
        return mp_group.inflight_grads[param_name]

    def commit_fp16_grad(self, param_name: str, rank: int) -> None:
        logger.debug(f"Trying to Commit FP16 Grad: {param_name}, Rank: {rank}")
        assert (
            param_name in self.param_name_to_cpu_param
        ), f"{param_name} not found"

        if self.iter_first:
            self.iter_first = False
            torch.cuda.nvtx.range_push("Optimizer")

        dp_group = self.rank_to_dp_group(rank)
        mp_group = dp_group.rank_to_mp_group(rank)
        param_info = self.param_name_to_cpu_param[param_name].info
        torch.cuda.nvtx.mark(
            f"Commit FP16 Grad: {param_name}, MPGroup: {mp_group.group_id}"
        )
        if mp_group.commit_grad(param_name) == param_info.ref_count:
            mp_group.reset_grad_hits(param_name)
            if dp_group.commit_grad_group(param_name) == len(
                dp_group.mp_groups
            ):
                logger.debug(
                    f"Committing FP16 Grad: {param_name}, Rank: {rank}"
                )
                dp_group.reset_grad_group_hits(param_name)
                cpu_param = self.param_name_to_cpu_param[param_name]
                grads = [
                    group.inflight_grads[param_name].data
                    for group in dp_group.mp_groups
                ]
                cpu_param.fp16_grads = grads
                self.active_grads.add(param_name)
                dp_group.grad_group_hits[param_name] = 0

                torch.cuda.nvtx.mark(f"Commit FP16 Grad: {param_name}")

    async def acquire_grad_lock(self, param_name: str, rank: int) -> None:
        dp_group = self.rank_to_dp_group(rank)
        mp_group = dp_group.rank_to_mp_group(rank)
        await mp_group.acquire_grad_lock(param_name)

    def release_grad_lock(self, param_name: str, rank: int) -> None:
        dp_group = self.rank_to_dp_group(rank)
        mp_group = dp_group.rank_to_mp_group(rank)
        mp_group.release_grad_lock(param_name)

    def finish(self) -> None:
        for cpu_param in self.param_name_to_cpu_param.values():
            self.free_shared_tensor(cpu_param.fp16_param)
        for dp_group in self.dp_groups:
            for mp_group in dp_group.mp_groups:
                for grad in mp_group.inflight_grads.values():
                    self.free_shared_tensor(grad)


class CPUManagerWrapper:
    def __init__(
        self, cpu_mgr: LoHanCPUParamManager, cpu_param_infos: List[ParamInfo]
    ) -> None:
        self.cpu_mgr = cpu_mgr
        self.cached_params: Dict[str, SharedTensor] = {}
        self.cached_grads: Dict[str, SharedTensor] = {}
        self.has_grad: Set[str] = set()
        self.grad_hits: Dict[str, int] = {}
        self.grad_events: Dict[str, torch.cuda.Event] = {}
        self.cpu_param_infos = {
            info.cpu_param_key: info for info in cpu_param_infos
        }

    def get_fp16_param(self, key: str) -> SharedTensor:
        if key not in self.cached_params:
            self.cached_params[key] = cast(
                SharedTensor,
                ray.get(self.cpu_mgr.get_fp16_param.remote(key)),  # type: ignore
            )
            self.cached_params[key].pin()
            self.cached_params[key].unregister()
        return self.cached_params[key]

    def get_fp16_grad(
        self, key: str, rank: int, is_offload: bool
    ) -> SharedTensor:
        if key not in self.cached_grads:
            self.cached_grads[key] = cast(
                SharedTensor,
                ray.get(self.cpu_mgr.get_fp16_grad.remote(key, rank)),  # type: ignore
            )
            self.cached_grads[key].pin()
            self.cached_grads[key].unregister()
        if is_offload:
            self.has_grad.add(key)
        return self.cached_grads[key]

    def need_fetch_grads(self, key: str) -> bool:
        return key in self.has_grad

    def add_grad_event(self, key: str, grad_event: torch.cuda.Event) -> None:
        self.grad_events[key] = grad_event

    def commit_fp16_grad(self, key: str, rank: int) -> None:
        self.cpu_mgr.commit_fp16_grad.remote(key, rank)  # type: ignore

    def finish(self) -> None:
        for key, value in list(self.cached_params.items()):
            value.unpin()
            value.shm.close()
        for key, value in list(self.cached_grads.items()):
            value.unpin()
            value.shm.close()

    def sync_workers(self) -> None:
        ray.get(self.cpu_mgr.wait_init.remote())  # type: ignore
        ray.get(self.cpu_mgr.sync_workers.remote())  # type: ignore

    def sync_optimizer(self) -> None:
        ray.get(self.cpu_mgr.sync_optimizer.remote())  # type: ignore

    def flush(self) -> None:
        pass
