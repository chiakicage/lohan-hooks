from typing import List

import torch
from lohan_hooks.parameter.param import LoHanParam, RawParam

from lohan_hooks.common import StorageStatus
from lohan_hooks.cpu_manager import CPUManagerWrapper

from lohan_hooks.logger import create_logger

logger = create_logger(__file__)
logger.setLevel("DEBUG")


class FetchInfo:
    pass


class LoHanParamGroup:
    def __init__(
        self,
        name: str,
        raw_params: List[RawParam],
        rank: int,
        device: torch.device,
        fetch_stream: torch.cuda.Stream,
        offload_stream: torch.cuda.Stream,
        cpu_mgr: CPUManagerWrapper,
    ) -> None:
        self.name = name
        self.rank = rank
        self.device = device
        self.params = [
            LoHanParam(raw_param.info, raw_param.param, self.device)
            for raw_param in raw_params
        ]
        self.status = StorageStatus.UNINITIALIZED

        self.fetch_event = torch.cuda.Event()
        self.offload_event = torch.cuda.Event()

        self.fetch_stream = fetch_stream
        self.offload_stream = offload_stream

        self.cpu_mgr = cpu_mgr

        logger.debug(f"LoHanParamGroup: {[r.info for r in raw_params]}")

    def init_weights(self) -> None:
        for param in self.params:
            param.status = StorageStatus.ON_CPU
        self.status = StorageStatus.OFFLOADED

    def fetch_params_async(self, with_grads: bool = False) -> int:
        logger.debug(f"Fetching {self.name}")
        fetch_size = 0
        with torch.cuda.stream(self.fetch_stream):
            for param in self.params:
                cpu_fp16_param = self.cpu_mgr.get_fp16_param(param.key)
                param.fetch_param_from_buffer_async(cpu_fp16_param.data)
                fetch_size += param.numel * param.dtype.itemsize
                if with_grads:  # Backward
                    if self.cpu_mgr.need_fetch_grads(param.key):
                        assert param.torch_param.requires_grad
                        cpu_fp16_grad = self.cpu_mgr.get_fp16_grad(
                            param.key, self.rank, False
                        )
                        param.fetch_grad_from_buffer_async(cpu_fp16_grad.data)
                    fetch_size += param.numel * param.dtype.itemsize
        self.status = StorageStatus.FETCHING
        self.fetch_event.record(self.fetch_stream)
        return fetch_size

    def wait_fetch_params(self) -> None:
        logger.debug(f"Waiting for fetching {self.name}")
        self.fetch_event.synchronize()
        for param in self.params:
            param.status = StorageStatus.READY
        self.status = StorageStatus.READY

    def fetch_params_sync(self, with_grads: bool = False) -> int:
        fetch_size = self.fetch_params_async(with_grads=with_grads)
        self.wait_fetch_params()
        return fetch_size

    def offload_params_async(
        self,
        with_grads: bool = False,
    ) -> int:
        logger.debug(f"Offloading {self.name}")
        offload_size = 0
        if with_grads:
            with torch.cuda.stream(self.offload_stream):
                for param in self.params:
                    if param.torch_param.requires_grad:
                        cpu_fp16_grad = self.cpu_mgr.get_fp16_grad(
                            param.key, self.rank, True
                        )
                        param.offload_grad_to_buffer_async(cpu_fp16_grad.data)
                    param.status = param.offloaded_param_status
                    param.free_gpu_param()
                    offload_size += param.numel * param.dtype.itemsize
            self.offload_event.record(self.offload_stream)
        else:
            for param in self.params:
                param.status = param.offloaded_param_status
                param.free_gpu_param()
                offload_size += param.numel * param.dtype.itemsize
        self.status = StorageStatus.OFFLOADING
        return offload_size

    def wait_offload_params(
        self,
        with_grads: bool = False,
    ) -> int:
        logger.debug(f"Waiting for offloading {self.name}")
        offload_size = 0
        if with_grads:
            self.offload_event.synchronize()
            for param in self.params:
                if param.torch_param.requires_grad:
                    self.cpu_mgr.commit_fp16_grad(param.key, self.rank)
                    param.free_gpu_grad()
                    offload_size += param.numel * param.dtype.itemsize
        self.status = StorageStatus.OFFLOADED
        return offload_size

    def offload_params_sync(
        self,
        with_grads: bool = False,
    ) -> int:
        offload_size = 0
        offload_size += self.offload_params_async(with_grads)
        offload_size += self.wait_offload_params(with_grads)
        return offload_size

    def query_offload_event(self) -> bool:
        return self.offload_event.query()
