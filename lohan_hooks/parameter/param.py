from typing import Union
from dataclasses import dataclass
import torch
from torch import nn

from lohan_hooks.common import StorageStatus


class ParamInfo:
    def __init__(
        self,
        name: str,
        cpu_param_key: str,  # used in cpu param manager
        ref_count: int,  # used in cpu param manager
        numel: int,
        shape: torch.Size,
        dtype: torch.dtype,
        requires_grad: bool,
    ) -> None:
        self.name = name
        self.cpu_param_key = cpu_param_key
        self.ref_count = ref_count
        self.numel = numel
        self.shape = shape
        self.dtype = dtype
        self.requires_grad = requires_grad

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return str(self)


@dataclass
class RawParam:
    info: ParamInfo
    param: Union[nn.Parameter, torch.Tensor]


class LoHanParam:
    def __init__(
        self,
        info: ParamInfo,
        param: Union[nn.Parameter, torch.Tensor],
        device: torch.device,
    ) -> None:
        self.status = StorageStatus.UNINITIALIZED
        self.info = info
        self.name = info.name
        self.key = info.cpu_param_key
        self.torch_param = param.contiguous()
        self.shape = param.shape
        self.storage_size = self.torch_param.untyped_storage().size()
        self.numel = self.torch_param.numel()
        self.device = device

        self.offloaded_param_status = StorageStatus.ON_CPU

        self.dtype = self.torch_param.dtype
        assert (
            self.dtype == torch.float16 or self.dtype == torch.bfloat16
        ), f"Unsupported dtype: {self.dtype}"

    def fetch_param_from_buffer_async(self, buffer: torch.Tensor) -> None:
        assert (
            self.status == StorageStatus.ON_CPU
        ), f"Status: {self.status}, Info: {self.info}"
        assert buffer.untyped_storage().size() >= self.storage_size
        torch.cuda.nvtx.mark(f"Fetching param of {self.info.name}")
        self.torch_param.untyped_storage().resize_(self.storage_size)
        self.torch_param.untyped_storage().copy_(
            buffer.untyped_storage()[: self.storage_size],
            non_blocking=True,
        )
        self.status = StorageStatus.FETCHING_FROM_CPU

    def fetch_grad_from_buffer_async(self, buffer: torch.Tensor) -> None:
        assert buffer.untyped_storage().size() >= self.storage_size
        torch.cuda.nvtx.mark(f"Fetching grad of {self.info.name}")
        self.torch_param.grad = torch.empty_like(self.torch_param)
        self.torch_param.grad.untyped_storage().copy_(
            buffer.untyped_storage()[: self.storage_size],
            non_blocking=True,
        )

    def offload_grad_to_buffer_async(self, buffer: torch.Tensor) -> None:
        assert (
            self.status == StorageStatus.READY
        ), f"Status: {self.status}, Info: {self.info}"
        assert buffer.numel() >= self.numel
        assert (
            buffer.dtype == self.torch_param.dtype
        ), f"Buffer dtype: {buffer.dtype}, Param dtype: {self.torch_param.dtype}"
        assert self.torch_param.requires_grad
        assert self.torch_param.grad is not None
        torch.cuda.nvtx.mark(f"Offloading grad of {self.info.name}")
        self.free_gpu_param()
        flattened_grad = self.torch_param.grad.view(-1)
        buffer.view(-1).narrow(0, 0, self.numel).copy_(
            flattened_grad, non_blocking=True
        )
        self.status = StorageStatus.OFFLOADING_TO_CPU

    def free_gpu_param(self) -> None:
        self.torch_param.untyped_storage().resize_(0)

    def free_gpu_grad(self) -> None:
        if self.torch_param.requires_grad:
            assert self.torch_param.grad is not None
            self.torch_param.grad = None
