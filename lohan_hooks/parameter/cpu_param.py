from typing import Optional, List
import cupy
import torch
from dataclasses import dataclass
from lohan_hooks.parameter.param import ParamInfo
from multiprocessing import shared_memory, resource_tracker


class SharedTensor:
    def __init__(self, dtype: torch.dtype, shape: torch.Size, shm_name: str):
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.data = torch.frombuffer(self.shm.buf, dtype=dtype).view(shape)

    def pin(self):
        cupy.cuda.runtime.hostRegister(
            self.data.data_ptr(),
            self.data.numel() * self.data.element_size(),
            0,
        )

    def unpin(self):
        cupy.cuda.runtime.hostUnregister(self.data.data_ptr())

    def unregister(self):
        resource_tracker.unregister(self.shm._name, "shared_memory")

    def __reduce__(self):
        return (SharedTensor, (self.data.dtype, self.data.shape, self.shm.name))


@dataclass
class LoHanCPUParam:
    info: ParamInfo
    fp16_param: SharedTensor
    fp16_grads: List[torch.Tensor]
    # fp32_grad: Optional[torch.Tensor]
    fp32_param: Optional[torch.Tensor]
