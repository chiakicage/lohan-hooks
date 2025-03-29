from typing import Dict, List, Callable, Optional
from functools import partial
import torch
from torch import nn
from lohan_hooks.logger import create_logger
from lohan_hooks.parameter.param import ParamInfo
from dataclasses import dataclass

logger = create_logger(__file__)


@dataclass
class ModelInfo:
    prefix: str
    cpu_param_infos: List[ParamInfo]
    leaf_modules: Dict[str, Dict[str, ParamInfo]]
    param_init_methods: Dict[str, Callable[[torch.Tensor], None]]


def get_model_info(
    model: nn.Module,
    prefix: str = "model",
    init_fn: Optional[Callable[[nn.Module, str, torch.Tensor], None]] = None,
) -> ModelInfo:
    visited_modules: Dict[nn.Module, str] = {}
    visited_params: Dict[nn.Parameter, str] = {}
    param_ref_count: Dict[str, int] = {}
    cpu_param_infos: List[ParamInfo] = []
    leaf_modules: Dict[str, Dict[str, ParamInfo]] = {}
    param_init_methods: Dict[str, Callable[[torch.Tensor], None]] = {}

    for name, param in model.named_parameters(
        prefix=prefix, remove_duplicate=False
    ):
        if param in visited_params:
            first_name = visited_params[param]
            logger.warning(
                f"Duplicate parameter: {name} detected, first seen at {first_name}"
            )
            # TODO(cage): handle ref count in model forward
            param_ref_count[first_name] += 1
            continue
        visited_params[param] = name
        param_ref_count[name] = 1

    for param, name in visited_params.items():
        cpu_param_info = ParamInfo(
            name=name,
            cpu_param_key=name,
            ref_count=param_ref_count[name],
            numel=param.numel(),
            shape=param.shape,
            dtype=param.dtype,
            requires_grad=param.requires_grad,
        )
        cpu_param_infos.append(cpu_param_info)

    for name, module in model.named_modules(
        prefix=prefix, remove_duplicate=False
    ):
        assert isinstance(name, str)
        assert isinstance(module, nn.Module)
        if module in visited_modules:
            first_name = visited_modules[module]
            raise ValueError(
                f"Duplicate module: {name} detected, first seen at {first_name}"
            )
        else:
            visited_modules[module] = name
        managed_param_infos: Dict[str, ParamInfo] = {}
        for n, p in module.named_parameters(prefix=name, recurse=False):
            cpu_param_key = visited_params[p]
            ref_count = param_ref_count[cpu_param_key]
            worker_param_info = ParamInfo(
                name=n,
                cpu_param_key=cpu_param_key,
                ref_count=ref_count,
                numel=p.numel(),
                shape=p.shape,
                dtype=p.dtype,
                requires_grad=p.requires_grad,
            )
            managed_param_infos[n] = worker_param_info

            if cpu_param_key not in param_init_methods and init_fn is not None:
                param_init_methods[cpu_param_key] = partial(init_fn, module, n)

        if len(managed_param_infos) > 0:
            leaf_modules[name] = managed_param_infos

    return ModelInfo(
        prefix=prefix,
        cpu_param_infos=cpu_param_infos,
        leaf_modules=leaf_modules,
        param_init_methods=param_init_methods,
    )
