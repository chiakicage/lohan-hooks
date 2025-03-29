from typing import Tuple

import torch
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaForCausalLM,
)


from lohan_hooks.logger import create_logger

logger = create_logger(__file__)


inited_inv_freq = False


def patch_llama_rope_init(
    model: LlamaForCausalLM, device: torch.device
) -> LlamaForCausalLM:
    def pre_rope_hook(
        rope: LlamaRotaryEmbedding, args: Tuple[torch.Tensor, ...]
    ) -> None:
        global inited_inv_freq
        if not inited_inv_freq:
            logger.info("Initializing inv_freq")
            inited_inv_freq = True
            assert isinstance(rope.inv_freq, torch.Tensor)
            inv_freq, rope.attention_scaling = rope.rope_init_fn(
                rope.config, device
            )
            rope.register_buffer("inv_freq", inv_freq, persistent=False)
            rope.original_inv_freq = rope.inv_freq

    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        assert isinstance(model.model.rotary_emb, LlamaRotaryEmbedding)
        model.model.rotary_emb.register_forward_pre_hook(pre_rope_hook)
    return model
