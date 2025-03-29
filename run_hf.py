import argparse
import logging

import ray
import torch
from lohan_hooks import LoHanComputeManager, LoHanCPUParamManager
from lohan_hooks.logger import create_logger
from lohan_hooks.ray_plugins import register_ray_plugins
from lohan_hooks.model_info import get_model_info, ModelInfo
from lohan_hooks.patches.patch_llama import patch_llama_rope_init
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    LlamaForCausalLM,
)

logger = create_logger(__file__)

register_ray_plugins()


@ray.remote(num_gpus=1, num_cpus=1)
class Worker:
    def __init__(
        self,
        model: LlamaForCausalLM,
        model_info: ModelInfo,
        cpu_mgr: LoHanCPUParamManager,
        rank: int,
        dtype: torch.dtype,
    ):
        self.model = model
        self.cpu_mgr = cpu_mgr
        self.rank = rank
        self.dtype = dtype
        self.device = torch.device(f"cuda:{rank}")
        patch_llama_rope_init(self.model, self.device)
        self.mgr = LoHanComputeManager(
            model,
            model_info,
            cpu_mgr,
            self.rank,
            self.device,
            memory_budget=1e9,  # 1GB
        )

    @torch.no_grad()
    def step(self, ids: torch.Tensor) -> float:
        ids = ids.to(self.device)
        logits = self.model(ids, return_dict=False, use_cache=False)[0]
        logger.info(logits.shape)

        return logits

    def exit(self) -> None:
        self.mgr.finish()


def get_dataloader(batch_size, max_seq_len, vocab_size):
    def dataloader_fn():
        torch.manual_seed(42)
        while True:
            input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_len))
            yield input_ids

    return dataloader_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=1024)

    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    assert args.model is not None

    config = AutoConfig.from_pretrained(args.model)

    ray.init(num_gpus=1)

    run_env = {
        "env_vars": {
            "OMP_NUM_THREADS": "16",
            "OMP_PROC_BIND": "master",
            "OMP_PLACES": "sockets",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        },
        "numa": 0,
    }

    with torch.device("meta"):
        meta_model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
        )

    model_info = get_model_info(meta_model, prefix="")

    # Create CPU Manager
    cpu_mgr = LoHanCPUParamManager.options(runtime_env=run_env).remote(
        1,
        model_info.cpu_param_infos,
    )

    ray.get(cpu_mgr.wait_init.remote())
    ray.get(cpu_mgr.load_weights.remote(args.model))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataloader_fn = get_dataloader(
        args.batch_size,
        args.max_seq_len,
        config.vocab_size,
    )
    dataloader = dataloader_fn()
    try:
        # Create Workers
        worker = Worker.options(runtime_env=run_env).remote(
            meta_model,
            model_info,
            cpu_mgr,
            0,
            torch.bfloat16,
        )

        logger.info("Workers Created")

        torch.manual_seed(42)
        for step in range(args.num_steps):
            input_ids = next(dataloader)

            results = ray.get(worker.step.remote(input_ids))
            print(results)

    except Exception as e:
        logger.exception(e)
    finally:
        worker.exit.remote()
        ray.get(cpu_mgr.finish.remote())
        worker.__ray_terminate__.remote()
        cpu_mgr.__ray_terminate__.remote()
        ray.shutdown()


if __name__ == "__main__":
    main()
