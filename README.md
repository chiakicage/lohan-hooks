# LoHan-Hooks
Hooks system in LoHan. The main system is in `lohan_hooks/compute_manager.py`, including torch hooks and prefetch. The parameter fetching and offloading function is in `lohan_hooks/parameter/param_group.py`, and parameters on CPU is managed by `lohan_hooks/cpu_manager.py`.

We use [Meta Device](https://pytorch.org/docs/stable/meta.html) to create models, then initialize and load weights at first forward pass in hooks system. Some module will initialize somethings when the module is created (like LlamaRotaryEmbedding), so we need to patch the model to add an additional initializing hook to it.

We use [nn.Module.register_module_(forward, ...)_hook](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html) to implement our hooks system. A `LoHanParameterGroup` will be attached to any `nn.Module` which has `nn.Parameter` (like `nn.Linear`), then manage its parameter by `LoHanParam`. `LoHanParam` will operate on the `untyped_storage()` of a tensor. From the perspective of torch, this tensor will always remain on the GPU even if it is offloaded, though it doesn't occupy any GPU memory at this point (like meta device).



Now, this repository use CPU parameter server in LoHan to fetch and offload parameters, you can replace it by modifying  `lohan_hooks/parameter/param_group.py` and `lohan_hooks/cpu_manager.py`.


## Set the environment

We use the blazing fast [uv](https://docs.astral.sh/uv/) to manage dependencies.

```shell
uv sync
source .venv/bin/activate # or other activate scripts for your own shell
python run_hf.py --model /data/models/Llama-3.2-1B/ --batch_size 1 --num_steps=10
```
If you want to change the index URL of pip in uv, you can modify your `~/.config/uv/uv.toml` as follows:

```toml
index-url = "https://mirrors.zju.edu.cn/pypi/web/simple"
```

## Develop Environment

```shell
uv sync --dev
pre-commit install
```
