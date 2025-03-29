# LoHan-Hooks
Hooks system in LoHan.
## Set the environment

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
