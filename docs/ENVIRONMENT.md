# 环境

- Python `>=3.10,<3.13`（与书本仓库 `pyproject.toml` 一致）。
- 在 [`../../LLMs-from-scratch`](../../LLMs-from-scratch) 执行 `pip install -e .` 安装 PyTorch、tiktoken、pytest 等。
- 主项目配置：[`configs/config.json`](../configs/config.json)。
- 语料目录可用环境变量 `TEAM_LLM_DATA_DIR`（在 `train.py` 实现后读取）。
