# 环境

- Python `>=3.10,<3.13`（与书本仓库 `pyproject.toml` 一致）。
- **本仓库**（[`team-mini-llm`](../)）使用 **uv** 管理依赖：在目录下执行 `uv sync`（开发再加 `--extra dev`），依赖见 [`pyproject.toml`](../pyproject.toml)，锁定见 [`uv.lock`](../uv.lock)。
- 对照官方书本代码时，可在 [`../../LLMs-from-scratch`](../../LLMs-from-scratch) 另行 `pip install -e .` 或对该仓库单独用 uv/pip（与 `mini-llm` 环境相互独立即可）。
- 主项目配置：[`configs/config.json`](../configs/config.json)。
- 语料目录可用环境变量 `TEAM_LLM_DATA_DIR`。
- 若工作区含同级 `LLMs-from-scratch`，`data_loader` 会优先使用该仓库 `ch02/01_main-chapter-code/` 下与配置 `filename` 同名的示例文件，便于离线开发、无需下载。
