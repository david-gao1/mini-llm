# 微型 LLM — 团队主项目（从零实现）

本目录与官方书本仓库 **`LLMs-from-scratch` 同级**，互不嵌套，避免与上游混淆。

```
001-360/
└── llms_team_work/
    ├── LLMs-from-scratch/   # 原书配套代码（勿改乱；仅作阅读 / 对照）
    └── team-mini-llm/       # 本仓库：你们手写的实现与实验
```

## 环境

在书本仓库中安装可编辑包（提供 `llms_from_scratch` 供**对照**或可选 import）：

```bash
cd ../LLMs-from-scratch
pip install -e .
```

回到本目录，用 **uv** 安装依赖（见根目录 [`pyproject.toml`](pyproject.toml) 与锁定文件 [`uv.lock`](uv.lock)）：

```bash
cd /path/to/team-mini-llm
uv sync
# 含 pytest 等开发依赖时：
uv sync --extra dev

uv run python train.py --help
uv run python train.py --config configs/config.json
uv run pytest
```

`uv sync` 会创建 `.venv` 并以可编辑方式安装本包，`import mini_llm` 无需再手动设 `PYTHONPATH`。

语料加载顺序：`TEAM_LLM_DATA_DIR` → 同级 `LLMs-from-scratch/ch02/...` 中的同名文件（离线）→ 按配置中的 `url` 下载到 `runs/<run_name>/data_cache/`。Checkpoint 默认写在 `runs/<run_name>/checkpoint_latest.pt`。

## 分工与交付物

章节任务、负责人与对应文件见 **[`docs/TEAM_WORK.md`](docs/TEAM_WORK.md)**。

组长 / PM + 架构视角：**全书模块映射、依赖与集成锚点、Harness 约定、各模块 DoD** 见 **[`docs/PROJECT_GOVERNANCE.md`](docs/PROJECT_GOVERNANCE.md)**。

## 文档与代码

| 路径 | 说明 |
|------|------|
| [`docs/PROJECT_GOVERNANCE.md`](docs/PROJECT_GOVERNANCE.md) | 治理与架构：章节映射、排期依赖、Harness、验收 |
| [`docs/TEAM_WORK.md`](docs/TEAM_WORK.md) | 分工表、交付物、整合说明 |
| [`docs/`](docs/) | 范围、环境、模块接口约定 |
| [`configs/config.json`](configs/config.json) | 超参与数据相关配置 |
| [`src/mini_llm/`](src/mini_llm/) | `tokenizer`、`data_loader`、`attention`、`model`、`generate` |
| [`train.py`](train.py) | 训练入口（预训练循环、损失、评估；见分工表） |
| [`REFERENCE.md`](REFERENCE.md) | 如何对照隔壁书本仓库的章节与路径 |

## 对照跑通（可选）

若需先用官方包跑通流程，可在书本仓库内使用其自带章节脚本（见 [`REFERENCE.md`](REFERENCE.md)），**不要**把主实现长期依赖在 `import llms_from_scratch` 上。
