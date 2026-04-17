# 微型 LLM — 团队主项目（从零实现）

本目录与官方书本仓库 **`LLMs-from-scratch` 同级**，互不嵌套，避免与上游混淆。

```
001-360/
└── llms_team_work/
    ├── LLMs-from-scratch/   # 原书配套代码（勿改乱；仅作阅读 / 对照）
    └── team-mini-llm/       # 本仓库：你们手写的实现与实验
```

## 环境

**Python `>=3.10,<3.13`**（与 [`pyproject.toml`](pyproject.toml) 一致）。

在书本仓库中安装可编辑包（提供 `llms_from_scratch` 供**对照**或可选 import）：

```bash
cd ../LLMs-from-scratch
pip install -e .
```

回到本目录，用 **uv** 安装依赖（见 [`pyproject.toml`](pyproject.toml) 与 [`uv.lock`](uv.lock)）：

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

对照原书章节与路径见 [`REFERENCE.md`](REFERENCE.md)。

## 集成验收（总闸门）

- **M1** — 小数据训练一步 loss 可算且有限  
- **M2** — checkpoint 可被生成读入并产出文本  

**Harness 拆解、分层、REQ 表与需求模板** 单独维护在 **[`HARNESS.md`](HARNESS.md)**（新增需求以该文件为准迭代）。

## 代码布局（`mini_llm`）

实现按 **01→05** 顺序分子目录；Python 模块名不能以数字开头，故采用前缀 **`m`**（如 `m01_tokenizer`）。实现写在各子目录的 `__init__.py` 中。

| 目录 | 顺序 | 章节 |
|------|------|------|
| `m01_tokenizer/` | 01 | 第 2 章 · 分词 |
| `m02_data_loader/` | 02 | 第 2 章 · 数据与滑动窗口 |
| `m03_attention/` | 03 | 第 3 章 |
| `m04_model/` | 04 | 第 4 章 |
| `m05_generate/` | 05 | 生成（第 4 章生成 + 第 6/7 章衔接） |

第 5 章预训练在根目录 [`train.py`](train.py)（接在 `m04_model` 之后；`m05_generate` 可与训练并行开发，推理依赖 checkpoint）。

导入示例：`from mini_llm.m04_model import GPTModel`、`from mini_llm import m01_tokenizer`。

## 分工与交付物

| 章节 | 负责人 | 任务 | 交付物（`src/mini_llm/`） |
|------|--------|------|---------------------------|
| 第 2 章 | 同学 A | BPE、数据加载、滑动窗口 | `m01_tokenizer/`、`m02_data_loader/` |
| 第 3 章 | 同学 B | Self-Attention、因果掩码、Multi-Head | `m03_attention/` |
| 第 4 章 | 同学 C | TransformerBlock、完整 GPT | `m04_model/` |
| 第 5 章 | 同学 D | 训练循环、损失、评估 | 根目录 [`train.py`](train.py) |
| 第 6/7 章 | 同学 E | 生成、top-k / temperature | `m05_generate/` |
| 整合 + 汇报 | 组长 | 合并、跑通、PPT、报告 | 完整项目 |

## 范围

**必做（约第 1–2 周）**：数据与分词 → 注意力 → 模型 → 预训练（`train.py` checkpoint），与上表阶段一致。

**第 3 周**：生成与采样（`m05_generate/`）；微调二选一（第 6 章式分类 / 第 7 章式指令），对照路径见 [`REFERENCE.md`](REFERENCE.md) 中 `ch06`/`ch07`。

**不做**：附录 D/E 全套、Llama/Qwen bonus、多卡 DDP（除非环境已就绪）。

## 模块接口（约定）

对照只读：[`../LLMs-from-scratch/pkg/llms_from_scratch/`](../LLMs-from-scratch/pkg/llms_from_scratch/)

- **tokenizer**：BPE；`vocab_size` 与 `configs/config.json` 中 `model.vocab_size` 一致。
- **data_loader**：`input` / `target` 为 `[B, T]`。
- **GPTModel**：`forward(in_idx) -> logits`，形状 `[B, T, vocab_size]`；`cfg` 含 `vocab_size`, `context_length`, `emb_dim`, `n_heads`, `n_layers`, `drop_rate`, `qkv_bias`。
- **训练**：`loss = cross_entropy(logits.flatten(0,1), target_batch.flatten())`。
- **生成**：自回归；temperature、top-k（`m05_generate/`）。

## 仓库索引

| 路径 | 说明 |
|------|------|
| [`configs/config.json`](configs/config.json) | 超参与数据相关配置 |
| [`src/mini_llm/`](src/mini_llm/) | 源码：`m01_tokenizer/` … `m05_generate/` |
| [`train.py`](train.py) | 训练入口（预训练循环、损失、评估） |
| [`REFERENCE.md`](REFERENCE.md) | 如何对照隔壁书本仓库的章节与路径 |
| [`HARNESS.md`](HARNESS.md) | Harness 工程：分层、Part I/II REQ、需求模板 |
| [`docs/README.md`](docs/README.md) | 模块级设计文档索引（如 m01 分词器详设） |

## 对照跑通（可选）

若需先用官方包跑通流程，可在书本仓库内使用其自带章节脚本（见 [`REFERENCE.md`](REFERENCE.md)），**不要**把主实现长期依赖在 `import llms_from_scratch` 上。
