# 微型 LLM — 团队主项目（从零实现）

> 如果你只是想快速知道「这个项目做了什么、有哪些能力、怎么演示」，先读全员版说明：[`docs/PROJECT_OVERVIEW_EVERYONE.md`](docs/PROJECT_OVERVIEW_EVERYONE.md)。

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

加载训练保存的 checkpoint 做文本生成（检验效果）：

> **注意：** 本仓库 WikiText 实验为 **英文**语料；检验时请用 **英文** `--prompt`。中文开头容易误判为「模型很差」，说明见 [`docs/DOMAIN-KNOWLEDGE.md`](docs/DOMAIN-KNOWLEDGE.md) 第 6.4 节、[`docs/RUN_REPORT_gpt2_small_wikitext103.md`](docs/RUN_REPORT_gpt2_small_wikitext103.md) 第七节。

> **`--temperature 0`（贪心）与小模型：** 每步取概率最大的 token，输出可复现，但很容易出现 **同一词语反复出现**（如连续 `Mary`），观感常比默认 `0.8` + top-k **更差**——这是贪心解码的常见现象，不是 checkpoint 损坏。日常肉眼看续写建议保持默认；详见同一运行报告第七节参数表。
>
> **多行命令：** 除最后一行外，每行行尾必须有 `\`；否则下一行的 `--prompt` 不会传给 Python，会出现 `command not found: --prompt`。

```bash
cd /path/to/team-mini-llm
uv run python generate_from_checkpoint.py \
  --checkpoint runs/gpt2_small_wikitext103/checkpoint_best.pt \
  --prompt "The history of"
```

详见 [`generate_from_checkpoint.py`](generate_from_checkpoint.py) 文件头注释；运行报告里亦写有示例 [`docs/RUN_REPORT_gpt2_small_wikitext103.md`](docs/RUN_REPORT_gpt2_small_wikitext103.md)。

**SMS 分类（ham/spam）**：先微调 [`finetune_classify.py`](finetune_classify.py)（推荐演示权重：`configs/config_classify_spam_phase_b.json` → `runs/spam_classify_phase_b/`），再推理；**省略 `--checkpoint` 时默认该路径**。

```bash
uv run python finetune_classify.py --config configs/config_classify_spam_phase_b.json
uv run python classify_sms.py --text "Thanks see you tomorrow"
uv run python eval_classify.py
```

对照结论与健康阈值见 [`docs/REPORT_ClassifySpamProbe.md`](docs/REPORT_ClassifySpamProbe.md)、验收手册 [`docs/OWNER_CHECKLIST.md`](docs/OWNER_CHECKLIST.md)。

**指令微调（第 7 章 SFT）**：[`finetune_instruction.py`](finetune_instruction.py) + [`configs/config_instruction_small.json`](configs/config_instruction_small.json)（默认 `smoke_trim` 缩短样本；须先有 `runs/gpt2_small_wikitext103/checkpoint_best.pt`）。正式 Small（全划分、`eval_val_batches: null`、多 epoch）见 [`configs/config_instruction_train_small.json`](configs/config_instruction_train_small.json)。只算 val CE：[`eval_instruction_loss.py`](eval_instruction_loss.py) 或 `finetune_instruction.py --eval-val-only --eval-checkpoint …`。预训练 vs SFT 对照生成：[`compare_instruction_generate.py`](compare_instruction_generate.py) + [`docs/prompts/instruction_compare_sample.txt`](docs/prompts/instruction_compare_sample.txt)。本轮 P3-02 **以 Small checkpoint 为唯一底座收口**；Medium 仅保留为 P1-07 独立预训练实验，不作为指令 SFT 底座。质检闭环见 [REQ-P3-02](docs/REQ-P3-02_InstructionSFTEvalAndQuality.md)。

```bash
uv run python finetune_instruction.py --config configs/config_instruction_small.json
```

长任务（如 WikiText-103）建议使用项目虚拟环境绝对路径 + 无缓冲日志，避免 `uv` 在不同终端上下文里切到上级目录导致找不到 `train.py`，以及日志长时间不刷新：

```bash
nohup env PYTHONUNBUFFERED=1 "/abs/path/to/team-mini-llm/.venv/bin/python" -u "/abs/path/to/team-mini-llm/train.py" --config "/abs/path/to/team-mini-llm/configs/config_medium.json" > "/abs/path/to/team-mini-llm/train_wt103.log" 2>&1 &
tail -f "/abs/path/to/team-mini-llm/train_wt103.log"
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
| `m06_classify_finetune/` | 06 | 第 6 章 · SMS 分类微调 |
| `m07_instruction_finetune/` | 07 | 第 7 章 · 指令 SFT 数据与 collate |

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

**第 3 周**：生成与采样（`m05_generate/`）；微调：**第 6 章式分类**（[`REQ-P2-02`](docs/REQ-P2-02_ClassifyFinetune.md)）与 **第 7 章式指令 SFT**（[`REQ-P3-01`](docs/REQ-P3-01_Ch07InstructionSFT.md)）；原书对照见 [`REFERENCE.md`](REFERENCE.md) `ch06`/`ch07`。

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
| [`src/mini_llm/`](src/mini_llm/) | 源码：`m01_tokenizer/` … `m07_instruction_finetune/` |
| [`train.py`](train.py) | 训练入口（预训练循环、损失、评估） |
| [`finetune_classify.py`](finetune_classify.py) | SMS Spam 分类微调（第六章对齐） |
| [`finetune_instruction.py`](finetune_instruction.py) | 指令 SFT（第七章对齐；[`configs/config_instruction_small.json`](configs/config_instruction_small.json)、[`configs/config_instruction_train_small.json`](configs/config_instruction_train_small.json)） |
| [`eval_instruction_loss.py`](eval_instruction_loss.py) | 仅计算指令 val CE（抽样 + 全量；等价 `--eval-val-only`） |
| [`compare_instruction_generate.py`](compare_instruction_generate.py) | 同一批 prompt 下预训练 vs SFT 对照生成（Markdown） |
| [`classify_sms.py`](classify_sms.py) | 加载微调后的权重（**默认** `runs/spam_classify_phase_b/checkpoint_best.pt`），单行英文短信 → `ham` / `spam` |
| [`generate_from_checkpoint.py`](generate_from_checkpoint.py) | 加载 `checkpoint_*.pt` 做文本生成（检验效果） |
| [`REFERENCE.md`](REFERENCE.md) | 如何对照隔壁书本仓库的章节与路径 |
| [`HARNESS.md`](HARNESS.md) | Harness 工程：分层、Part I/II REQ、需求模板 |
| [`openspec/README.md`](openspec/README.md) | OpenSpec：[`specs/pretraining`](openspec/specs/pretraining/spec.md)、[`generation`](openspec/specs/generation/spec.md)、[`classify-sms`](openspec/specs/classify-sms/spec.md)、[`instruction-sft`](openspec/specs/instruction-sft/spec.md) |
| [`docs/README.md`](docs/README.md) | 模块级设计文档索引（如 m01 分词器详设） |
| [`.cursor/skills/team-mini-llm-domain/SKILL.md`](.cursor/skills/team-mini-llm-domain/SKILL.md) | Cursor Agent Skill：模块边界、术语、生成踩坑（与 DOMAIN-KNOWLEDGE 对齐） |

## 对照跑通（可选）

若需先用官方包跑通流程，可在书本仓库内使用其自带章节脚本（见 [`REFERENCE.md`](REFERENCE.md)），**不要**把主实现长期依赖在 `import llms_from_scratch` 上。
