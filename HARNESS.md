# Harness 工程

**行为契约（OpenSpec）**与命令判据对照：[`openspec/README.md`](openspec/README.md) · [`specs/pretraining/spec.md`](openspec/specs/pretraining/spec.md) 等。

总闸门 **M1 / M2** 见根目录 [`README.md`](README.md)。本文说明如何按 **Harness** 拆解交付物；**之后新增需求请按「需求模板」写**，避免只有功能描述没有验收手段。

## 思想

每个交付物都对应**边界契约**（输入 / 输出 / 形状或行为）+ **可执行 Harness**（`pytest`、或带固定配置的 `train.py` 一次运行）+ **通过判据**（数值有限、文件落盘、形状一致）。实现服从契约，验收服从 Harness，不在此之外另开「口头完成」。

## 分层

| 层 | 含义 | 典型 Harness |
|----|------|----------------|
| **L0** | 纯函数 / 单模块，无 I/O | `uv run pytest tests/...` 中单测 |
| **L1** | 单模块 + 配置或小张量 | 单测中构造 `config` / 固定随机种子 |
| **L2** | 模块链（如 loader → model → loss） | `tests/` 中集成测试或短脚本 |
| **L3** | 端到端预训练 / 生成 | `uv run python train.py --config configs/config.json`；checkpoint 与 `m05_generate` |

## 需求模板（复制使用）

| 字段 | 说明 |
|------|------|
| **REQ-ID** | 如 `REQ-P1-03` |
| **范围** | 涉及目录 / 责任人 |
| **契约** | 公开 API、张量形状、与 `configs/config.json` 的字段对应关系 |
| **Harness** | 哪条命令或哪个测试文件函数 |
| **通过判据** | 可观察、可重复（loss 有限、文件存在、形状 `(B,T,C)` 等） |
| **依赖** | 必须先绿的 REQ-ID |

## Part I（第 2–5 章 · 预训练闭环）

与 README 中「必做（约第 1–2 周）」一致；**总闸门 M1 = L3 通过判据**。

**P1-01** 的设计思路与理论说明见 [`docs/m01_tokenizer.md`](docs/m01_tokenizer.md)。

| REQ-ID | 交付 | 契约要点 | Harness | 通过判据 |
|--------|------|----------|---------|----------|
| P1-01 | `m01_tokenizer` | 与 `model.vocab_size` 一致；可 encode/decode | L0：`pytest` 中与 tokenizer 相关用例 | 往返一致、长度合理 |
| P1-02 | `m02_data_loader` | `input`/`target`：`[B,T]`；步长与 `context_length` 对齐 | L1/L2：loader + 小 batch | batch 形状正确、无越界 |
| P1-03 | `m03_attention` | 因果 mask；多头输出与 `emb_dim` 一致 | L0/L1：attention 用例 | 与已知张量对比或梯度有限 |
| P1-04 | `m04_model` | `logits`：`[B,T,V]` | L1/L2：`test_model_forward` 等 | 前向可跑、形状匹配配置 |
| P1-05 | `train.py` | CE loss；eval；`checkpoint_latest.pt` 路径与 `output_dir`/`run_name` | L3：`uv run python train.py --config configs/config.json` | **M1**：若干 step 后 train/val loss 为有限实数；checkpoint 写出 |
| — | **闸门 M1** | 上述 REQ 依赖链闭合 | L3 + 单测绿 | `pytest` 通过 + 训练不 NaN |

## Part II（第 3 周 · 生成与可选微调）

| REQ-ID | 交付 | 契约要点 | Harness | 通过判据 |
|--------|------|----------|---------|----------|
| P2-01 | `m05_generate` | 自回归；temperature、top-k；消费 checkpoint | L2/L3：加载 `runs/.../checkpoint_latest.pt` 跑短生成 | 输出为 token 序列 / 可 decode 文本 |
| — | **闸门 M2** | 训练 → 生成链路 | L3 | checkpoint 被加载且生成非空 |
| P2-02 | `m06_classify_finetune` + `finetune_classify.py` | SMS 微调；冻结 + 换 head；checkpoint **写入**含 `spam_max_length`；训练结束输出混淆矩阵 / spam PRF1 / **FN CSV** | L0：`pytest tests/test_classify_finetune.py`（前 6 个用例）+ `tests/test_classify_metrics.py`（见 SPEC §P2-02）；L3：`finetune_classify.py`；L2：`eval_classify.py --checkpoint runs/spam_classify_phase_b/checkpoint_best.pt`（不重训即可复评；演示默认） | test accuracy ≥ 90% |
| P2-03 | `classify_sms.py` | 加载 **分类** checkpoint；encode/load；stdout `ham`\|`spam` | L0：同上文件（后 2 个用例，见 SPEC §P2-03）；L2：`classify_sms.py --checkpoint … --text "..."` | 编码与 Dataset 一致；CLI 可运行 |

**Backlog（不阻塞上述 Harness）**：可选增强以 **`BL-P2-02-xx`** 记在 [`docs/REQ-P2-02_ClassifyFinetune.md`](docs/REQ-P2-02_ClassifyFinetune.md) §10（**BL-P2-02-02 已完成**，公式与行为见同文档 **§11**；其余如加权 CE、`--smoke`、官方 GPT-2 对照等仍为 todo）。

## Part III（第 7 章 · 指令微调）

| REQ-ID | 交付 | 契约要点 | Harness | 通过判据 |
|--------|------|----------|---------|----------|
| P3-01 | `m07_instruction_finetune` + `finetune_instruction.py` | 对齐书本 `instruction-data.json`；`format_input` / collate（pad→`ignore_index`）；**LM 头** CE(`ignore_index=-100`)；写出 `runs/<run_name>/checkpoint_best.pt`（含 `instruction_meta`） | L0：`pytest tests/test_instruction_finetune.py`；L3：`uv run python finetune_instruction.py --config configs/config_instruction_small.json`（须先有 Small 预训练 `.pt`；默认 `smoke_trim` 缩短数据） | loss 有限；`checkpoint_best.pt` 落盘 |
| P3-02 | （todo）全 val / epoch best / 对照生成 / 正式训练配方 | 见 [`docs/REQ-P3-02_InstructionSFTEvalAndQuality.md`](docs/REQ-P3-02_InstructionSFTEvalAndQuality.md) | 现阶段：[`docs/OWNER_CHECKLIST.md`](docs/OWNER_CHECKLIST.md) **Part III** | 以 REQ §4 阶段 A/B/C 为准 |

**Backlog**：**DPO / 偏好微调** → [`docs/REQ-P3-01_Ch07InstructionSFT.md`](docs/REQ-P3-01_Ch07InstructionSFT.md) §9（不阻塞 P3-01 SFT）。**指令质检与优化闭环** → [**REQ-P3-02**](docs/REQ-P3-02_InstructionSFTEvalAndQuality.md)（不阻塞 P3-01 轨道 A）。
