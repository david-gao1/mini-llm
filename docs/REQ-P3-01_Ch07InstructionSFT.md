# REQ-P3-01：第 7 章指令微调（SFT）——对齐书本数据与脚本，双轨底座（Small + GPT-2 Medium）

**所属**：[SPEC.md](../SPEC.md) → Part III · 指令微调（第 7 章）  
**依赖**：[REQ-P1-04](REQ-P1-04_Model.md)（`GPTModel`）、[REQ-P1-05](REQ-P1-05_Train.md)（预训练产物）；**轨道 B** 另依赖 [REQ-P1-07](REQ-P1-07_GPT2Medium.md)（Medium 预训练 checkpoint 可用）  
**被依赖**：无（后续可增加「指令推理 CLI」独立 REQ）  
**状态**：todo（本文档为范围与验收叙事；实现待开工）  
**原书对照**：与 `team-mini-llm` **同级**的 [`../../LLMs-from-scratch/ch07/`](../../LLMs-from-scratch/ch07/) · 主线脚本 `01_main-chapter-code/gpt_instruction_finetuning.py` · [`REFERENCE.md`](../REFERENCE.md)

---

## 1. 业务逻辑（为什么做）

> **一句话**：书上第七章是让模型学会「按指令回答」——把 **指令（+ 可选输入）+ 回答** 拼成一段文本，用 **下一词预测** 训练；loss 里对 **填充位置**（以及书上那样对「指令段」）做 mask，让梯度主要花在 **学会生成回答** 上。  
> **本轮只做 SFT**，不搞偏好学习；**DPO 等记 §9 backlog**，以后有空再接。

**和第 6 章的差别（人话）**：第六章是「整条短信打一个标签」；这里是「一段指令 → 模型要说一小段人话回答」，更像后续聊天产品的雏形，但 **仍然是监督微调**，不是 RL。

**已定范围（与你的决策对齐）**：

| # | 决策 | 含义 |
|---|------|------|
| 1 | **参考书里的数据与脚本** | 数据格式、拼接模板、`instruction-data.json` 来源与划分比例、以及 `custom_collate_fn`（pad、`ignore_index`）等 **优先逐段对齐** [`gpt_instruction_finetuning.py`](../../LLMs-from-scratch/ch07/01_main-chapter-code/gpt_instruction_finetuning.py)；再在 `mini_llm` 里用 **本仓库** `GPTModel` / `tiktoken` / 设备选择习惯重写，而不是复制粘贴依赖全书私有 `previous_chapters`。 |
| 2 | **只做 SFT** | 训练目标仅为指令数据的 CE / LM loss；**不包含** DPO、RM、PPO 等。 |
| 3 | **双轨底座** | **轨道 A**：**GPT-2 Small** + **很短指令集**（对齐书本 `test_mode` 思想：可先抽几十条 + pytest 固定种子，快速闭环）；**轨道 B**：以书上强调的 **GPT-2 Medium（约 355M 量级，与本仓库 `config_medium.json` / P1-07 一致）** 预训练 checkpoint 为起点，在同一套指令数据管线上跑 **更长** 的训练（资源与 `allowed_max_length` 等单独配置）。 |

---

## 2. 设计思路（怎么做）

**参考形态（书本 · 已实现逻辑）**：

- **数据**：默认远程 JSON（书上 URL；可镜像到本仓库 `runs/…/data_cache/` 以免重复下载）。
- **样本字段**：`instruction` / `input`（可空）/ `output`。
- **拼文模板**：`format_input` → `### Instruction:` … 可选 `### Input:` …；回答前加 `\n\n### Response:\n{output}`（与书一致，便于对照调试）。
- **批处理**：按 batch 内最长序列 pad；`targets` 里对 **padding** 标 `ignore_index`（默认 `-100`），使 `CrossEntropyLoss(ignore_index=…)` 不计入 pad；书上并对「连续 pad token」除第一个外继续 mask（对齐实现细节见原脚本）。
- **训练**：在 **完整 LM 头**（词表大小）上做 next-token loss，**不**换成二分类头（与第 6 章不同）。

**本仓库落地形态（建议，实施时可微调命名）**：

- 新模块：例如 `mini_llm.m07_instruction_finetune`（Dataset、collate、`format_input`、可选下载封装）。
- 新脚本：例如根目录 `finetune_instruction.py`（加载 **预训练** checkpoint → 训练循环 → 写出 **`checkpoint_best.pt` 或 `checkpoint_latest.pt`**，字段需与后续「指令生成」脚本约定）。
- 新配置：例如 `configs/config_instruction_small.json`、`configs/config_instruction_medium.json`（`pretrained_checkpoint`、`data.url` / 本地路径、`finetune.*`、`allowed_max_length`、batch、epoch、lr）。

**轨道 A vs B（同管线，换底座与规模）**：

| 轨道 | 预训练起点 | 指令数据规模（首期） | 说明 |
|------|------------|----------------------|------|
| **A** | `runs/gpt2_small_wikitext103/checkpoint_best.pt`（或等价 Small） | **很短**（对齐书本 test 小集 / 自建迷你 JSON） | 先验证：**loss 有限、可保存、可与书本 loss 趋势量级对照**（不强求数值完全一致）。 |
| **B** | P1-07 产出的 **Medium** checkpoint | 完整 `instruction-data.json` 或书上同等划分 | **依赖 Medium 已训稳**；注意显存与 `batch_size` / `allowed_max_length`。 |

---

## 3. 架构定位（在哪里）

```text
  instruction-data.json ──► InstructionDataset ──► DataLoader(custom collate)
                                    │
  pretrained LM checkpoint ─────────┴──► GPTModel（LM 头 V=vocab）
                                    │
                              finetune_instruction.py
                                    │
                         runs/…/checkpoint_*.pt（供后续生成脚本加载）
```

---

## 4. 契约要点（草案）

- **Dataset**：`__getitem__` 返回单条样本的 **token id 序列**（可与书本一致：整条含 Response）。
- **collate**：输出 `(inputs [B,T], targets [B,T])`，`targets` 中 pad（及书上约定位置）为 `ignore_index`。
- **checkpoint**：至少含 `model_state_dict`、`optimizer`（可选）、**训练用过的 `allowed_max_length`** 或与指令模板相关的 config，便于推理侧对齐。

（细化到字段级 Table 可在实现阶段补全，对齐 `finetune_classify.py` 写法。）

---

## 5. Harness / 验收（草案）

| 层级 | 命令（拟定） | 通过判据 |
|------|----------------|----------|
| L0 | `pytest tests/test_instruction_finetune.py`（Dataset / collate / mask） | 形状与 mask 行为与书本逻辑一致；固定种子可复现 |
| L2/L3 | `uv run python finetune_instruction.py --config configs/config_instruction_small.json` | loss 有限、无 NaN；写出 checkpoint |
| 定性 | 加载微调后权重 + 现有生成入口（或后续 CLI） | 对 **固定 2～3 条**指令，输出更像「回答」而非纯维基续写（写入 REPORT 或截图备忘即可） |

正式条目落地后同步 [`HARNESS.md`](../HARNESS.md)、[`SPEC.md`](../SPEC.md) §P3-01。

---

## 6. 依赖与阻塞

- **轨道 A**：Small 预训练权重路径存在即可。
- **轨道 B**：**阻塞于 [REQ-P1-07](REQ-P1-07_GPT2Medium.md)** 产出可用 Medium checkpoint；未就绪前先完成轨道 A。

---

## 7. 与原书章节的扩展对照（可选章节，**不纳入**本条 REQ 验收）

| 原书目录（同级 `LLMs-from-scratch/ch07/`） | 内容 | 本条 REQ |
|---------------------------------------------|------|----------|
| `02_dataset-utilities/` | 近重复样本等数据清洗 | 可选后续 |
| `04_preference-tuning-with-dpo/` | **DPO / 偏好** | **§9 backlog** |
| `06_user_interface/` | 简易 UI | 可选后续 |

---

## 8. 文档与导航

- 完成后更新：[`SPEC.md`](../SPEC.md)、[`HARNESS.md`](../HARNESS.md)、[`docs/README.md`](README.md)、[`REFERENCE.md`](../REFERENCE.md)、[`README.md`](../README.md) 第 3 周描述。
- **学习问题**可追加 [`LEARNING_LOG.md`](LEARNING_LOG.md)（指令 vs 分类、mask 含义等）。

---

## 9. Backlog（偏好学习 / DPO · **后续**）

> **人话**：SFT 是「标准答案抄写作业」；**DPO** 一类方法是「给你两篇回答，告诉你哪篇更好」，模型学偏好。**本条 REQ 不做**，避免和 SFT 验收缠在一起。

| ID | 主题 | 说明 | 原书参考 |
|----|------|------|----------|
| **BL-P3-01-01** | 偏好数据 + DPO 训练环 | 对齐 `ch07/04_preference-tuning-with-dpo/`（或书中更新路径）；依赖本条 SFT checkpoint 与配对数据 | `LLMs-from-scratch/ch07/04_preference-tuning-with-dpo/` |
| **BL-P3-01-02** | 评测脚本 | 可对齐书上 `ollama_evaluate.py` 思路或自建简易 BLEU/人工表 | `01_main-chapter-code/ollama_evaluate.py` |

---

## 10. 修订记录

| 日期 | 变更 |
|------|------|
| 2026-04-30 | 初稿：书面确认 **参考书数据/脚本**、**仅 SFT**、**Small 短集 + Medium 双轨**；DPO 记入 §9。 |
