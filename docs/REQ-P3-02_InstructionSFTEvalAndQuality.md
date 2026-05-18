# REQ-P3-02：指令 SFT 效果检验、训练监控与质量优化

**所属**：[SPEC.md](../SPEC.md) → Part III · 指令微调（质量与评测闭环）  
**依赖**：[REQ-P3-01](REQ-P3-01_Ch07InstructionSFT.md)（训练脚本与数据管线已实现）、[REQ-P1-05](REQ-P1-05_Train.md)（Small 预训练 checkpoint）  
**被依赖**：暂无（后续若有「固定题库 + 自动打分」可再拆 REQ）  
**状态**：✅ **已收口**（2026-05-17：以 Small checkpoint 为唯一底座；§4.1 / §4.2 / §4.3 已合主线。自动评分与批量导出保留为 backlog）  
**现象记录**：[RUN_REPORT_instruction_sft_small.md](RUN_REPORT_instruction_sft_small.md)（冒烟跑通、日志解读）  
**OpenSpec（行为契约）**：[指令 SFT · `instruction-sft/spec.md`](../openspec/specs/instruction-sft/spec.md) 正文需求已覆盖 P3-02 的可验收行为。

---

## 1. 业务逻辑：为什么要单开一条 REQ？

### 先打个比方

[REQ-P3-01](REQ-P3-01_Ch07InstructionSFT.md) 像「工厂把流水线装好了，冒烟批能出货」。  
本条 REQ 像「**质检科**」：用什么量具、抽多少样、达不到什么数算返工；以及产线参数（数据量、轮数、解码）怎么调，才从「能出货」变成「能用的货」。

### 最关键的一句话

> **把「指令微调后好不好」从纯主观变成可复现的闭环**：全量或约定口径下的 **val loss**、固定 **prompt 清单**上的 **预训练 vs SFT 对照生成**、以及（backlog）自动/半自动评分；同时把训练脚本里 **抽样 eval / checkpoint 时机** 的坑用**配置或代码**收掉。

### P3-01 已交付 vs 本条要补齐

| P3-01 已覆盖 | 本条要补齐 |
|--------------|-----------|
| `finetune_instruction.py` + `m07` + 单测 | **训练监控**：全 val（或可配置）loss，而非仅 `eval_iter` 个 batch |
| `smoke_trim` 快速跑通 | **正式 Small 配方**：全量数据、多 epoch、`eval_val_batches: null` 的全 val 口径 |
| REQ §5「肉眼 2～3 条」 | **可抄的检验命令 + 记录表**（同 prompt 双 checkpoint 对照） |
| — | **checkpoint 策略**：epoch 末若优于历史 best 是否覆盖保存（与当前「仅按步 eval」对齐） |

---

## 2. 已记录问题（事实与解读）

以下来自 **Small + `config_instruction_small.json`（`smoke_trim=24`）** 及生成试验，**不**说明 P3-01 实现错误。

| 编号 | 现象 | 解读（讨论） |
|------|------|----------------|
| **Q-1** | 训练日志里 `val_loss` 约 8 → 4，但 **不能与预训练 Wiki `val_loss≈3.31` 比大小** | 数据分布与 mask 不同，属不同「科目」；见运行报告 **术语与口径** |
| **Q-2** | 训练中 `val_loss` 仅 **前 `eval_iter` 个 val batch** 的平均 | **趋势可信**，绝对值非全 val；长训应用应提供 **全 val** 或 `eval_iter = len(loader)` 的配置约定 |
| **Q-3** | **Step 10** 存盘的 `best_val_loss` 与 **epoch 末** 打印的 `val_loss` 可不一致；末值有时更好 | 当前逻辑仅在 **`global_step % eval_freq == 0`** 时更新 best；epoch 末未强制再比一次 |
| **Q-4** | `generate_from_checkpoint.py` 在「法国首都」类 **instruction 模板** 下，续写为 **维基碎片式英文**，**未**稳定输出「Paris」等短答 | **预期内**：数据极少、epoch 极少、LM 非检索；底座仍为自回归 **续写**，非 Chat 对齐 |
| **Q-5** | 默认 `temperature` + 较长 `--max-new-tokens` 易 **飘题** | 检验时应 **约定**解码参数（短续写、多跑几次或降低 temperature），否则主观对照噪声大 |

---

## 3. 优化方向（设计讨论 → 将落入 §4 验收）

### 3.1 训练侧（主因）

- **数据量**：生产配置 **`smoke_trim: null`**，使用划分后 **完整** instruction 集（或文档约定子集比例）。  
- **轮数与超参**：在 Small 上至少 **多 epoch**（具体数以 `val_loss` 平台期为准，写进 RUN_REPORT）；Medium 不作为本轮 P3 底座。  
- **监控口径**：增加 **`--eval-full-val`** 或 `eval_iter: null` 表示扫完整个 val DataLoader；日志中区分 **`val_loss_sampled`** vs **`val_loss_full`**（命名可在实现时定）。  
- **checkpoint**：在 **每个 epoch 结束** 用与 best 相同口径再算一次 val，若更优则 `torch.save`（或与按步 best 取 **min**，实现二选一并在 SPEC 写清）。

### 3.2 解码侧（检验时可控变量）

- **固定** `max_new_tokens`（指令问答建议先 **≤64** 或 ≤128，避免长跑题）。  
- 对照时 **预训练 / SFT 使用同一 decoding 超参**。  
- 文档中引用 [DOMAIN-KNOWLEDGE.md](DOMAIN-KNOWLEDGE.md) 已有 **`temperature=0` 与重复** 的说明。

### 3.3 评测侧（从轻到重）

| 层级 | 内容 | 价值 |
|------|------|------|
| **L0** | 单测 `test_instruction_finetune.py` + checkpoint 存在性 | 管线未坏 |
| **L1** | 固定 **3～10 条** 英文 instruction prompt，**同一字符串** 下对比两个 `.pt`  stdout | 低成本、可贴 RUN_REPORT |
| **L2** | 全 val **cross-entropy（与训练同 collate）** | 可汇报「SFT 后困惑度」 |
| **L3（backlog）** | 外链 LLM 打分或本地小模型打分（对齐书 `ollama_evaluate.py` 思路） | [REQ-P3-01](REQ-P3-01_Ch07InstructionSFT.md) **§9 BL-P3-01-02** 延伸 |

---

## 4. 交付与验收（草案）

**实现进度（仓库，2026-05-17）**：§4.1 / §4.2 / §4.3 均已在 `finetune_instruction.py`、`configs/config_instruction_train_small.json`、`compare_instruction_generate.py`、`eval_instruction_loss.py`、`README.md`、`docs/README.md`、`OWNER_CHECKLIST.md` 与 OpenSpec 中落地。本轮验收限定在 Small checkpoint；Medium 不作为底座。

### 4.1 阶段 A · 文档与可执行清单

| # | 交付物 | 怎样算过 |
|---|--------|----------|
| A1 | **[`RUN_REPORT_instruction_sft_small.md`](RUN_REPORT_instruction_sft_small.md)** 与本 REQ 说明「冒烟报告 + 事后复评」口径；正式 Small 跑使用 `config_instruction_train_small.json` 复现 | Reviewer 照抄命令能复现对照 |
| A2 | [`README.md`](../README.md) 与 [`docs/README.md`](README.md) 链到本条 REQ 与对照脚本 | 索引导航一致 |
| A3 | [`OWNER_CHECKLIST.md`](OWNER_CHECKLIST.md) 增加 **P3-02** 章节（可复制的 bash、判断阈值说明） | 负责人能自主验收 |

### 4.2 阶段 B · 训练脚本增强（代码）

| # | 交付物 | 怎样算过 |
|---|--------|----------|
| B1 | `finetune_instruction.py`（或抽出模块）支持 **全 val 评估** 的配置项 | 配置设全量后，日志出现 **单一明确字段**（如 `val_loss_full`），且与抽样值可区分 |
| B2 | **epoch 末** 与历史 best **同口径** 比较并更新 `checkpoint_best.pt`（若实现为「与按步 best 取最优」，须在注释与 SPEC 说明） | 复现：构造「末步优于中间步」场景能存到正确权重 |
| B3 | 新增或扩展配置：**`config_instruction_train_small.json`**（示例名）— `smoke_trim: null`、**`num_epochs` ≥ 2** 等 | `uv run python finetune_instruction.py --config …` 无 NaN，产出新 RUN_REPORT 草稿 |

### 4.3 阶段 C · 可选工具脚本

| # | 交付物 | 怎样算过 |
|---|--------|----------|
| C1 | `compare_instruction_generate.py` 或 Makefile 目标：对 **固定 prompt 文件** 两次调用 `generate_from_checkpoint.py`（预训练 vs SFT） | 一次命令产出并列文本或 Markdown 片段，便于贴报告 |
| C2 | `eval_instruction_loss.py`：加载 checkpoint + 数据配置，**只算 val loss**（不调优） | 与训练内 `calc_loss_*` 结果一致（同一 seed、loader） |

### 4.4 Backlog（不阻塞 B）

- **BL-P3-02-01**：外链/本地模型对生成结果打分（对齐 P3-01 §9 BL-P3-01-02）。  
- **BL-P3-02-02**：导出 `instruction-data-with-response*.json` 式 batch 推理结果（与 SUB 「刻意未实现」对齐，可选）。

---

## 5. 依赖与顺序

1. 本轮 P3-02 以 **Small 预训练 checkpoint** 为底座；P1-07 Medium 不阻塞、不作为 SFT 底座。  
2. L1 对照实验须 **固定解码参数**，并在报告中 **逐字记录**。  
3. 正式 Small 训练优先使用 [`configs/config_instruction_train_small.json`](../configs/config_instruction_train_small.json)；冒烟复评可继续使用 `configs/config_instruction_small.json`。

---

## 6. 文档索引

新增或更新：[`SPEC.md`](../SPEC.md)、[`HARNESS.md`](../HARNESS.md)、[`OWNER_CHECKLIST.md`](OWNER_CHECKLIST.md)、[`REFERENCE.md`](../REFERENCE.md)、[`LEARNING_LOG.md`](LEARNING_LOG.md) 中与指令质检相关的思考题。  
**OpenSpec（行为契约）**：已固化的可验收行为见 [`instruction-sft/spec.md`](../openspec/specs/instruction-sft/spec.md) 正文 **需求** + **场景**；文末路线图仅保留自动评分 / 批量导出等非阻塞 backlog。

---

## 7. 修订记录

| 日期 | 变更 |
|------|------|
| 2026-05-17 | 收口：Small-only 质检闭环完成；Medium 不作为本轮 SFT 底座；自动评分 / 批量导出保留为 backlog。 |
| 2026-05-07 | 标注进行中：§4.2/§4.3 已在主线实现（见 §4 顶部进度）。 |
| 2026-05-05 | 初稿：记录 Q-1～Q-5、优化三向（训练/解码/评测）、阶段 A/B/C 验收与 backlog。 |
