# 指令监督微调（SFT）

## 目的

从已有 **自回归 GPT** 预训练权重出发，在 **书本格式指令数据**（`instruction` / 可选 `input` / `output`）上继续训练，使模型在「带 `### Instruction:` / `### Response:` 模板」的英文文本上预测下一 token；**不得**在填充（pad）位置上对交叉熵损失计分。

本规格覆盖两部分：

| 范围 | 含义 |
|------|------|
| **核心训练契约（P3-01）** | 模板、`ignore_index`、微调入口、`checkpoint_best.pt` 与 `instruction_meta`、数据通路单测 |
| **监控与对照（P3-02）** | 验证集 loss **抽样 vs 全量**的可配置口径、**epoch 边界**与 checkpoint 对齐、**不改权重**的 val 评估入口、**固定解码**下的预训练 vs SFT **成对生成** |

业务叙事与分阶段 backlog 仍见 [`docs/REQ-P3-01_Ch07InstructionSFT.md`](../../../docs/REQ-P3-01_Ch07InstructionSFT.md)、[`docs/REQ-P3-02_InstructionSFTEvalAndQuality.md`](../../../docs/REQ-P3-02_InstructionSFTEvalAndQuality.md)；**验收句式以本文件「需求 / 场景」为准**。

## 非目标

- **不得**将本能力等同于「可靠事实问答」或与 Chat 类产品对齐；数据量很少、训练很短时，生成质量**不在**本规格保证范围内。  
- **可以**后续通过 [REQ-P3-01 §9](../../../docs/REQ-P3-01_Ch07InstructionSFT.md) 所列 backlog 扩展 DPO 等；**当前规格不包含**偏好学习。  
- **当前规格不包含**外链或本地大模型对生成结果的 **自动打分**（见文末 **路线图 · 仍为 backlog**）。

## 参阅文档

| 文档 | 用途 |
|------|------|
| [REQ-P3-01](../../../docs/REQ-P3-01_Ch07InstructionSFT.md) | 业务边界、Small 底座、验收草案 |
| [REQ-P3-01SUB](../../../docs/REQ-P3-01SUB_Ch07InstructionBookAlignment.md) | 与书本 `gpt_instruction_finetuning.py` 对齐细则 |
| [REQ-P3-02](../../../docs/REQ-P3-02_InstructionSFTEvalAndQuality.md) | 质检闭环：问题记录（Q-1～Q-5）、阶段 A/B/C 叙事 |
| [HARNESS.md](../../../HARNESS.md) Part III | 命令级 Harness 与通过判据 |
| [SPEC.md](../../../SPEC.md) · P3-01 / P3-02 | 配置字段、脚本路径、实现状态 |
| [changes/archive/instruction-sft-p3-02-monitor](../../changes/archive/instruction-sft-p3-02-monitor/) | P3-02 已合并变更：**proposal / design / tasks** 归档 |

---

## 变更追溯（changes）

| 变更 ID | 状态 | 说明 |
|---------|------|------|
| [`instruction-sft-p3-02-monitor`](../../changes/archive/instruction-sft-p3-02-monitor/) | **已合并** | 验证口径、`epoch_val_full`、对照脚本等已吸收进本文件 **需求**；归档目录保留提案与设计留痕 |

进行中或大跨度变更仍可按 [`changes/README.md`](../../changes/README.md) 新建 `openspec/changes/<change-id>/`。

---

## 技术设计（Design）

本节描述 **实现形状**（模块、配置键、数据流），与上文 **需求** 验收句式对应；细节以 [`SPEC.md`](../../../SPEC.md) · P3-01 / P3-02 为准。

### 组件边界

| 构件 | 职责 | 非职责 |
|------|------|--------|
| [`m07_instruction_finetune`](../../../src/mini_llm/m07_instruction_finetune/__init__.py) | `format_input`、划分、`InstructionDataset`、collate、`ignore_index` | 不写训练循环、不写 checkpoint 聚合策略 |
| [`finetune_instruction.py`](../../../finetune_instruction.py) | AdamW、按步 / 按 epoch 评估、`best_val_loss`、`checkpoint_best.pt`、`--eval-val-only` | 不修改 causal LM 算子 |
| [`generate_from_checkpoint.py`](../../../generate_from_checkpoint.py) | 任意兼容 ckpt 的自回归续写 | 不内置「双模型并排」逻辑（由 [`compare_instruction_generate.py`](../../../compare_instruction_generate.py) 组合） |
| [`eval_instruction_loss.py`](../../../eval_instruction_loss.py) | 调用与训练一致的 val DataLoader + CE | 不做反向传播 |

### 配置键（`instruction_finetune` 段）

| 键 | 类型 | 语义 |
|----|------|------|
| `eval_iter` | int 或 JSON `null` | 按步评估时 **train** 侧最多扫几个 batch；为 `null` 时 train **与** val（若无 `eval_val_batches`）均扫满 |
| `eval_val_batches` | int、JSON `null` 或省略 | 省略则与 `eval_iter` 对齐；`null` 表示按步评估时 **val 扫满** |
| `eval_freq` | int | 每多少个 **optimizer step** 触发一次按步评估 |
| `epoch_val_full` | bool | `true`（默认）：epoch 末用 **`val_loss_full`** 参与 best 比较并可写盘 |
| `smoke_trim` | int 或 `null` | 非 `null` 时三段划分各截前 k 条（冒烟） |

### 评估与 loss 数据流（简图）

```text
instruction JSON → split → Dataset → DataLoader(collate)
                                           │
                                           ▼
                              GPTModel.forward → CE(ignore_index)
                                           │
           ┌───────────────────────────────┴───────────────────────────────┐
           │ calc_loss_loader_instruction(loader, num_batches=None|cap)       │
           │   num_batches=None → 全 loader 平均 batch loss                   │
           │   num_batches=k   → 前 k 个 batch 平均                           │
           └─────────────────────────────────────────────────────────────────┘
```

### 设计不变量（与其它规格对齐）

- **张量契约**：与预训练相同，`input` / `target` 形如 `[B, T]`，CE 在展平后对 `-100` 忽略。  
- **checkpoint 磁盘格式**：须含 `model_state_dict`、`config`、`instruction_meta`（及脚本写入的 `instruction_finetune_config` 等），以便 [`generate_from_checkpoint.py`](../../../generate_from_checkpoint.py) 重建模型。

---

## 设计与取舍（P3-02：改了什么、为什么这样改）

以下内容是对实现选择的 **动机说明**，不等同于额外强制条款；条款见下文 **需求**。

1. **为什么要区分「抽样 val」和「全量 val」**  
   训练中频繁扫 **完整**验证集在大数据上很贵；用 **前若干个 batch** 的平均 loss 足以看 **趋势**。但日志若只写 `val_loss` 而不说明是抽样，容易误把它当成「整体验证集困惑度」（REQ-P3-02 **Q-2**）。因此在日志中 **应当**区分 **`val_loss_sampled`**（抽样口径）与 **`val_loss_full`**（完整验证集平均），并通过配置让调用方选择按步评估时 val 侧扫几批还是扫全集。

2. **为什么要 `eval_val_batches`（且允许与 `eval_iter` 脱钩）**  
   `eval_iter` 继续约束 **train / val 两侧用于「按步监测」的 batch 上限**的默认值；单独增加 **`eval_val_batches`**，使得 **train 仍可抽样**、**val 可按步扫全集**（JSON `null`），避免「要么两边全扫要么两边都抽样」的二选一。若 **`eval_iter` 本身为 `null`**，则按步评估时 train 与 val **均**扫完整 DataLoader（小数据专用，大训慎用）。

3. **为什么要 `epoch_val_full`（默认 `true`）**  
   旧逻辑仅在 `global_step % eval_freq == 0` 时用抽样 val 更新 best；**epoch 最后一个 batch 之后**算出的 val 有时优于中途 eval，却 **不会触发保存**（REQ-P3-02 **Q-3**）。因此在每个 epoch 结束额外计算 **`val_loss_full`**，若优于当前记录的 **`best_val_loss`**，则 **覆盖写入** `checkpoint_best.pt`。默认开启，使「一轮看完数据后的最优」有机会落盘；冒烟或对照实验可在配置中显式关掉。

4. **`best_val_loss` 数值口径会否混用？**  
   **会。** 中途保存可能依据 **抽样** val，epoch 末可能依据 **全量** val。二者在同一变量上与「更小更好」比较，是为了 **实现简单**、且绝大多数情况下 epoch 末 full 与抽样同单调方向。若项目要求 **严格单一口径**，调用方应配置 **`eval_val_batches: null`**（按步即全 val）或关闭中途按步 save（当前实现仍保留按步 save；更严策略可作为后续变更）。

5. **为什么要 `compare_instruction_generate.py` / `eval_instruction_loss.py`**  
   **成对生成**：主观对照时须 **固定解码超参**，否则会混入 sampling 噪声（REQ-P3-02 **Q-5**）；脚本用同一参数两次调用 `generate_from_checkpoint.py`，便于贴报告。  
   **仅算 val loss**：训练与「事后复检」共用同一 collate / DataLoader 构造，避免手写一行命令却漏掉配置字段。

---

## 需求

### 需求：指令字符串模板

系统 **应当**将每条训练样本拼成单一 UTF-8 字符串，包含固定引导语、`### Instruction:` 与任务正文、可选的 `### Input:`、以及 `### Response:` 与参考答案全文，与 REQ-P3-01SUB 所述书本模板一致。

#### 场景：省略可选 input

- **给定** JSON 条目中 `input` 为空或缺失  
- **当**为编码构建模板字符串时  
- **那么**字符串 **不得** 多出一段无依据的非空 `Input` 区（格式与 `m07_instruction_finetune` 参考实现一致）。

---

### 需求：填充位不计入损失

系统 **必须**在微调交叉熵中使用 `ignore_index`（约定 `-100`）排除填充位置，使模型不被训练去「预测填充 token」。

#### 场景：变长样本成批

- **给定**一批长度不等的已编码样本，经 collate 得到固定形状的 `inputs` 与 `targets`  
- **当**计算损失时  
- **那么**处于填充位置的 target **应当**被交叉熵归约忽略（等价地：这些位置无梯度）。

---

### 需求：微调入口与 checkpoint

系统 **应当**提供入口程序：加载与 [`GPTModel`](../../../SPEC.md) 兼容的预训练 checkpoint，执行指令 SFT，并在配置的 run 目录下持久化 **`checkpoint_best.pt`**（或等价的「验证最优」路径）。

#### 场景：冒烟配置跑通

- **给定**合法的 Small 预训练 checkpoint（如 WikiText-103 的 `checkpoint_best.pt`）与 [`configs/config_instruction_small.json`](../../../configs/config_instruction_small.json)  
- **当**以该配置执行指令微调脚本时  
- **那么**正常情形下流程 **应当**结束且损失中不出现 NaN  
- **且** **应当**按配置将最优 checkpoint 写入 run 输出目录（如 `runs/<run_name>/checkpoint_best.pt`）。

#### 场景：checkpoint 携带指令元数据

- **给定**一次成功的指令 SFT  
- **当**写入最优 checkpoint 时  
- **那么**文件 **应当**包含 `instruction_meta`（或等价字段），记录模板标识、pad token id、`ignore_index`、长度策略等，以便推理或后续训练与当时假设对齐。

---

### 需求：验证集 loss 的可配置口径（抽样 / 全量）

系统 **应当**支持通过配置区分「按步监测」时验证集 loss 是 **前若干个 batch 的平均** 还是 **完整验证集的平均**，并在日志中使用可区分字段名（见下条需求）。

#### 场景：`eval_val_batches` 覆盖 val 侧 batch 数

- **给定**[`finetune_instruction.py`](../../../finetune_instruction.py) 所用 JSON 配置中，`instruction_finetune.eval_iter` 为正整数，且 **`eval_val_batches`** 为 JSON **`null`**  
- **当**执行按步评估（`global_step % eval_freq == 0`）  
- **那么**验证集 loss **应当**按 **完整 val DataLoader** 计算批次平均（与 [`calc_loss_loader_instruction`](../../../finetune_instruction.py) 中 `num_batches is None` 语义一致）。

#### 场景：`eval_iter` 为 null（train 与 val 均全量）

- **给定**`instruction_finetune.eval_iter` 为 JSON **`null`**  
- **当**执行按步评估  
- **那么**训练侧与验证侧 loss **应当**均基于 **各自完整** DataLoader 的批次平均（适用于极小冒烟集；大数据慎用）。

---

### 需求：日志区分抽样验证 loss 与全量验证 loss

系统 **应当**在训练日志中明确区分 **抽样口径**的验证 loss 与 **全验证集口径**的验证 loss，避免将二者误读为同一数值。

#### 场景：按步打印

- **给定**按步评估触发  
- **当**打印该行日志  
- **那么**验证集一侧 **应当**使用字段名 **`val_loss_sampled`**（或等价、在 SPEC 中枚举的同一命名），表示当前配置下「非全量 val」的监测值。

#### 场景：epoch 结束摘要

- **给定**任意 epoch 的训练循环已处理完该 epoch 内全部训练 batch  
- **当**打印 epoch 结束摘要  
- **那么**输出 **应当**同时包含 **`val_loss_sampled`**（与按步监测一致的配置口径）与 **`val_loss_full`**（完整验证集批次平均 CE）。

---

### 需求：epoch 边界与 checkpoint 最优对齐

系统 **应当**支持配置：**在每个 epoch 结束时**，用 **全量验证集 loss**（`val_loss_full`）与当前记录的 **`best_val_loss`** 比较；若更优且数值有效，则 **更新** `checkpoint_best.pt`。

#### 场景：`epoch_val_full` 开启（默认）

- **给定**配置中 `instruction_finetune.epoch_val_full` 为 **`true`**（或未配置且实现默认值即为 true）  
- **当**某个 epoch 结束且已算出 **`val_loss_full`**  
- **那么**若 **`val_loss_full` < `best_val_loss`**（且非 NaN）  
- **那么**系统 **应当**将当前模型状态写入配置的 **`checkpoint_best.pt`**，并更新 **`best_val_loss`**。

#### 场景：`epoch_val_full` 关闭

- **给定**`epoch_val_full` 为 **`false`**  
- **当**epoch 结束  
- **那么**系统 **不得**仅凭 **`val_loss_full`** 覆盖由按步评估建立的「最优」checkpoint（仍可打印 `val_loss_full` 供人工阅读，具体行为以 SPEC 为准）。

---

### 需求：仅验证集 loss 评估入口（不写训练）

系统 **应当**提供 **不写回 checkpoint、不做反向传播** 的入口：给定与训练一致的数据配置与待评估的 `.pt`，输出 **抽样口径与全量口径**的验证 CE。

#### 场景：CLI 等价行为

- **给定**合法指令微调配置与含 `model_state_dict` 的 checkpoint  
- **当**执行 **`finetune_instruction.py --eval-val-only --eval-checkpoint <path>`** 或 **[`eval_instruction_loss.py`](../../../eval_instruction_loss.py)**（参数等价映射）  
- **那么**标准输出 **应当**包含 **`val_loss_sampled`** 与 **`val_loss_full`**（命名与 SPEC 一致），进程退出码 **应当**在成功时为 0。

---

### 需求：预训练与 SFT 的成对生成对照

系统 **应当**提供脚本（或 Makefile 目标）：对 **同一批固定 UTF-8 prompt**，在 **完全一致**的解码超参下，分别加载 **预训练 checkpoint** 与 **指令 SFT checkpoint**，调用与 [`generate_from_checkpoint.py`](../../../generate_from_checkpoint.py) 相同的生成逻辑并产出 **可并列阅读** 的输出（例如 Markdown）。

#### 场景：固定 prompt 文件

- **给定**[`compare_instruction_generate.py`](../../../compare_instruction_generate.py)、两个可读 checkpoint 路径，以及含至少一段非空 prompt 的文本文件（文件中以约定分隔符区分多条 prompt，见 SPEC）  
- **当**执行该脚本  
- **那么**进程 **应当**成功结束时，为每条 prompt 打印 **预训练** 与 **SFT** 各一段生成文本，且二者使用的 **`max_new_tokens` / `temperature` / `top-k` / `device`** 一致。

---

### 需求：数据通路的自动化测试

系统 **必须**提供自动化测试，校验指令数据划分、下载/缓存（本地路径）及与书本参考一致的 collate/掩码行为。

#### 场景：指令微调测试模块通过

- **给定**项目开发环境（如适用则 `uv sync --extra dev`）  
- **当**执行 `pytest tests/test_instruction_finetune.py` 时  
- **那么**全部用例 **应当**通过。

---

### 需求：正式 Small 训练配方配置（可选项）

系统 **应当**提供一份 **非冒烟**、面向 Small 预训练底座的 JSON 配置示例：**不使用 `smoke_trim`**（完整划分）、**多个 epoch**、并按需在按步评估时对 val 使用全量（例如 **`eval_val_batches`:** **`null`**），以便复现「比冒烟更接近书本尺度的」训练；详见 [`configs/config_instruction_train_small.json`](../../../configs/config_instruction_train_small.json)。

#### 场景：配置可被入口加载

- **给定**该配置文件路径  
- **当**执行 `finetune_instruction.py --config <path>`  
- **那么**解析 **应当**成功（字段含义见 SPEC）。

---

## 实施任务与门禁（Tasks）

以下为 **工程侧检查清单**，与 [REQ-P3-02 §4](../../../docs/REQ-P3-02_InstructionSFTEvalAndQuality.md) 阶段对应；已合并入主线的条目标记 ✅，自动评分 / 批量导出等非阻塞项留在路线图。

### 阶段 A · 文档与可执行清单

| ID | 任务 | 门禁 / 证据 |
|----|------|-------------|
| A1 | `RUN_REPORT_instruction_sft_small.md` 或正式 Small 报告中增加 **可复制检验步骤**（L0+L1） | ✅ 本 REQ / OWNER_CHECKLIST / README 已给出可复制命令；正式 Small 报告可按需追加 |
| A2 | 根 [`README.md`](../../../README.md) / [`docs/README.md`](../../../docs/README.md) 指向 REQ-P3-02 与对照脚本 | ✅ 导航 grep「P3-02」可见 |
| A3 | [`OWNER_CHECKLIST`](../../../docs/OWNER_CHECKLIST.md) Part III 覆盖 **eval_val_only**、**compare** | ✅ 已与现行脚本对齐；随 REQ 关闭再核对 |

### 阶段 B · 训练脚本（`finetune_instruction.py`）

| ID | 任务 | 门禁 / 证据 |
|----|------|-------------|
| B1 | 支持 **`eval_val_batches`** / **`eval_iter: null`**（全 val / 全 loader 语义） | ✅ + `pytest` 含 `_resolve_eval_batch_limits` |
| B2 | **`epoch_val_full`** 与 **`val_loss_full`** 参与 best | ✅ 日志含 epoch 末双字段 |
| B3 | [`config_instruction_train_small.json`](../../../configs/config_instruction_train_small.json) 可被加载 | ✅ JSON + SPEC 引用 |

### 阶段 C · 工具脚本

| ID | 任务 | 门禁 / 证据 |
|----|------|-------------|
| C1 | [`compare_instruction_generate.py`](../../../compare_instruction_generate.py) | ✅ 固定解码双 checkpoint |
| C2 | [`eval_instruction_loss.py`](../../../eval_instruction_loss.py) 或 `--eval-val-only` | ✅ 两行 val loss |

### 合并闸门（本能力「可演示」最小集）

| # | 门禁 | 通过判据 |
|---|------|----------|
| G1 | `pytest tests/test_instruction_finetune.py` | 全绿 |
| G2 | （可选）`--eval-val-only` 对已知 ckpt | stdout 两行 loss，exit 0 |
| G3 | （可选）`compare_instruction_generate.py` 双 ckpt | Markdown 两段生成 |

---

## 路线图（仍为 backlog / 非本规格正式需求）

下列项 **尚未**在本文件中升格为「必须 / 应当」条款；后续若决定实现，应新建变更并在本文件补充 **需求 + 场景**：

- 外链或本地模型对生成结果的 **自动评分**（对齐书本 `ollama_evaluate.py` 思路）。  
- 批量导出「instruction + 模型续写」JSON（REQ-P3-01SUB / REQ-P3-02 backlog）。  
- 若实践要求 **checkpoint 全程只用单一 loss 口径**，可增加配置项以 **禁止**按步抽样保存 best（当前实现允许抽样与 full 混用 **`best_val_loss`**，见上文 **设计与取舍 §4**）。
