# REQ-P2-02：分类微调（SMS Spam 二分类）

**所属**：[SPEC.md](../SPEC.md) → Part II · 模块 06  
**依赖**：[REQ-P1-04](REQ-P1-04_Model.md)（GPTModel）、[REQ-P1-05](REQ-P1-05_Train.md)（预训练 checkpoint）、P1-07（GPT-2 Small 预训练完成）  
**被依赖**：[REQ-P2-03](REQ-P2-03_ClassifySmsInfer.md)（SMS 分类推理脚本）  
**状态**：进行中（主线微调 + **BL-P2-02-02 评估扩展已并入**，见 §11）  
**分析报告**：[REPORT_ClassifySpamProbe.md](REPORT_ClassifySpamProbe.md)（探针误判、高置信 FN、改进计划）  
**领域详解**：[DOMAIN-KNOWLEDGE.md](DOMAIN-KNOWLEDGE.md) **§6.6**（混淆矩阵、PR/F1、FN 导出与 best checkpoint）

---

## 1. 业务逻辑（为什么做）

### 预训练模型为什么可以使用

预训练在本仓库中的具体产出是 **GPT-2 Small 架构 + WikiText-103** 的一次完整跑（约 163M 参数，详见 [`RUN_REPORT_gpt2_small_wikitext103.md`](RUN_REPORT_gpt2_small_wikitext103.md)）；REQ 中的 **P1-07** 在 SPEC 里仍对应「大规模预训练」主线，微调 checkpoint 路径默认指向该 Small 运行的 `checkpoint_best.pt`。

**「基础语法和词汇能力」如何理解、算不算成立：**  
语言模型只被训练做 **下一 token 预测**。我们没有单独跑语法标注集或词汇量测验；所谓「语法、词汇」是对现象的**通俗说法**：验证集 **val_loss** 持续下降（本跑最佳约 **3.31**，困惑度 PPL≈**27**），说明在 **WikiText 风格的英文正文**上，模型给「像维基那样接续」的序列更高概率——这在统计意义上等同于学到了该域的用词与共现规律（常被口语化成「语法 / 搭配」）。**严谨表述**：具备 **WikiText 域上的语言建模能力**；**不宣称**已通过独立的语法或语义评测。

**如何评估（由严到松）：**

| 方式 | 看什么 | 本项目现状 |
|------|--------|------------|
| **验证集语言建模** | `val_loss`、PPL=`exp(val_loss)` | 已有完整曲线与附录表，见运行报告 |
| **生成人工检视** | 英文 prompt 续写是否可读、是否维基体 | 辅助信号，不能代替定量指标 |
| **跨域困惑度** | 换语料（如新闻）算 PPL | 未做；预期与 Wiki 域会有落差 |
| **下游微调** | 如 SMS 分类 accuracy | 验证「表征是否好用」，不直接等于语法分数 |

它仍然只会按 **自回归 LM** 工作——没有任何内置的 **判别头**；分类要靠本章微调加上。

微调的目标是：**复用预训练学到的表征，让模型学会一个新任务——判断一条短信是正常短信（ham）还是垃圾短信（spam）**。

为什么可以这样做：
- 预训练学到的 token embedding 和 Transformer 层已经适应了 **英文 token 序列的统计结构**（在上述度量意义上）；分类阶段再用少量参数对齐标签空间即可
- 分类任务只需要在模型顶部换一个分类头（768 → 2），训练少量参数就能达到高准确率
- 这种"冻结底层 + 微调顶层"的迁移学习是 NLP 的标准范式



###  为什么选 SMS Spam 作为首个微调任务：
- 二分类是最简单的分类任务，便于验证整条微调链路
- 数据集小（约 1500 条平衡后），训练速度快（几分钟）
- 与书本第六章对齐，便于对照学习
- 准确率指标直观，一看就知道训练是否成功

---

## 2. 设计思路（怎么做）

**方案**：新建模块 `m06_classify_finetune` 封装数据管道 + 评估函数，新建脚本 `finetune_classify.py` 完成模型改造与训练循环。

**为什么不修改 GPTModel 类本身**：
- GPTModel 的 forward 输出 `[B, T, V]` logits，这个接口在预训练和生成中都是正确的
- 分类任务只需要取 `logits[:, -1, :]`（最后一个 token 的输出），然后在脚本层面做 `argmax`
- 替换 `out_head`（从 `Linear(768, 50257)` 到 `Linear(768, 2)`）在脚本中动态完成，不污染原模型定义
- 保持 m04_model 的单一职责

### **为什么取最后一个 token 做分类**：

GPT 的自注意力带了 **因果掩码（causal mask）**：可以理解为「从左读到右」，在某个位置上更新表征时，**不能用到该位置之后的任何 token**。因此：
- **靠前的 token**：它们的表征里**没有**后面词语的信息（因为它们在当时还不允许往后看）。
- **最后一个 token**：它是整条序列里**第一个允许「看见完整前缀」**的位置——在经过多层 Transformer 之后，它的表征已经把前面各个位置的信号经过多次传递与混合。

我们做短信分类时，需要一句「整句话说了啥」的信号。**最简单的一种用法**：取出最后一个位置的隐向量（或它上面的 logits），给它接一个分类头。**注意**：这里的「摘要」是工程师口语——指的是 **一个融合了整条前缀信息的向量**，方便分类；**并不是**模型自动生成了一段人类可读的摘要文本。

代码里对应：`logits = model(batch)[:, -1, :]`，`[:, -1, :]` 就是「只取最后一个 token 那一列」。

### **为什么冻结大部分层、只解冻末层**：
- 预训练权重已经包含有用的语言表示，全量微调会破坏这些表示（灾难性遗忘）。
- **训练快、不易过拟合、对小数据集友好**：梯度只更新少数参数，优化器状态（Adam 的动量等）也小得多。

**「只训练最后一个块 + final_norm + 新头」**——白话版（细节见 [`m04_model`](../src/mini_llm/m04_model/__init__.py)）：

- **不动**：前面的词向量、位置向量，以及 **除最后一层以外的所有 Transformer 层**（默认 12 层里冻住前 11 层）。里面已经装好了「看懂英文」的本领，尽量别改坏。
- **要动**：
  - **最后一层 Transformer**：专门再学一点，把特征拧成更适合「这条短信是正常还是垃圾」。
  - **final_norm**：进输出头前的那一下归一化，跟着最后一层一起调。
  - **输出头**：预训练是猜下一个词（5 万多个类）；分类时换成 **2 类**（ham / spam），相当于换一块小「打分板」，从头学。

所以：**大半个模型当固定特征提取器，只练最上面薄薄几层。** 在本仓库 GPT-2 Small 默认配置下，大约只有 **百分之五、六** 的参数会更新（约 **7M** 量级，其余冻结）；`finetune_classify.py` 开头会打印准确比例。把 `unfreeze_last_n_blocks` 调大，要练的层数会跟着变多。

### **为什么要平衡数据集**：

标签含义：**ham** = 正常短信；**spam** = 垃圾短信（数据集沿用业界习惯的英文标签）。

- 原始 SMS 数据中 ham:spam ≈ 87%:13%，极度不平衡
- 不平衡会导致模型只学会"都预测 ham"也能得 87% 准确率
- 下采样 ham 到与 spam 同等数量（从多数类里**随机丢掉一部分 ham**，使两类条数一样），迫使模型真正学会区分两类


> 下采样：数据里某一类太多时，故意少用一点，把这一类样本数量压下去。
> 这里的做法是：spam 比较少，ham 很多；就从 ham 里随机抽和 spam 同样多条，没用的 ham 先不放进来训练。这样两类各占一半，模型不能只靠「全猜 ham」刷准确率。
> 相对的概念是 上采样：少数类不够就多复制或人造样本把它「凑多」。我们用的是下采样 ham，实现简单，代价是一部分正常短信没被用上。


**关键设计决策**：
- pad_token_id = 50256（GPT-2 词表最后一个可用 ID）
- max_length 由 train 集中最长序列决定，val/test 对齐到同一长度
- loss 函数：`cross_entropy(logits[:, -1, :], labels)`，只取最后一个 token 位置
- 优化器：AdamW，lr=5e-5，weight_decay=0.1

---

## 3. 架构定位（在哪里）

```text
     ┌────────────────────────────────────────────────────┐
     │  finetune_classify.py                              │
     │  ├─ 加载预训练 checkpoint                           │
     │  ├─ 改造模型（冻结 + 换 head + 解冻末层）            │
     │  ├─ 训练循环（loss + eval + accuracy）              │
     │  ├─ 保存分类模型 checkpoint                         │
     │  └─ 末段：加载 **best val** → test 混淆矩阵/PRF1/FN CSV │
     └──────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┼───────────────────────┐
        ▼           ▼                       ▼
 ┌──────────┐ ┌──────────┐       ┌──────────────────────┐
 │ m04_model│ │m01_token │       │ m06_classify_finetune │
 │ GPTModel │ │ tiktoken │       │ SpamDataset, metrics  │
 └──────────┘ └──────────┘       │ collect_predictions…  │
                                 └───────────┬──────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
          [REQ-P2-03] classify_sms.py（单条推理）          eval_classify.py（整集评估 + FN CSV）
```

**上游**：预训练 checkpoint（P1-07 产出）→ GPTModel 权重  
**本模块**：数据下载 + Dataset + accuracy + **扩展评估（混淆矩阵、per-class PRF、FN 导出）**  
**下游**：分类 checkpoint → [REQ-P2-03](REQ-P2-03_ClassifySmsInfer.md)（`classify_sms.py`）；离线复评 → [`eval_classify.py`](../eval_classify.py)

---

## 4. 输入 / 输出契约

### m06_classify_finetune 公开 API

```python
download_and_prepare_spam(data_dir: Path) -> tuple[Path, Path, Path]
    # 下载 UCI SMS Spam zip → 解压 → 平衡 → 70/10/20 切分
    # 返回 (train.csv, val.csv, test.csv)

class SpamDataset(Dataset):
    def __init__(self, csv_path: Path, max_length: int | None = None,
                 pad_token_id: int = 50256)
    def __len__(self) -> int
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]
        # (token_ids: [max_length] int64, label: int64 标量)

calc_accuracy_loader(
    loader: DataLoader, model: nn.Module, device: torch.device,
    num_batches: int | None = None,
) -> float
    # 返回 0.0~1.0 之间的准确率

collect_predictions_loader(
    loader: DataLoader, model: nn.Module, device: torch.device,
) -> tuple[Tensor, Tensor]
    # shuffle=False 时与 Dataset 行序一致；返回 (preds, targets)，均为 1-D long

confusion_counts_binary_spam(targets: Tensor, preds: Tensor) -> tuple[int, int, int, int]
    # ham=0, spam=1 → (TN, FP, FN, TP)；定义见 §11

prf1_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]
    # precision/recall/f1_spam 与 ham 各三项（浮点）

accuracy_from_counts(tn: int, fp: int, fn: int, tp: int) -> float

export_false_negative_spam_csv(
    dataset: SpamDataset, targets: Tensor, preds: Tensor, out_path: Path,
    *, max_rows: int | None = None,
) -> int
    # 真实 spam、预测 ham；列 Index, Label, Pred, Text；返回写入条数

format_classification_eval_lines(...) -> list[str]
    # 供脚本打印混淆表格与 PRF 行
```

单条推理、`encode_spam_text_for_model`、`load_spam_classifier_checkpoint`、CLI 与 checkpoint **读取**契约见 **[REQ-P2-03](REQ-P2-03_ClassifySmsInfer.md)**。

### 张量契约

| 阶段 | 张量 | 形状 | 类型 |
|------|------|------|------|
| DataLoader 输出 | token_ids batch | `[B, max_length]` | int64 |
| DataLoader 输出 | labels batch | `[B]` | int64 |
| 模型前向 | logits（分类后） | `[B, max_length, num_classes]` | float32 |
| 取最后 token | logits[:, -1, :] | `[B, num_classes]` | float32 |
| loss | cross_entropy | 标量 | float32 |

### finetune_classify.py 入口

```
uv run python finetune_classify.py --config configs/config_classify_spam.json
```

可选覆盖：`--checkpoint <path>`（覆盖配置中的 `pretrained_checkpoint`）。

### 输出

- `runs/spam_classify/checkpoint_best.pt`：最高 val accuracy；写入字段含 **`spam_max_length`**、**`pad_token_id`** 等（完整列表见 [REQ-P2-03](REQ-P2-03_ClassifySmsInfer.md) §4）。
- 训练日志：每 eval_freq 步打印 train/val loss，每 epoch 末打印 accuracy。
- **训练结束后（§11）**：从磁盘 **重新加载 best checkpoint 权重** 到内存（避免「最后一 epoch」与「选模权重」不一致），在 **test** 上打印 **混淆矩阵**、**spam/ham 的 P/R/F1**，并将漏判样本导出为 **`runs/<run_name>/test_false_negative_spam.csv`**（真实 spam、预测 ham）。

### eval_classify.py（不重训复评）

```
uv run python eval_classify.py --checkpoint runs/spam_classify/checkpoint_best.pt
```

- 默认 `test.csv`：checkpoint 内 `finetune_config.data.data_dir` + `/test.csv`（相对仓库根解析）。
- 默认 FN 输出：`checkpoint` 同目录下的 **`eval_false_negative_spam.csv`**。
- 可选：`--test-csv`、`--fn-out`、`--batch-size`、`--max-fn-rows`、`--device`。

### 探针清单（人工 / 脚本回归）

- [`docs/probes/classify_spam_probes.json`](../docs/probes/classify_spam_probes.json)：固定英文句 + `expected` ∈ `{ham,spam}`；**不保证**与任意 checkpoint 一致，用于记录期望与防回归对照（详见 [DOMAIN-KNOWLEDGE §6.6](DOMAIN-KNOWLEDGE.md)）。

---

## 5. 业务规则

| # | 规则 | 说明 |
|---|------|------|
| R1 | 数据平衡 | 下采样 ham 到与 spam 同数量（各 747 条），random_state=123 |
| R2 | 切分比例 | 70% train / 10% val / 20% test，打乱后切分 |
| R3 | 编码 | tiktoken GPT-2 BPE，与预训练阶段一致 |
| R4 | 填充 | 不足 max_length 的序列用 pad_token_id=50256 填充到右侧 |
| R5 | 截断 | 超过 max_length 的序列截断到 max_length |
| R6 | max_length 对齐 | train 集 max_length=None 时自动取最长序列；val/test 使用 train 的 max_length |
| R7 | 冻结策略 | 先冻结全部 → 替换 out_head → 解冻最后 N 个 Transformer block + final_norm |
| R8 | 分类取值 | `model(batch)[:, -1, :]`，取最后一个 token 的 logits |
| R9 | 损失函数 | `cross_entropy(logits[:, -1, :], labels)` |
| R10 | 标签映射 | ham → 0, spam → 1 |

---

## 6. 验收标准

| # | 输入 | 预期输出 |
|---|------|---------|
| AC1 | `SpamDataset(train.csv)` 的每个样本 | `token_ids.shape == [max_length]`，`label ∈ {0, 1}` |
| AC2 | `SpamDataset` 中 pad 的位置 | `token_ids[原始长度:] == pad_token_id` |
| AC3 | 训练 5 step，loss | 有限实数，非 NaN |
| AC4 | `calc_accuracy_loader` | 返回 float ∈ [0.0, 1.0] |
| AC5 | 完整 5 epoch 训练 | test accuracy >= 90%（我们用自训练权重，预期略低于书上的 95%+） |
| AC6 | checkpoint 文件 | `runs/spam_classify/checkpoint_best.pt` 存在 |
| AC7 | `finetune_classify.py` 正常结束且存在 best | stdout 含 **test** 混淆矩阵（TN/FP/FN/TP）及 spam/ham **P/R/F1** |
| AC8 | 同上 | `runs/<run_name>/test_false_negative_spam.csv` 存在（FN 可为 0 行时文件仍写出表头） |
| AC9 | `eval_classify.py` + 有效分类 checkpoint + 现有 `test.csv` | 退出码 0；打印与 AC7 同结构的指标行 |
| AC10 | `pytest tests/test_classify_metrics.py` | 全部通过（混淆计数、PRF、`collect_predictions`、`export_fn`、探针 JSON schema） |

---

## 7. 配置依赖

| config.json 字段 | 用途 |
|-------------------|------|
| `pretrained_checkpoint` | 预训练 checkpoint 路径 |
| `data.data_dir` | SMS 数据下载/缓存目录 |
| `finetune.num_classes` | 分类类别数（2） |
| `finetune.num_epochs` | 微调轮数 |
| `finetune.batch_size` | 批大小 |
| `finetune.learning_rate` | AdamW 学习率 |
| `finetune.weight_decay` | AdamW 权重衰减 |
| `finetune.eval_freq` | 每 N 步评估 loss |
| `finetune.eval_iter` | 评估时取多少 batch |
| `finetune.max_length` | null=自动取最长 |
| `finetune.unfreeze_last_n_blocks` | 解冻的末尾 Transformer block 数 |

---

## 8. 代码落位与依赖

| 项 | 路径 |
|----|------|
| 模块实现 | `src/mini_llm/m06_classify_finetune/__init__.py` |
| 微调脚本 | `finetune_classify.py` |
| 离线评估脚本 | `eval_classify.py`（混淆矩阵 / PRF1 / FN CSV） |
| 配置文件 | `configs/config_classify_spam.json` |
| 推理脚本 | 见 [REQ-P2-03](REQ-P2-03_ClassifySmsInfer.md) · `classify_sms.py` |
| 探针数据 | `docs/probes/classify_spam_probes.json` |
| 测试 | `tests/test_classify_finetune.py`、`tests/test_classify_metrics.py`；推理侧见 REQ-P2-03 |
| 依赖模块 | `mini_llm.m04_model`（GPTModel）、tiktoken（经 `SpamDataset`） |
| 依赖库 | `torch >= 2.0.0`、`tiktoken`、`pandas`（读 CSV） |

---

## 9. 与教程第六章（书本）实现的差异与可选补充

### 9.1 核心算法（一致）

以下与第六章 **一致**：SMS Spam 数据来源与平衡方式（下采样 ham）；70/10/20 划分；固定长度 padding（50256）；**冻结大部分参数 → 替换 `out_head` 为 `num_classes` → 解冻最后一层 Transformer + `final_norm`**；用 **最后一个 token** 的 logits 做 `cross_entropy`；AdamW（lr、weight_decay 与章节脚本同量级）；epoch 末打印 accuracy。

### 9.2 实现与环境（不同）

| 方面 | 书本 ch06（`gpt_class_finetune.py`） | 本项目 |
|------|--------------------------------------|--------|
| **预训练权重从哪来** | 下载 OpenAI GPT-2 checkpoint，`load_weights_into_gpt` 灌进模型 | 读取 **`train.py` 产出的 `.pt`**（`model_state_dict` + `config`），与全书前文 **同一套 `GPTModel`** |
| **架构细节** | 官方小 GPT-2：`qkv_bias=True`，常见实现带 **weight tying**（书中加载逻辑相关） | 本仓库：`qkv_bias=False`，**无 tying**（`tok_emb` / `out_head` 独立）——与 [`m04_model`](../src/mini_llm/m04_model/__init__.py) 一致 |
| **参数量级** | 约 124M（官方 Small） | 与本仓库预训练 **同一配置**（脚本会打印 `total`；体量与 GPT-2 Small 同级，具体数以 checkpoint 内 `config.model` 为准） |
| **设备** | 章节脚本默认 CUDA / CPU | **`device: auto`**：CUDA → MPS → CPU（适配 Apple Silicon） |
| **代码形态** | 单文件内_dataset / train / plot 混写 | **`m06_classify_finetune`**（数据与 metric）+ **`finetune_classify.py`**（编排）+ **JSON 配置** + **`pytest`** |
| **训练过程产物** | Matplotlib 保存 loss/accuracy 曲线 PDF | **不打曲线**；按 **验证集 accuracy** 存 `checkpoint_best.pt`，末尾报告 **test accuracy** |
| **超参灵活性** | 解冻层数固定为 **末 1 块** | `finetune.unfreeze_last_n_blocks` **可配置**（默认 1，与书一致） |
| **SpamDataset** | 构造时 **传入 `tokenizer`** | 模块内 **`tiktoken.get_encoding("gpt2")`**（效果等价，接口略简） |
| **可选 `--test_mode`** | 极小随机模型 + CPU，便于仓库自测 | **未实现**；依赖 **`tests/test_classify_finetune.py`** 做小模型 smoke |

### 9.3 与 §10 的关系

§9.1–§9.2 描述与第六章的差异；**书上弱或未写、但有价值的增强**统一记在 **§10 可选增强 backlog**。**BL-P2-02-02（混淆矩阵、PRF、FN 导出）已落地**，契约与术语定义见 **§11**。

---

## 10. 可选增强 backlog（有时间再实现）

以下 **不属于** 当前 P2-02/P2-03 的验收范围；实现后请在本表更新 **状态**、并视情况增加 `HARNESS` / `SPEC` 条目。**BL-P2-02-02** 的矩阵定义、公式与 best-checkpoint 行为见 **§11**。

| ID | 主题 | 说明 | 建议落点 | 状态 |
|----|------|------|----------|------|
| **BL-P2-02-01** | 单条推理 CLI | 读一行英文短信 → stdout `ham`/`spam`，便于演示 | [REQ-P2-03](REQ-P2-03_ClassifySmsInfer.md) · [`classify_sms.py`](../classify_sms.py) | **已完成** |
| **BL-P2-02-02** | 分类指标扩展 | 除 accuracy 外报告 **spam** 的 precision / recall / F1；打印 **混淆矩阵**（TN/FP/FN/TP）；可选写入 `finetune_classify.py` 结束阶段或独立 `eval_classify.py` | `m06`：`collect_predictions_loader` / `confusion_counts_binary_spam` / `prf1_from_counts` / `export_false_negative_spam_csv`；[`finetune_classify.py`](../finetune_classify.py) 训练结束加载 **best val** 权重后打印并导出 `test_false_negative_spam.csv`；[`eval_classify.py`](../eval_classify.py) 仅评估 | **已完成** |
| **BL-P2-02-03** | 预训练起点对照实验 | 在 **同一套 `GPTModel`** 上对齐加载 **OpenAI GPT-2 Small**（bias/tying 等与本书差异需单独处理），再跑 **同一套** SMS 微调流程，对比 test 指标 | 独立实验分支或 `docs/` 实验笔记；非主线 | todo |
| **BL-P2-02-04** | 不平衡损失权重 | **保留全量 ham**，不设平衡集；使用 `CrossEntropyLoss(weight=…)`（按类频次反比或其它配方），与当前 **下采样 ham** 做对照 | `SpamDataset`/`download_and_prepare_spam` 可选模式 + `finetune_classify.py` | todo |
| **BL-P2-02-05** | CI / 冒烟 `--smoke` | 无大 checkpoint、无下载数据时：**极小随机 `GPTModel`** + 伪造 batch，跑固定 **N 步** optimizer step，断言 loss 有限（对齐书本 `test_mode`） | `finetune_classify.py --smoke` 或独立 pytest + `@pytest.mark.smoke` | todo |

**备注**

- **BL-P2-02-02** 动机：上线场景往往 **漏判 spam（FN）** 比误判 ham 更敏感；平衡集上 accuracy 容易「好看」，仍需看 spam 一侧指标。
- **BL-P2-02-03** 工程量偏大，仅建议在主线稳定后作为拓展。
- **BL-P2-02-05** 可与现有 `tests/test_classify_finetune.py` 小模型用例互补：冒烟侧重 **端到端训练循环** 而非仅 Dataset。

本节取代原 §9.3 列表；验收仍以 REQ §6 与 [`HARNESS.md`](../HARNESS.md) 为准。

---

## 11. 二分类评估落地说明（BL-P2-02-02）

本节约定 **标签**：**ham = 0**，**spam = 1**（与 CSV、`classify_sms` 一致）。

### 11.1 混淆矩阵（二分类）

在 **真实标签为行、预测标签为列**（列为 pred_ham / pred_spam）时：

|  | pred ham (0) | pred spam (1) |
|--|--------------|---------------|
| **true ham (0)** | TN | FP |
| **true spam (1)** | FN | TP |

- **TN**：真 ham 判 ham（正确拒绝误判）。
- **FP**：真 ham 判 spam（**误判骚扰**：正常短信被当成垃圾）。
- **FN**：真 spam 判 ham（**漏判**：垃圾短信未被拦下）；过滤场景往往 **业务代价更高**。
- **TP**：真 spam 判 spam（正确拦截）。

与 **`sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0,1])`** 在 `labels` 顺序为 `[0,1]` 时的二维数组布局一致（左上 TN，右上 FP，左下 FN，右下 TP）。

### 11.2 Precision / Recall / F1（按类）

记 spam 为 **正类**（检出目标）时：

- **Precision_spam** = TP / (TP + FP)：在所有「被判成 spam」的样本里，有多少真是 spam。
- **Recall_spam** = TP / (TP + FN)：在所有「真是 spam」的样本里，有多少被抓到。
- **F1_spam** = 2 · Precision_spam · Recall_spam / (Precision_spam + Recall_spam)（调和平均；任一为 0 则需谨慎解读）。

对 **ham** 可把 ham 视作正类对称定义（实现中为 TN / (TN+FN)、TN / (TN+FP) 等，见 `prf1_from_counts` 源码）。

**Accuracy** = (TN + TP) / (TN + FP + FN + TP)。在 **平衡测试集** 上 accuracy 与两类错误「可比」，但仍应用 **Recall_spam** 单独看守漏。

### 11.3 为何最终报告加载 **best val** 权重

训练循环在每个 epoch 末可能更新「当前」权重；**磁盘上的 best** 按 **验证集 accuracy** 写入，不一定等于 **最后一个 epoch**。若在最后一 epoch 权重上算 test，会与「实际交付的 checkpoint」不一致。故 **`finetune_classify.py`** 在打印混淆矩阵与 FN CSV **之前**从 `checkpoint_best.pt` **reload** `model_state_dict`。

### 11.4 与单条推理的关系

- **`classify_sms.py`**：一条文本 → stdout `ham`/`spam`；`--probs` 给 softmax 观感（stderr）。
- **`eval_classify.py` / 训练末段**：整表 **test.csv** → 全局 TN/FP/FN/TP 与 PRF；**FN CSV** 便于批量审视漏判话术。

**更细的直觉、陷阱与探针用途**见 [DOMAIN-KNOWLEDGE.md](DOMAIN-KNOWLEDGE.md) **§6.6**。
