# REQ-P2-03：classify_sms——输入一行英文短信，stdout 输出 `ham` 或 `spam`（分类 checkpoint 须先由 P2-02 训好）

**所属**：[SPEC.md](../SPEC.md) → Part II · 推理脚本（分类）  
**依赖**：[REQ-P2-02](REQ-P2-02_ClassifyFinetune.md)（`finetune_classify`：SMS ham/spam 分类 checkpoint）、[REQ-P1-04](REQ-P1-04_Model.md)（GPTModel）  
**被依赖**：无  
**状态**：✅ 已完成  
**可选后续（共享 backlog）**：[REQ-P2-02 §10](REQ-P2-02_ClassifyFinetune.md)（含 **`--json`** 等推理增强；以及 **BL-P2-02-03**：「换一套预训练底座再微调 SMS」的对照实验——**解释见 REQ-P2-02 §10.1**，与本 REQ 的推理契约无关）；**批量指标**见 **`eval_classify.py`** 与 REQ-P2-02 **§11**。

---

## 1. 业务逻辑（读完就知道「要干嘛」）

### 先打个比方

**工厂已经把滤波器做好了（训练存的 `.pt`）。**本 REQ 只是门口的检测仪：**塞进一条英文短信 → 亮绿灯(ham)或红灯(spam)**——不负责召回流水线重做滤网。

### 最关键的一句话

> **给定磁盘上的短信分类 checkpoint**，读一行英文短信，按训练同一规则编码，在 **stdout 打印一行 `ham` 或 `spam`**；若要概率可以看 stderr。**不负责**下载或对齐别的预训练权重（那是 [REQ-P2-02 §10](REQ-P2-02_ClassifyFinetune.md) backlog）。

上游 [REQ-P2-02](REQ-P2-02_ClassifyFinetune.md) 产出 **带分类头的权重**。业务上要的是 **判别**：该不该拦这条短信，不是写诗式的续写。

**为什么单独叫 P2-03**：微调管「怎么训」；这里只管「盘上已有模型时怎么加载、怎么编码、怎么出标签」。它和 [`generate_from_checkpoint.py`](../generate_from_checkpoint.py) 并排：**生成** vs **短信二分类**。

**边界**：换预训练底座等对照见 [REQ-P2-02 §10.1](REQ-P2-02_ClassifyFinetune.md)；训完仍是普通 `.pt`，本 REQ **只管推理读入**。

---

## 2. 设计思路（怎么做）

**方案**：根目录脚本 **`classify_sms.py`** + `m06_classify_finetune` 内 **`encode_spam_text_for_model`** / **`load_spam_classifier_checkpoint`**（与 `SpamDataset` **同一套** GPT-2 BPE、截断、右填充）。

**为什么必须把 `spam_max_length` 写进 checkpoint**（由 `finetune_classify.py` 保存）：训练时所有样本 pad 到 train 集最长长度 `T`；推理若用别的 `T`，最后一个 token 位置的 pad 分布与训练不一致，准确率不可预期。

**旧 checkpoint 无 `spam_max_length` 时**：脚本退回 `model.context_length` 并 **stderr 警告**，或用户显式 **`--max-length`**。

**为什么 `--probs` 打到 stderr**：stdout 保持「单行标签」，便于 shell 管道把类别传给下游；概率仅供人工查看。

**关键设计决策**：
- **默认 checkpoint（演示 / 少漏判 spam）**：`runs/spam_classify_phase_b/checkpoint_best.pt`（须先按 [`REPORT_ClassifySpamProbe.md`](REPORT_ClassifySpamProbe.md) §7.3 跑阶段 B；见对照结论 §7.3「对照实验说明了什么」）。仅跑 [`configs/config_classify_spam.json`](../configs/config_classify_spam.json) 时产出为 `runs/spam_classify/checkpoint_best.pt`（基线目录）。
- 设备：`auto` → CUDA → MPS → CPU（与 `train.py` 一致）
- 标签打印名：`0 → ham`，`1 → spam`（与 P2-02 CSV 映射一致）

---

## 3. 架构定位（在哪里）

```text
     classify_sms.py
           │
           ├─ load_spam_classifier_checkpoint → GPTModel + 分类 out_head
           ├─ encode_spam_text_for_model → [1, max_length]
           └─ model(batch)[:, -1, :] → argmax / softmax
```

**上游**：`finetune_classify.py` 写出的 `.pt`（非预训练 LM checkpoint）  
**下游**：stdout / stderr（无额外模块）

---

## 4. 输入 / 输出契约

### m06_classify_finetune（推理用）

```python
encode_spam_text_for_model(text: str, max_length: int, *, pad_token_id=50256) -> Tensor
    # 形状 [1, max_length]；与 SpamDataset 单行编码一致

load_spam_classifier_checkpoint(path: Path | str, device: torch.device) -> tuple[nn.Module, dict]
    # meta：spam_max_length | None, pad_token_id, num_classes, best_val_accuracy, finetune_config
```

### classify_sms.py CLI

| 参数 | 说明 |
|------|------|
| `--checkpoint` | 分类 checkpoint；**省略时默认** `runs/spam_classify_phase_b/checkpoint_best.pt`（须已训练）；基线对照用 `runs/spam_classify/checkpoint_best.pt` |
| `--text` | 一条短信；省略则从 **stdin 读一行** |
| `--device` | `auto` \| `cpu` \| `cuda` \| `mps` |
| `--max-length` | 覆盖 checkpoint 中的序列长度 |
| `--probs` | 将各类 softmax 概率打印到 **stderr** |

**stdout**：单行 `ham` 或 `spam`（或未来 `num_classes>2` 时的数字 id 名）。

### 分类 checkpoint 字段（读取侧契约）

与 P2-02 写入一致，推理脚本至少依赖：

| 键 | 含义 |
|----|------|
| `model_state_dict` | 含分类 `out_head` 的完整权重 |
| `config` | 内含 `model` 子字典（架构） |
| `num_classes` | 输出维度 |
| `spam_max_length` | **推荐**存在；与训练 pad 长度一致 |
| `pad_token_id` | 默认 50256 |

可选：`finetune_config`、`best_val_accuracy`、`epoch`、`global_step`。

### 张量契约

| 步骤 | 形状 |
|------|------|
| `encode_spam_text_for_model` | `[1, max_length]` |
| `model(batch)` | `[1, max_length, num_classes]` |
| 决策用 logits | `[1, num_classes]`（取 `[:, -1, :]`） |

---

## 5. 业务规则

| # | 规则 | 说明 |
|---|------|------|
| R1 | 编码一致 | 与 P2-02 `SpamDataset` 相同 tiktoken、gpt2 |
| R2 | 长度一致 | 默认使用 checkpoint `spam_max_length` |
| R3 | 非预训练 ckpt | `load_state_dict` 需匹配分类头形状；误传 LM-only ckpt 会报错 |
| R4 | 语言 | 数据为英文短信；非英文 token 行为未保证 |
| R5 | eval | `model.eval()` + `torch.no_grad()` |

---

## 6. 验收标准

| # | 输入 | 预期输出 |
|---|------|---------|
| AC1 | `encode_spam_text_for_model` vs 同行 `SpamDataset` | 张量逐元素一致 |
| AC2 | 最小伪造分类 checkpoint + `load_spam_classifier_checkpoint` | 前向 `[1, num_classes]` 有限 |
| AC3 | `classify_sms.py --help` | 退出码 0 |
| AC4 | 有效分类 checkpoint + `--text "..."` | stdout 单行 `ham` 或 `spam` |

---

## 7. 代码落位与依赖

| 项 | 路径 |
|----|------|
| 推理脚本 | [`classify_sms.py`](../classify_sms.py) |
| 模块函数 | [`src/mini_llm/m06_classify_finetune/__init__.py`](../src/mini_llm/m06_classify_finetune/__init__.py) |
| 测试 | [`tests/test_classify_finetune.py`](../tests/test_classify_finetune.py)（`test_encode_*`、`test_load_*`） |
| SPEC | [`SPEC.md`](../SPEC.md) §P2-03 |

---

## 8. 与 P2-02 的分工

| P2-02 | P2-03 |
|-------|-------|
| 下载数据、`SpamDataset`、训练循环、`checkpoint_best.pt` **写入**；**eval_classify.py**、训练末段在 **test** 上输出混淆矩阵 / PRF / FN CSV（REQ-P2-02 **§11**） | checkpoint **读取**、单条推理 CLI、`encode`/`load` 契约 |
