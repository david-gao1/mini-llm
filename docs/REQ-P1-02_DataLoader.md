# REQ-P1-02：滑动窗口 Dataset 与 DataLoader

**所属**：[SPEC.md](../SPEC.md) → Part I · 模块 02  
**依赖**：[REQ-P1-01](REQ-P1-01_Tokenizer.md)（encode_text 编码语料）  
**被依赖**：[REQ-P1-05](REQ-P1-05_Train.md)（训练循环消费 DataLoader）  
**状态**：✅ 已完成

---

## 1. 业务逻辑（读完就知道「要干嘛」）

### 先打个比方

一整段语料像一卷超长磁带；训练语言模型时不能整条塞进显卡。**滑动窗口**就像用固定长度的窗口在带上滑动：窗口里是「已经看到的词」，模型要学的是 **紧接着的下一个词**。窗口每次挪动（步长由 `stride` 决定），就做出成千上万道「填空题」。

### 最关键的一句话

> **把长文本切成许多 `(input, target)` 对**：`target` 等于 `input` 向右挪一格——这就是「猜下一个词」的监督信号；再加上 **语料从哪读**、**train/val 怎么切**。没有这一块，`train.py` 拿不到批次。

### 这条 REQ 交付什么

| | |
|--|--|
| **切分** | 定长窗口、`stride`、`context_length` 对齐 |
| **加载** | 环境变量 → 同级书仓库 → URL 下载缓存 |
| **划分** | 按比例切成训练 / 验证，便于看过拟合 |

---

## 2. 设计思路（怎么做）

**方案**：滑动窗口 tokenize + PyTorch Dataset/DataLoader。

**为什么先 tokenize 整段再滑窗，而非逐条 tokenize**：
- 整段 tokenize 保证 BPE 合并跨越窗口边界时不丢失信息
- 一次 encode 调用比 N 次快得多

**为什么按字符比例切分而非按 token 或按样本**：
- 字符切分简单可靠，不需要先 tokenize 全量语料再分割
- 对于短语料（如 the-verdict.txt），差异可忽略

**语料加载三级回退设计**：
- 开发者可能在离线环境、有同级书本仓库、或需要联网——三级回退覆盖全场景
- 环境变量 → 同级仓库探测 → URL 下载并缓存，优先级从确定到不确定

**关键设计决策**：
- `stride = context_length`（训练时无重叠），避免数据泄漏
- `drop_last=True` 在训练集避免不完整 batch 干扰梯度
- 文本不足一个完整窗口时产生空 Dataset，不报错（由 train.py 检查）

---

## 3. 架构定位（在哪里）

```text
     ┌─────────────────────────────────────────────────────┐
     │  语料来源                                             │
     │  ① TEAM_LLM_DATA_DIR 环境变量                        │
     │  ② LLMs-from-scratch/ch02/.../the-verdict.txt       │
     │  ③ config.data.url 下载到 runs/<run>/data_cache/    │
     └────────────────────┬────────────────────────────────┘
                          │ load_text()
                          ▼
     ┌─────────────────────────────────────────────────────┐
     │  m02_data_loader                                     │
     │   load_text() → 原始文本                              │
     │   train_val_dataloaders() → (train_loader, val_loader)│
     │     └── GPTDataset: 滑动窗口 (input[T], target[T])   │
     │     └── create_dataloader: 封装 DataLoader            │
     └────────────────────┬────────────────────────────────┘
                          │
                          ▼
               ┌──────────────────┐
               │  train.py 训练循环 │
               │  for batch in ..  │
               └──────────────────┘
```

**上游**：原始语料文件（本地 / 网络）  
**下游**：train.py 训练循环

---

## 4. 输入 / 输出契约

### GPTDataset

```python
class GPTDataset(Dataset):
    def __init__(self, txt: str, max_length: int, stride: int) -> None
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]  # (input[T], target[T])
```

- 每条样本：`input[T]` 和 `target[T]`，均为 `torch.long`
- batch 后形状：`[B, T]`

### load_text

```python
load_text(data_cfg: dict, cache_dir: Path | None = None) -> str
```

### create_dataloader / train_val_dataloaders

```python
create_dataloader(text, batch_size, max_length, stride, shuffle, drop_last, num_workers=0) -> DataLoader
train_val_dataloaders(full_text, train_ratio, model_cfg, train_cfg) -> tuple[DataLoader, DataLoader]
```

---

## 5. 业务规则

| # | 规则 | 说明 | 示例 |
|---|------|------|------|
| R1 | 滑动窗口 | `input = ids[i:i+T]`，`target = ids[i+1:i+T+1]` | target 是 input 右移 1 位 |
| R2 | stride 控制重叠 | stride < max_length → 窗口有重叠；stride = max_length → 无重叠 | 训练时 stride = context_length |
| R3 | 短文本安全 | 文本不足一个完整窗口 → 空 Dataset（len=0），不抛异常 | 10 个 token，窗口 64 → 空 |
| R4 | 加载优先级 | 环境变量 → 同级书本仓库 → URL 下载缓存 | 三级 fallback |
| R5 | 下载缓存 | URL 下载后写入 `cache_dir`，后续离线复用 | `runs/team_gpt/data_cache/` |
| R6 | 字符切分 | `split_idx = int(train_ratio * len(text))` | 90% 训练 / 10% 验证 |
| R7 | 训练集 shuffle | `shuffle=True, drop_last=True` | 随机打乱，丢弃不完整 batch |
| R8 | 验证集不 shuffle | `shuffle=False, drop_last=False` | 确保评估一致性 |
| R9 | ID 范围 | 所有 token id ∈ [0, vocab_size) | 由 m01 tokenizer 保证 |

---

## 6. 验收标准

| # | 输入 | 预期输出 |
|---|------|---------|
| AC1 | 构造 GPTDataset，取一条样本 | `input.shape == target.shape == (T,)` |
| AC2 | 检查 input 与 target 关系 | `target == input` 右移 1 位（`input[1:] == target[:-1]`） |
| AC3 | DataLoader batch | `input_batch.shape == (B, T)` |
| AC4 | 所有 token id | `0 <= id < 50257` |
| AC5 | stride=1 vs stride=T | stride=1 产生更多样本 |
| AC6 | 文本仅 5 个 token，max_length=64 | `len(dataset) == 0` |
| AC7 | train_val split，train_ratio=0.9 | 两个 loader 均非空，数据不重叠 |
| AC8 | 包可导入 | `import mini_llm` 成功 |

单测 8 用例，全部 ✅。

---

## 7. 代码落位与依赖

| 项 | 路径 |
|----|------|
| 模块实现 | `src/mini_llm/m02_data_loader/__init__.py` |
| 测试 | `tests/test_data_loader.py`、`tests/test_imports.py` |
| 依赖模块 | `mini_llm.m01_tokenizer`（encode_text） |
| 依赖库 | `torch >= 2.0.0` |
| 默认语料 | `the-verdict.txt`（书本 ch02 示例短篇） |
