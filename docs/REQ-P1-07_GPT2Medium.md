# REQ-P1-07：GPT-2 Medium 规模训练 + WikiText 语料

**所属**：[SPEC.md](../SPEC.md) → Part I · 模型与数据升级  
**依赖**：[REQ-P1-06](REQ-P1-06_TrainOptimize.md)（训练优化已到位：MPS / scheduler / early stopping）  
**被依赖**：无  
**状态**：进行中（WikiText-2 已完成 → WikiText-103 训练中）  
**OpenSpec（行为契约）**：[预训练 · `pretraining/spec.md`](../openspec/specs/pretraining/spec.md)（Medium 完成见该文件 **路线图（P1-07）**）

---

## 1. 业务逻辑（读完就知道「要干嘛」）

### 先打个比方

教学demo用的是「玩具显微镜」：**很小的模型 + 极短语料**，为的是先证明链路能转。要进一步看清「真的在学语言」，就要换成 **更大的 GPT（Medium）+ 维基级语料**——数据多到不可能背完，曲线才像正经实验。

### 最关键的一句话

> **把默认小模型 + 短文**，升级到 **GPT-2 Medium（约 4 亿参数量级）+ WikiText（先 -2 再 -103 raw）**，稳定下载数据、调好 batch/梯度累积，目标是一份 **更像样的预训练 checkpoint**，给后面分类 / 指令微调当底座。

### 分两步走（白话）

1. **WikiText-2**：先把 Medium 架构和管线跑顺，暴露数据量瓶颈。  
2. **WikiText-103（raw）**：用更大的正文压住过拟合，避免 `<unk>` 噪音。

当前进度与数值见文内 §2 及 [`TRAINING_LOG.md`](TRAINING_LOG.md)。

---

## 2. 设计思路（怎么做）

**方案**：小幅改动 `load_text` 支持 HuggingFace 数据源 + 新配置文件。

**阶段一 WikiText-2**：
- 零代码改动，仅配置文件，URL 直接下载 train.txt
- batch_size=1, num_epochs=10, patience=20
- 目的：快速验证架构 + 暴露瓶颈

**阶段二 WikiText-103（当前）**：
- `load_text` 新增 `source="huggingface"` 模式，通过 HuggingFace `datasets` 库下载
- 使用 **raw 版本**（`wikitext-103-raw-v1`）：原始文本，无 `<unk>` 预处理产物
- batch_size=1 + gradient_accumulation_steps=4（等效 batch=4，避免 OOM），num_epochs=3（大语料不需要多轮）

**为什么从 WikiText-2 升级到 WikiText-103**：
- WikiText-2 训练结果（Run 2）显示 Epoch 5 后 train/val gap 达 2.0，**数据量是核心瓶颈**
- WikiText-103 数据量 50 倍，参数/数据比从 ~160:1 改善为 ~3:1，大幅缓解过拟合
- 使用 raw 版本消除 `<unk>` 污染问题

**为什么用 HuggingFace 而非 URL 直下**：
- WikiText-103 原始 S3 托管链接已失效（403 Forbidden）
- HuggingFace Datasets 是标准替代方案，自动下载 + 缓存 + 版本管理
- 下载后转存为本地 txt，后续训练离线复用

**关键设计决策**：
- `batch_size=1` + `gradient_accumulation_steps=4`（等效 batch=4，但峰值内存降至 ~8-10 GB）
- `num_epochs=3`（500MB 语料 × 3 轮，大模型预训练通常 1-2 轮即可）
- `eval_freq=2000`（步数多，降低评估频率）
- `patience=20`（WikiText-2 实验验证的合理值）
- `drop_rate=0.1`（大语料过拟合风险低）

> **内存优化说明**：原 `batch_size=4` 配置在 M3 Max 36GB 上实测内存溢出（终端进程占用 55GB+，系统 swap 严重）。
> 改用 `batch_size=1` + 梯度累积 4 步，数学上等价（相同的梯度均值），但峰值激活值内存降低 4 倍。

---

## 3. 架构定位（在哪里）

```text
     ┌─────────────────────────────────────────────────────────┐
     │  零代码改动                                               │
     │                                                         │
     │  m01-m05 / train.py  →  完全复用，不动一行                 │
     └─────────────────────────────────────────────────────────┘

     ┌─────────────────────────────────────────────────────────┐
     │  新增文件                                                │
     │                                                         │
     │  configs/config_medium.json   ← 唯一新增                 │
     │    ├─ data.url → WikiText-2 train.txt (PyTorch 仓库)     │
     │    ├─ model → GPT-2 Medium 架构 (1024/16/24)            │
     │    └─ train → 适配大模型的超参                             │
     │                                                         │
     │  runs/gpt2_medium_wikitext2/  ← 训练时自动创建            │
     │    ├─ data_cache/wikitext2_train.txt                     │
     │    ├─ checkpoint_latest.pt                               │
     │    └─ checkpoint_best.pt                                 │
     └─────────────────────────────────────────────────────────┘
```

**上游**：WikiText-2 train.txt（PyTorch examples 仓库托管）  
**下游**：训练产出的 checkpoint 可供 m05 generate 加载生成

---

## 4. 输入 / 输出契约

### 输入

```bash
uv run python train.py --config configs/config_medium.json
```

> 终端后台运行建议（避免日志缓冲与工作目录漂移）：
>
> ```bash
> nohup env PYTHONUNBUFFERED=1 "/abs/path/to/team-mini-llm/.venv/bin/python" -u "/abs/path/to/team-mini-llm/train.py" --config "/abs/path/to/team-mini-llm/configs/config_medium.json" > "/abs/path/to/team-mini-llm/train_wt103.log" 2>&1 &
> tail -f "/abs/path/to/team-mini-llm/train_wt103.log"
> ```

### 数据源

| 项 | WikiText-2（阶段一，已完成） | WikiText-103 raw（阶段二，当前） |
|----|---|---|
| 数据集 | WikiText-2 tokenized | WikiText-103 raw v1 |
| 来源 | PyTorch examples URL 直下 | HuggingFace `datasets` 库 |
| HF 路径 | — | `Salesforce/wikitext` / `wikitext-103-raw-v1` |
| 文件大小 | ~10 MB | **~500 MB** |
| token 数 | ~2.5M | **~130M** |
| `<unk>` 问题 | 有（tokenized 版本低频词被替换） | **无（raw 版本保留原始文本）** |

### 模型配置

```json
{
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": false
}
```

### 模型参数量详解（~406M）

| 组件 | 形状 | 参数量 | 说明 |
|------|------|-------:|------|
| **Token Embedding** | 50257 × 1024 | 51,463,168 | 每个 token 映射为 1024 维向量；50257 = GPT-2 BPE 词表大小 |
| **Position Embedding** | 1024 × 1024 | 1,048,576 | 每个位置（0~1023）一个 1024 维向量，注入序列位置信息 |
| **TransformerBlock × 24 层** | | **302,088,192** | |
| &emsp;W_query | 1024 × 1024 | 1,048,576 | 将输入投影为 Query，拆分到 16 个头（每头 64 维） |
| &emsp;W_key | 1024 × 1024 | 1,048,576 | 将输入投影为 Key，与 Query 做点积计算注意力权重 |
| &emsp;W_value | 1024 × 1024 | 1,048,576 | 将输入投影为 Value，被注意力权重加权求和 |
| &emsp;out_proj | 1024 × 1024 | 1,048,576 | 多头合并后的线性投影，恢复维度 |
| &emsp;FFN up | 1024 × 4096 | 4,194,304 | 前馈网络上投影，扩展 4 倍（1024→4096），增加表达能力 |
| &emsp;FFN down | 4096 × 1024 | 4,194,304 | 前馈网络下投影，压回原维度（4096→1024） |
| &emsp;LayerNorm × 2 | 1024 × 2 × 2 | 4,096 | 每层两个 LayerNorm，各含 scale + shift 参数 |
| &emsp;**单层小计** | | **12,587,008** | ~12.6M / 层 |
| **Final LayerNorm** | 1024 × 2 | 2,048 | 最后一层归一化的 scale + shift |
| **LM Head (out_head)** | 1024 × 50257 | 51,463,168 | 将隐藏状态映射回词表维度，输出 logits；无 bias |
| | | | |
| **总计** | | **406,065,152** | **~406M 参数** |

> **与标准 GPT-2 Medium（355M）的差异**：标准 GPT-2 的 Token Embedding 和 LM Head **共享权重**（weight tying），节省了 ~51M 参数。本项目为教学清晰起见，两者独立，因此总参数量略大。

### 训练超参详解（WikiText-103 当前配置）

| 参数 | 值 | 对比 WikiText-2 | 说明 |
|------|------|------|------|
| `learning_rate` | 3e-4 | 不变 | AdamW 初始学习率；配合 cosine scheduler 衰减 |
| `weight_decay` | 0.1 | 不变 | 权重衰减正则化 |
| `num_epochs` | **3** | 10→3 | 500MB 语料不需要多轮，1-2 轮已充分 |
| `batch_size` | **1** | 1→1 | 单样本前向，配合梯度累积降低峰值内存 |
| `gradient_accumulation_steps` | **4** | 无→4 | 每 4 个 micro-step 做一次 optimizer.step，等效 batch=4 |
| `eval_freq` | **2000** | 100→2000 | 优化器步数（~31.7K/epoch），降低评估频率 |
| `eval_iter` | 10 | 不变 | 评估时取 10 个 batch 的平均 |
| `checkpoint_every_steps` | **8000** | 500→8000 | 步数多，降低 IO 频率 |
| `grad_clip` | 1.0 | 不变 | 梯度裁剪阈值 |
| `warmup_ratio` | 0.1 | 不变 | 前 10% 步数线性升温 |
| `min_lr_ratio` | 0.1 | 不变 | cosine 衰减下限 = 3e-5 |
| `patience` | 20 | 不变 | WikiText-2 实验验证的合理值 |
| `start_context` | "The history of" | 不变 | 维基百科风格的生成起始文本 |

### 内存估算（M3 Max 36GB，WikiText-103 配置）

| 项 | 估算 | 说明 |
|----|-----:|------|
| 模型参数（float32） | 1.6 GB | 406M × 4 bytes |
| 梯度 | 1.6 GB | 与参数等大 |
| AdamW 状态（动量 + 方差） | 3.2 GB | 参数量 × 2 × 4 bytes |
| 激活值（batch=1, context=1024） | ~2 GB | batch_size=1，梯度累积不增加峰值激活 |
| 数据集张量 | ~0.8 GB | 130M tokens × 8 bytes |
| **总计** | **~9-10 GB** | 36GB 内存充裕 |

> **历史教训**：原 `batch_size=4` 配置实测峰值 55GB+（系统 swap），远超估算的 15-17 GB。
> 原因：float32 下 24 层 Transformer 的中间激活值（注意力矩阵 `1024×1024×16heads`、FFN 中间层等）随 batch 线性增长，
> 且 MPS 后端的内存分配开销大于 CUDA。改为 `batch_size=1` + 梯度累积后问题解决。

### 训练规模估算

| 项 | WikiText-2（已完成） | WikiText-103（当前） |
|----|------|------|
| 训练集 token 数 | ~2.4M | **~130M** |
| 每 epoch 步数 | ~2,200（batch=1） | **~31,700（batch=1, accum=4，优化器步数）** |
| 总步数（全 epoch） | ~22,000（10 ep） | **~95,000（3 ep，优化器步数）** |
| 预计总训练时间 | ~2 小时 | **~20-30 小时**（可能 early stop） |

### 输出

- `runs/gpt2_medium_wikitext2/checkpoint_latest.pt`
- `runs/gpt2_medium_wikitext2/checkpoint_best.pt`
- 训练日志（loss 曲线 + 生成样本）

---

## 5. 业务规则

| # | 规则 | 说明 | 示例 |
|---|------|------|------|
| R1 | 最小代码改动 | `load_text` 新增 `source="huggingface"` 模式，其余 m01-m05 不动 | 向后兼容 |
| R2 | 独立 run 目录 | `run_name="gpt2_medium_wikitext103"`，不覆盖原有实验 | 多套实验并存 |
| R3 | HuggingFace 下载 + 缓存 | 首次运行通过 `datasets` 库下载，转存本地 txt，后续离线复用 | ~500MB |
| R4 | Raw 版本 | 使用 `wikitext-103-raw-v1`（原始文本），不含 `<unk>` 预处理产物 | 消除 unk 污染 |
| R5 | 字符比例 split | 0.95 train / 0.05 val，与现有逻辑一致 | ~475MB / ~25MB |
| R6 | batch_size=1 + grad_accum=4 | 等效 batch=4，峰值内存 ~9-10GB（实测 batch=4 直接 OOM） | 等效 batch 可调 |
| R7 | MPS 加速 | M3 Max 自动选择 MPS（P1-06 已支持） | device="auto" |
| R8 | Early stopping | patience=20，WikiText-2 实验验证的合理值 | 连续 20 次不降则停 |
| R9 | 原有 config 不动 | `configs/config.json` 保持小模型配置 | 可随时回退 |

---

## 6. 验收标准

| # | 场景 | 预期 |
|---|------|------|
| AC1 | 启动训练 | 日志显示 `Device: mps`，`Model parameters: ~355M` |
| AC2 | 数据下载 | `runs/gpt2_medium_wikitext103/data_cache/wikitext103_raw_train.txt` 存在（~500MB） |
| AC3 | 首个 eval | train_loss 接近 ln(50257) ≈ 10.8（随机初始化） |
| AC4 | 训练 2-3 epoch 后 | train_loss 明显下降（< 5），val_loss 同步下降 |
| AC5 | 训练结束 | val_loss 与 train_loss 差距 < 1.0（过拟合可控） |
| AC6 | 生成文本 | 维基百科风格英语段落，无 `<unk>` 污染 |
| AC7 | checkpoint | `checkpoint_latest.pt` 和 `checkpoint_best.pt` 均存在 |
| AC8 | 内存 | 训练过程中不 OOM，峰值 < 20GB |
| AC9 | val_loss | 优于 WikiText-2 实验的 5.44（数据量 50× 应有显著提升） |

---

## 7. 运行记录

### Run 1（2026-04-26）— patience=5，早停过激

| 指标 | 值 |
|------|------|
| 总步数 | ~3,500 / 22,000（16%） |
| 实际 epoch | ~1.5 / 10 |
| 最佳 val_loss | 5.8121（Step 3000） |
| Early stop | Step 3500 触发（patience=5 × eval_freq=100 = 仅容忍 500 步） |
| 生成质量 | 差，大量 `<unk>` 和碎片句子 |

**诊断**：val_loss 在 5.81-5.86 之间正常波动（差距 <0.05），并非真正过拟合。patience=5 对 406M 大模型太紧，导致严重欠训练。

**调整**：`patience` 从 5 → **20**（容忍 2000 步波动），重新运行。

### Run 2（2026-04-27）— patience=20，训练充分，WikiText-2 最终轮

| 指标 | 值 |
|------|------|
| 总步数 | 13,300 / 22,000（60%） |
| 实际 epoch | 6 / 10 |
| 最佳 val_loss | **5.4392**（Step 11300，Epoch 5） |
| 困惑度（perplexity） | e^5.4392 ≈ **230** |
| Early stop | Step 13300 触发（patience=20，Epoch 6 中段） |
| 学习率 | warmup 0→3e-4（Step 0-2200），cosine 衰减至 1.48e-4 |

**val_loss 下降曲线（关键节点）**：

| Step | Epoch | val_loss | 事件 |
|------|-------|----------|------|
| 100 | 1 | 8.2206 | 随机初始化 |
| 1000 | 1 | 6.2070 | warmup 阶段快速下降 |
| 1700 | 1 | 6.0600 | Epoch 1 结束前 |
| 2200 | 1→2 | 6.0412 | warmup 结束，lr 到达峰值 3e-4 |
| 3000 | 2 | 5.8211 | 进入 cosine 衰减 |
| 4500 | 2 | 5.7296 | Epoch 2 结束前 |
| 6800 | 3 | 5.5802 | |
| 9000 | 4 | 5.4942 | |
| 11300 | 5 | **5.4392** | **最佳** |
| 13300 | 6 | 5.4798 | early stop 触发 |

**过拟合分析**：

| Epoch | train_loss | val_loss | gap | 状态 |
|-------|-----------|----------|-----|------|
| 1 | 5.2 - 6.4 | 6.0 - 6.4 | ~0.3 | 健康 |
| 2 | 5.0 - 5.5 | 5.7 - 6.0 | ~0.7 | 轻微 |
| 3 | 4.3 - 4.9 | 5.6 - 5.7 | ~1.0 | 中等 |
| 4 | 4.0 - 4.6 | 5.5 - 5.6 | ~1.2 | 中等 |
| 5 | 3.6 - 4.1 | 5.4 - 5.5 | ~1.4 | 偏大 |
| **6** | **3.3 - 3.8** | **5.47 - 5.56** | **~2.0** | **明显过拟合** |

**生成质量逐 epoch 变化**：

| Epoch | 生成样本（start_context = "The history of"） | 评价 |
|-------|----------------------------------------------|------|
| 1 | `...the <unk> 's <unk> to be not survived . The fort , and the <unk>...` | 有句式，大量 `<unk>` |
| 2 | `...the United States . Astr <unk> , the other and some of the <unk>` | 出现实体名 |
| 3 | `...the area of the first of five years 's death . This is located at...` | 句子结构完整 |
| 4 | `...In the church 's political situation , the late 1669 , and the <unk>...` | 出现年份，有历史语境 |
| 5 | `...the British , including the German positions ; this was appointed to the Royal Society...` | **最佳**，语义连贯 |
| 6 | `...the U.S. state that had been...The U.S. U.S. cru. Army...` | 开始重复，退化 |

**诊断**：
- patience=20 效果良好，模型在合理时机停止
- val_loss=5.44（困惑度 230）是 406M 模型在 ~10MB 数据上从零训练的合理水平
- 核心瓶颈是 **数据量不足**（10MB 对 406M 参数太少，参数/数据比严重失衡）
- `<unk>` 是 WikiText-2 数据集本身的预处理产物（低频词被替换），非模型 bug
- 参考：预训练 GPT-2 Medium 在 WikiText-2 上困惑度 ≈ 22（用了 40GB WebText）

**决策**：切换到 **WikiText-103**（~500MB，50× 数据量），从根本上缓解过拟合。

---

## 8. 代码落位与依赖

| 项 | 路径 |
|----|------|
| 配置文件 | `configs/config_medium.json` |
| 数据加载（修改） | `src/mini_llm/m02_data_loader/__init__.py`（新增 `_load_from_huggingface`） |
| 训练脚本（复用） | `train.py` |
| 数据缓存 | `runs/gpt2_medium_wikitext103/data_cache/wikitext103_raw_train.txt` |
| Latest checkpoint | `runs/gpt2_medium_wikitext103/checkpoint_latest.pt` |
| Best checkpoint | `runs/gpt2_medium_wikitext103/checkpoint_best.pt` |
| 数据源 | HuggingFace `Salesforce/wikitext` / `wikitext-103-raw-v1` |
| 新增依赖 | `datasets>=2.14.0`（pyproject.toml） |
| 硬件要求 | M3 Max 36GB（MPS 加速），或同等 CUDA GPU |
