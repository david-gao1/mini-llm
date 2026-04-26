# REQ-P1-07：GPT-2 Medium 规模训练 + WikiText-2 语料

**所属**：[SPEC.md](../SPEC.md) → Part I · 模型与数据升级  
**依赖**：[REQ-P1-06](REQ-P1-06_TrainOptimize.md)（训练优化已到位：MPS / scheduler / early stopping）  
**被依赖**：无  
**状态**：todo

---

## 1. 业务逻辑（为什么做）

P1-01 到 P1-06 使用的是 29M 参数模型 + 20KB 短篇小说（the-verdict.txt），虽然验证了完整链路，但存在两个根本限制：

- **模型太小**：29M 参数，只有 6 层 Transformer，表达能力有限
- **语料太小**：20KB 文本，模型几十个 epoch 就能完全背诵，无法观察真正的语言建模能力

升级到 **GPT-2 Medium（355M 参数）** + **WikiText-2（10MB 维基百科文章）** 后：

- 模型参数量增长 **12 倍**，24 层 Transformer，接近工业级小模型的架构
- 语料增长 **500 倍**，模型不可能背诵，必须学习真正的语言模式
- 在 M3 Max 36GB 上可行：预估内存占用 ~12-14 GB，MPS 加速

这一步是从"教学验证"到"实际训练"的跨越。

---

## 2. 设计思路（怎么做）

**方案**：零代码改动，仅新增一份配置文件 `configs/config_medium.json`。

**为什么不改代码（方案 A）**：
- WikiText-2 的 train.txt 是一个完整文本文件，与 the-verdict.txt 格式完全兼容
- 现有的 `load_text` 三级回退 + URL 下载缓存机制直接适用
- 现有的 `train_val_dataloaders` 按字符比例 split 足够（0.95 train / 0.05 val）
- 不动代码 = 不引入新 bug，已有 24 个测试继续保护

**为什么选 WikiText-2 而非 WikiText-103**：
- WikiText-103 有 181MB，355M 模型在 M3 Max 上训一轮要几小时
- WikiText-2 的 10MB 规模与 355M 模型匹配更好，10 epoch 约 1-2 小时可完成
- 作为教学项目，能在合理时间内看到结果更重要

**为什么 context_length 可以上 1024**：
- M3 Max 36GB 统一内存，batch_size=1 时内存约 12-14 GB
- 1024 context 让模型看到更长的上下文依赖，WikiText-2 的段落正好需要这个长度
- GPT-2 原版就是 1024 context

**关键设计决策**：
- `batch_size=1`（内存安全）配合 1024 context
- `num_epochs=10`（10MB 语料不需要太多轮）
- `patience=5`（大语料收敛更慢，早停宽松一些）
- `start_context="The history of"`（维基百科风格的起始文本）
- `drop_rate=0.1`（大语料过拟合风险低，不需要激进正则化）

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

### 数据源

| 项 | 详情 |
|----|------|
| 数据集 | WikiText-2（Salesforce / PyTorch examples 托管） |
| 下载 URL | `https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt` |
| 文件大小 | ~10 MB |
| 内容 | 维基百科优质文章，英语长文，保留章节结构 |
| token 数 | ~2.5M tokens（GPT-2 BPE） |

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

### 训练超参详解

| 参数 | 值 | 说明 |
|------|------|------|
| `learning_rate` | 3e-4 | AdamW 初始学习率；配合 cosine scheduler 衰减 |
| `weight_decay` | 0.1 | 权重衰减正则化；大语料上不需要太激进 |
| `num_epochs` | 10 | 10MB 语料 × 10 轮 ≈ 25M token 的训练量 |
| `batch_size` | 1 | context=1024 时单 batch 占内存 ~12GB，batch=1 确保不 OOM |
| `eval_freq` | 100 | 每 100 步评估一次 train/val loss |
| `eval_iter` | 10 | 评估时取 10 个 batch 的平均，结果更稳定 |
| `checkpoint_every_steps` | 500 | 每 500 步保存一次 latest checkpoint |
| `grad_clip` | 1.0 | 梯度裁剪阈值，防止大模型训练中的梯度爆炸 |
| `warmup_ratio` | 0.1 | 前 10% 步数线性升温（0→lr），避免初始阶段大梯度 |
| `min_lr_ratio` | 0.1 | cosine 衰减下限 = lr × 0.1 = 3e-5 |
| `patience` | 5 | 连续 5 次评估 val_loss 不创新低则 early stop |
| `start_context` | "The history of" | 维基百科风格的生成起始文本 |

### 内存估算（M3 Max 36GB）

| 项 | 估算 | 说明 |
|----|-----:|------|
| 模型参数（float32） | 1.6 GB | 406M × 4 bytes |
| 梯度 | 1.6 GB | 与参数等大 |
| AdamW 状态（动量 + 方差） | 3.2 GB | 参数量 × 2 × 4 bytes |
| 激活值（batch=1, context=1024） | ~6 GB | 24 层 × 中间张量 |
| **总计** | **~12-14 GB** | 36GB 内存富余 |

### 训练规模估算

| 项 | 估算 |
|----|------|
| 训练集 token 数 | ~2.4M tokens |
| 每 epoch 步数（batch=1, stride=1024） | ~2,200 步 |
| 10 epoch 总步数 | ~22,000 步 |
| 单 epoch 时间（M3 Max MPS） | ~8-12 分钟 |
| 总训练时间（可能 early stop） | ~1-2 小时 |

### 输出

- `runs/gpt2_medium_wikitext2/checkpoint_latest.pt`
- `runs/gpt2_medium_wikitext2/checkpoint_best.pt`
- 训练日志（loss 曲线 + 生成样本）

---

## 5. 业务规则

| # | 规则 | 说明 | 示例 |
|---|------|------|------|
| R1 | 零代码改动 | 仅新增 config 文件，不改 m01-m05 和 train.py | 方案 A |
| R2 | 独立 run 目录 | `run_name="gpt2_medium_wikitext2"`，不覆盖原有 team_gpt | 两套实验并存 |
| R3 | 自动下载缓存 | 首次运行下载 WikiText-2 到 `data_cache/`，后续离线复用 | ~10MB |
| R4 | 字符比例 split | 0.95 train / 0.05 val，与现有逻辑一致 | ~9.5MB / ~0.5MB |
| R5 | batch_size=1 | 1024 context + 355M 参数，单 batch 确保内存安全 | ~12-14GB |
| R6 | MPS 加速 | M3 Max 自动选择 MPS（P1-06 已支持） | device="auto" |
| R7 | Early stopping | patience=20，大模型 loss 波动大需更多容忍 | 连续 20 次不降则停 |
| R8 | 原有 config 不动 | `configs/config.json` 保持 29M 小模型配置 | 可随时回退 |

---

## 6. 验收标准

| # | 场景 | 预期 |
|---|------|------|
| AC1 | 启动训练 | 日志显示 `Device: mps`，`Model parameters: ~355M` |
| AC2 | 数据下载 | `runs/gpt2_medium_wikitext2/data_cache/wikitext2_train.txt` 存在（~10MB） |
| AC3 | 首个 eval | train_loss 接近 ln(50257) ≈ 10.8（随机初始化） |
| AC4 | 训练 2-3 epoch 后 | train_loss 明显下降（< 5），val_loss 同步下降 |
| AC5 | 训练结束 | val_loss < train_loss 的差距远小于 the-verdict 实验（过拟合轻微） |
| AC6 | 生成文本 | 与维基百科风格相似的英语段落（有章节标题、事实性描述） |
| AC7 | checkpoint | `checkpoint_latest.pt` 和 `checkpoint_best.pt` 均存在 |
| AC8 | 内存 | 训练过程中不 OOM，M3 Max 36GB 足够 |
| AC9 | 总时间 | 10 epoch 在 1-2 小时内完成 |

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

---

## 8. 代码落位与依赖

| 项 | 路径 |
|----|------|
| 配置文件（新增） | `configs/config_medium.json` |
| 训练脚本（复用） | `train.py` |
| 数据缓存 | `runs/gpt2_medium_wikitext2/data_cache/wikitext2_train.txt` |
| Latest checkpoint | `runs/gpt2_medium_wikitext2/checkpoint_latest.pt` |
| Best checkpoint | `runs/gpt2_medium_wikitext2/checkpoint_best.pt` |
| 数据源 URL | `https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt` |
| 硬件要求 | M3 Max 36GB（MPS 加速），或同等 CUDA GPU |
