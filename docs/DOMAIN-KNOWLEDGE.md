# 项目领域知识（DDD 视角）

> **目的**：让人和 AI 在每次对话时共享相同的上下文，消除重复解释。
> 本文档用 DDD（领域驱动设计）的思想组织，不是为了"套概念"，而是为了把零散知识结构化，让任何人（包括未来的自己和 AI）都能快速建立全局认知。
>
> **最后更新**：2026-04-30

---

## 1. 战略全景（Strategic Overview）

### 1.1 领域定义

本项目是一个**从零实现的 GPT 预训练系统**，对照《Build a Large Language Model (From Scratch)》一书，目标是在 M3 Max 36GB 笔记本上完成从 BPE 分词到 GPT-2 Medium（406M 参数）预训练的完整链路。

**不是**：不是推理服务、不是多卡分布式训练系统。Phase 2 开始包含分类微调。

### 1.2 核心子域

| 子域 | 类型 | 说明 |
|------|------|------|
| **分词** | 支撑子域 | 将原始文本转化为 token 序列，消费 tiktoken 库 |
| **数据管道** | 支撑子域 | 语料获取、缓存、滑动窗口切分、DataLoader 构建 |
| **模型** | 核心子域 | Transformer 架构实现（Attention + FFN + GPTModel） |
| **训练** | 核心子域 | 预训练循环、损失计算、优化器、调度器、checkpoint |
| **生成** | 核心子域 | 自回归文本生成（贪心 / temperature / top-k） |
| **分类微调** | 核心子域 | 冻结预训练权重 + 换分类头，在下游任务上训练少量参数 |

### 1.3 限界上下文（Bounded Contexts）

#### 1.3.1 含义（在本仓库里指什么）

代码按**边界**分成多块：**块内部**沿用一套固定的约定（有哪些类型、如何调用）；**越过边界**时只使用事先说清的 **接口**——参数与返回值的类型、尤其是 **张量的形状与 dtype**——不必了解对方内部的实现细节。

**「限界上下文」在这里指的就是**：这样一种「对内一致、对外只走约定接口」的包边界。

落到本项目：

- **`src/mini_llm/m01_*` … `m05_*`** 各是一块边界；对外以各包 **`__init__.py`** 里公开的类型和函数为准。
- **跨边界传递的数据要尽量收敛**：多数是 **张量** 及其 **形状 / dtype**。例如 **`GPTModel`** 只吃 token 索引（整数张量）、吐出 logits（浮点张量）；它不读配置文件、不访问 HuggingFace——那是 **`m02`** / **`train.py`** 的职责。
- **`train.py`**、**`generate_from_checkpoint.py`** 处在更外层：负责读配置、创建对象、控制循环；**不实现** Attention、LayerNorm 等内部运算。

这样划分之后：**改数据管道时可限定在 `m02`**；排查问题时可以沿 **编码 → 样本对齐 → 前向形状 → loss** 分段验证；多人协作时只要对齐公开的函数签名与张量约定即可对接。

从预训练角度看，全链路是：**文本 → token ID → `(input, target)` 批次 → `forward` 得到 logits → 交叉熵 → 反向传播**。下面的每个 **`m0N`** 对应其中一段固定职责。

#### 1.3.2 各边界在链路中的职责

上一节把预训练拆成「文本 → … → 反向传播」。本节先用表格对齐**每一块落在链路的哪一步、对外约定什么**；表格下面是各块的**作用**——即在整系统里**为什么要单独分出去**、它**解决什么问题**。

| 目录 | 承担的步骤 | 接口层面的要点 |
|------|-------------|----------------|
| **m01_tokenizer** | 字符串 ↔ token ID 列表 | 与词表大小对齐；不参与训练循环 |
| **m02_data_loader** | 语料加载、滑动窗口、`(input, target)`、`DataLoader`、可选 `.pt` 缓存 | 输出批次形状与 `context_length`、batch size 一致 |
| **m03_attention** | 因果多头自注意力（供 `m04` 组装） | 一般不直接被 `train.py` import；由 `GPTModel` 内部使用 |
| **m04_model** | `GPTModel`：嵌入 + Transformer 块 + 输出头 | `forward`: `[B,T]` → logits `[B,T,V]` |
| **m05_generate** | 对已加载模型做自回归采样 | 反复 `forward`，每步用末尾位置 logits 采样下一个 token |
| **train.py** | 组装 config、数据、模型、优化器；训练与 checkpoint | 调度 `calc_loss_batch`、`evaluate_model` 等，不写底层算子 |
| **m06_classify_finetune** | SMS Spam 数据下载/平衡/Dataset + 分类 loss/accuracy + **扩展评估** | `SpamDataset`；`calc_accuracy_loader`；**`collect_predictions_loader` / `confusion_counts_binary_spam` / `prf1_from_counts` / `export_false_negative_spam_csv`** |
| **finetune_classify.py** | 加载预训练 checkpoint → 冻结 + 换 head → 分类微调训练 | 编排 m04 + m06；末段 **reload best val** → test 混淆矩阵、PRF、**FN CSV** |
| **eval_classify.py** | 已有分类 checkpoint → **不重训**复评 test | 读 `finetune_config.data.data_dir/test.csv`；打印指标；默认写出 **eval_false_negative_spam.csv** |
| **generate_from_checkpoint.py** | 加载 checkpoint，只做推理 | 仅路径 + 生成参数；不应混入训练专用逻辑 |

**各块的作用（为何要单独一块）：**

- **m01_tokenizer**：神经网络只能处理整数索引。**作用**是把人类可读文本变成固定词表下的 ID，并在生成后把 ID 解码回字符串；换语料、换实验时，编码规则集中在一处，模型代码不必关心 BPE 细节。
- **m02_data_loader**：训练需要成批的 `(input, target)`，还要处理下载、缓存、train/val 划分。**作用**是把「原始语料文件」变成 **`DataLoader` 能喂给训练循环的张量流**；重启训练时还可靠磁盘缓存跳过重复 tokenize，缩短启动时间。
- **m03_attention**：自注意力是 Transformer 里最易错、也最可复用的一块。**作用**是实现「带因果掩码的多头注意力」，供 **`GPTModel` 堆叠**；单独成模块便于单测和对照书本公式，而不把整个模型塞进一个文件。
- **m04_model**：这是**可学习的预测器本体**。**作用**是在给定一段上下文 token 后，为**每一个位置**给出「下一个 token」在词表上的分数（logits）；预训练的交叉熵损失、生成时的采样，都建立在这个 **`forward`** 之上。
- **m05_generate**：训练阶段关心的是「整段序列上的监督信号」；生成阶段关心的是「从已有前缀**一步一步**长出后面的 token」。**作用**是封装自回归循环（截断到 `context_length`、取最后一步 logits、按策略采样），避免在多个脚本里复制同一套循环逻辑。
- **train.py**：超参、优化器、调度器、eval、early stop、checkpoint、心跳日志等都和「模型数学」无关但缺一不可。**作用**是**编排**：按 config 把 **m02、m04** 接起来，驱动梯度更新与持久化，使「可训练」成为一个完整命令。
- **m06_classify_finetune**：微调需要与预训练不同的数据格式（CSV 分类标签）和评估方式（accuracy / **混淆矩阵** / **per-class F1** 而非 perplexity）。**作用**是封装数据集管道、`SpamDataset`、分类 loss、批量预测收集与 **TN/FP/FN/TP → PRF** 及 **FN 样本 CSV**；与预训练数据管道（m02）并行。
- **finetune_classify.py**：分类微调编排入口：训练结束后 **加载磁盘上的 best checkpoint**（与最后一 epoch 权重区分），再在 **test** 上打全局指标并导出漏判 spam，保证报告与 **交付物 checkpoint** 一致。
- **eval_classify.py**：在无 GPU 训练复盘或调阈值调研时，对任意已保存的 `.pt` **重复 test 集评估**，不必重跑 epoch。
- **generate_from_checkpoint.py**：推理不应加载整套训练数据。**作用**是从磁盘读取 **checkpoint** 里的权重与 config，重建 **`GPTModel`**，再交给 **m05** 输出文本；与训练入口分离，避免误把训练依赖带进演示或验收场景。

**三条主路径（便于对照）：**
- **预训练**：**m02 → m04 → loss → backward**
- **生成**：**m05** 反复调用 **m04** 的 `forward`，外层自回归循环
- **分类微调**：**m06（数据）→ m04（改造后）→ last-token logits → CE loss → backward**

#### 1.3.3 结构图（模块边界与张量流向）

```
┌──────────────────────────────────────────────────────────────────┐
│ mini_llm 包（src/mini_llm/）                                     │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐          │
│  │ m01_tokenizer│──>│ m02_data_    │   │ m03_attention│          │
│  │              │   │   loader     │   │              │          │
│  │ encode/decode│   │ Dataset/     │   │ MultiHead    │          │
│  └─────────────┘   │ DataLoader   │   │ Attention    │          │
│                     └──────────────┘   └──────┬───────┘          │
│                                               │                  │
│                                        ┌──────▼───────┐          │
│                                        │ m04_model    │          │
│                                        │ GPTModel     │          │
│                                        └──────┬───────┘          │
│                                               │                  │
│                     ┌──────────────┐   ┌──────▼───────┐          │
│                     │ m05_generate │<──│ (forward)    │          │
│                     │ 自回归采样    │   └──────────────┘          │
│                     └──────────────┘                              │
│                                                                  │
│                     ┌──────────────────┐                         │
│                     │ m06_classify_    │                         │
│                     │ finetune         │                         │
│                     │ SpamDataset /    │                         │
│                     │ accuracy / loss  │                         │
│                     └──────────────────┘                         │
└──────────────────────────────────────────────────────────────────┘
          │                 │                        │
          ▼                 ▼                        ▼
┌──────────────────────────────────────────────────────────────────┐
│ train.py / finetune_classify.py（编排层）                         │
│                                                                  │
│ train.py: 读 config → 加载数据 → 构建模型 → 预训练循环            │
│ finetune_classify.py: 加载 ckpt → 冻结 + 换 head → 分类微调；末段 reload best → test 指标 + FN CSV │
└──────────────────────────────────────────────────────────────────┘
```

依赖关系简述：**m02** 消费 **m01** 的编码结果（或缓存 tensor）；**m04** 内部使用 **m03**；**m05** 依赖训练好的 **m04**；**m06** 消费 **m01**（tiktoken 编码）和 **m04**（GPTModel）。**train.py** 组装 **m02、m04** 及优化逻辑；**finetune_classify.py** 组装 **m04、m06** 及微调逻辑；**generate_from_checkpoint.py** 组装 **m04、m05** 与权重加载。

#### 1.3.4 边界规则（实现与阅读代码时）

- 每个 **`m0N_*`** 对外通过 **`__init__.py` 公开 API**；其它包不要依赖其内部私有模块路径。
- 模块之间优先用 **张量契约** 通信：**形状**（如 `[B, T]`、`[B, T, V]`）、**dtype**（索引常用 `long`，logits 为浮点）。
- **预训练编排**以 **`train.py`** 为单一入口为宜，避免复制一套训练循环。
- **推理入口**（如 **`generate_from_checkpoint.py`**）单独维护，不把训练-only 的逻辑塞进去。

---

## 2. 统一语言（Ubiquitous Language）

以下术语在项目中有明确含义，对话时请直接使用，不需要解释：

| 术语 | 含义 | 代码位置 |
|------|------|----------|
| **token** | BPE 编码后的整数 ID（词表范围 0–50256） | `encode_text()` 返回值 |
| **context_length** | 模型能看到的最大 token 窗口长度 | config `model.context_length` |
| **stride** | 滑动窗口步长（当前 = context_length，无重叠） | `GPTDataset` 构造参数 |
| **sample** | 一个 (input, target) 对，各为 `[T]` 长 tensor | `GPTDataset.__getitem__` |
| **micro-step** | 一次 forward + backward（batch_size=1） | `train.py` 内循环 |
| **optimizer step** | 累积 `grad_accum_steps` 个 micro-step 后做一次 `optimizer.step()` | `global_step` 计数 |
| **effective batch** | `batch_size × gradient_accumulation_steps` | 当前 = 1×4 = 4 |
| **eval** | 每 `eval_freq` 个 optimizer step 计算 train/val loss | `evaluate_model()` |
| **patience** | 连续 N 次 eval val_loss 不降则 early stop | config `train.patience` |
| **run** | 一次完整训练实验（含配置、数据缓存、checkpoint） | `runs/<run_name>/` 目录 |
| **闸门 M1** | 预训练闭环验收：loss 有限 + checkpoint 写出 | `HARNESS.md` |
| **闸门 M2** | 训练→生成链路验收：checkpoint 加载 + 文本生成 | `HARNESS.md` |
| **token cache** | 预先 tokenize 好的 `.pt` 文件，跳过重复 tokenization | `data_cache/*.pt` |
| **微调 (finetune)** | 在预训练权重基础上，用下游任务数据继续训练少量参数 | `finetune_classify.py` |
| **分类头** | 替换 LM head 的 `Linear(emb_dim, num_classes)` | `model.out_head` 被替换 |
| **冻结 (freeze)** | 将参数 `requires_grad=False`，训练时不更新 | 冻结除末层外的全部参数 |
| **last-token logits** | 取序列最后位置的输出做分类（因果注意力下信息最完整） | `model(batch)[:, -1, :]` |
| **accuracy** | 分类正确数 / 总样本数 | `calc_accuracy_loader()` |

---

## 3. 聚合与实体（Aggregates & Entities）

### 3.1 配置聚合根（Config）

配置是整个系统的"事实来源"，所有模块从中读取参数。

```
config.json / config_medium.json
├── run_name          运行名称（决定输出目录）
├── seed              随机种子
├── device            "auto" / "cpu" / "cuda" / "mps"
├── data              数据子域配置
│   ├── source        "url" | "huggingface"
│   ├── filename      缓存文件名
│   ├── train_ratio   train/val 切分比例
│   └── (hf_path, hf_name, hf_split)  HuggingFace 专用
├── model             模型架构配置
│   ├── vocab_size    50257（与 tiktoken GPT-2 对齐）
│   ├── context_length
│   ├── emb_dim, n_heads, n_layers
│   ├── drop_rate, qkv_bias
└── train             训练超参配置
    ├── learning_rate, weight_decay
    ├── num_epochs, batch_size
    ├── gradient_accumulation_steps
    ├── eval_freq, eval_iter
    ├── checkpoint_every_steps
    ├── grad_clip, warmup_ratio, min_lr_ratio
    ├── patience
    └── start_context
```

**两套配置**：

| | `config.json` | `config_medium.json` |
|---|---|---|
| 用途 | 小模型快速验证 | GPT-2 Medium 正式训练 |
| 语料 | the-verdict.txt (20KB) | WikiText-103 raw (500MB) |
| 参数量 | 29M (6层/384d/6头) | 406M (24层/1024d/16头) |
| batch | 8 | 1 + grad_accum=4 |
| 内存 | ~0.7GB | ~9-10GB |

### 3.2 数据管道实体

```
语料原始文本 (str)
    │
    ▼ encode_text()
token_ids (list[int])
    │
    ▼ np.array() → torch.from_numpy()    ← 关键：不用 torch.tensor(list)
token_tensor (Tensor[long], 连续一维)
    │
    ▼ 滑动窗口 (stride=context_length)
GPTDataset: _tokens + _offsets
    │
    ▼ DataLoader (batch_size, shuffle)
(input_batch[B,T], target_batch[B,T])  ← 张量契约
```

**缓存层级**（首次加载后的复用路径）：

```
runs/<run_name>/data_cache/
├── wikitext103_raw_train.txt    L1缓存：原始文本（HuggingFace → 本地 txt）
├── train_tokens.pt              L2缓存：tokenized tensor（后台异步写入）
└── val_tokens.pt                L2缓存：tokenized tensor
```

启动时判断优先级：**L2 命中 → 秒加载** > L1 命中 → 需 tokenize > 无缓存 → 需下载+tokenize

### 3.3 模型实体（GPTModel）

```
GPTModel (406M params for medium)
├── tok_emb: Embedding(50257, emb_dim)       词嵌入
├── pos_emb: Embedding(context_length, emb_dim)  位置嵌入
├── drop_emb: Dropout
├── trf_blocks: Sequential(TransformerBlock × n_layers)
│   └── TransformerBlock
│       ├── norm1 → MultiHeadAttention → drop → residual
│       └── norm2 → FeedForward(emb→4×emb→emb) → drop → residual
├── final_norm: LayerNorm
└── out_head: Linear(emb_dim, vocab_size)    LM head（无 weight tying）
```

**与标准 GPT-2 的差异**：tok_emb 和 out_head **不共享权重**（教学清晰），因此 406M > 标准 355M。

### 3.4 训练循环（Application Service）

```
for epoch in range(num_epochs):
    for input_batch, target_batch in train_loader:
        loss = CE(model(input), target) / grad_accum_steps
        loss.backward()
        micro_step++
        if micro_step % grad_accum_steps == 0:    ← 累积够了才更新
            clip_grad_norm → optimizer.step → scheduler.step → zero_grad
            global_step++
            if global_step % eval_freq == 0:
                evaluate → early_stopping check
            if global_step % ckpt_every == 0:
                save_checkpoint
    print_sample()   ← 每 epoch 结束生成一段文本
```

---

## 4. 上下文映射（Context Map）

### 4.1 模块依赖关系

```
m01_tokenizer ← 无上游依赖（封装 tiktoken）
      ↓
m02_data_loader ← 依赖 m01（encode_text）
      ↓
m03_attention ← 无上游依赖（纯 PyTorch）
      ↓
m04_model ← 依赖 m03（MultiHeadAttention）
      ↓
m05_generate ← 依赖 m04（model.forward）
      ↓
m06_classify_finetune ← 依赖 m01（tiktoken）+ m04（GPTModel）
      ↓
train.py ← 编排预训练（m02 + m04）
finetune_classify.py ← 编排微调（m04 + m06）
```

### 4.2 外部依赖（防腐层）

| 外部系统 | 集成方式 | 防腐层 |
|----------|----------|--------|
| **tiktoken** | m01_tokenizer 封装 | `GPT2Tokenizer` 类屏蔽 tiktoken API |
| **HuggingFace datasets** | m02_data_loader 封装 | `_load_from_huggingface()` 下载后转为纯 txt |
| **PyTorch MPS** | train.py | `_pick_device("auto")` 自动探测 |

---

## 5. 运行时约束与硬件上下文

### 5.1 硬件环境

- **机器**：MacBook Pro M3 Max，36GB 统一内存
- **加速**：MPS（Metal Performance Shaders），无 CUDA
- **磁盘**：SSD，读写速度不是瓶颈

### 5.2 内存预算（WikiText-103 配置）

| 项 | 占用 |
|----|------|
| 模型参数 (float32) | 1.6 GB |
| 梯度 | 1.6 GB |
| AdamW 状态 (m + v) | 3.2 GB |
| 激活值 (batch=1, ctx=1024) | ~2 GB |
| 数据集 tensor | ~0.9 GB |
| **峰值总计** | **~9-10 GB** |

> **历史教训**：batch_size=4 时实测峰值 55GB+，MPS 后端激活值内存开销远超 CUDA 估算。改为 batch_size=1 + grad_accum=4 后解决。

### 5.3 训练规模

| | 小模型 (config.json) | Medium (config_medium.json) |
|---|---|---|
| 每 epoch 步数 | ~12 (batch=8) | ~27,483 micro / ~6,870 optimizer |
| 总 optimizer 步数 | ~600 (50 ep) | ~20,600 (3 ep) |
| 预计总时间 | ~10 分钟 | ~20-30 小时 |
| 产出 | loss→0.005，完全记忆 | 目标 val_loss < 5.0 |

---

## 6. 已沉淀的经验（踩坑记录）

这些是实际运行中积累的知识，每一条都对应一个真实问题。

### 6.1 数据加载

| 问题 | 根因 | 解法 | 状态 |
|------|------|------|------|
| `torch.tensor(list[int])` 卡死 | 112M Python int 对象逐个拆箱极慢 + 释放后内存碎片 | 改用 `np.array()` → `torch.from_numpy()` | 已修复 |
| 每次重启 tokenize 30s | 无 token 缓存 | 首次 tokenize 后保存 `train_tokens.pt` / `val_tokens.pt` | 已修复 |
| `torch.save` 阻塞训练启动 | 同步写 859MB 文件 | 改为后台 daemon 线程异步写入 | 已修复 |
| 日志文件 0B（看似没启动） | `nohup` 重定向时 Python stdout 有缓冲 | 加 `PYTHONUNBUFFERED=1` + `python -u` | 已修复 |
| `uv run` 找不到 `train.py` | 终端 cwd 不在项目目录，`uv` 解析到上级 | 使用 `.venv/bin/python` 绝对路径启动 | 已修复 |

### 6.2 训练

| 问题 | 根因 | 解法 |
|------|------|------|
| batch_size=4 OOM (55GB+) | MPS 激活值内存远超估算 | batch_size=1 + grad_accum=4 |
| patience=5 过早 early stop | 406M 模型 val_loss 正常波动 0.05，5 次 eval 就触发 | patience=20 |
| 生成含大量 `<unk>` | WikiText-2 tokenized 版本的预处理产物 | 切换到 WikiText-103 **raw** 版本 |
| WikiText-2 过拟合 (gap=2.0) | 10MB 数据对 406M 参数太少 (参数/数据比 160:1) | 切换到 WikiText-103 (500MB, 比值 3:1) |
| 「卡住」在 Gradient accumulation 后长时间无新日志 | `eval_freq=2000`：第一条带 train/val loss 的输出要等到 **optimizer step 2000**，此前内层循环不写 stdout（约 **8000** 次 micro-batch）；406M+MPS 若每步数秒则可能要 **数小时** 才有第一条 eval | `train.py` 心跳：`heartbeat_every_steps`（默认每 100 步）；或暂时减小 `eval_freq` |

### 6.3 可靠启动命令

```bash
# 后台运行 + 无缓冲日志 + 绝对路径（最稳定）
nohup env PYTHONUNBUFFERED=1 \
  /abs/path/to/team-mini-llm/.venv/bin/python -u \
  /abs/path/to/team-mini-llm/train.py \
  --config /abs/path/to/team-mini-llm/configs/config_medium.json \
  > /abs/path/to/team-mini-llm/train_wt103.log 2>&1 &

tail -f /abs/path/to/team-mini-llm/train_wt103.log
```

### 6.4 生成与人工检验（中英文边界）

用于 **`generate_from_checkpoint.py`** 或训练结束时的 **`print_sample`** 时，容易出现「明明 loss 还行，生成却很怪」的误判，多数与 **prompt 语言与训练域不一致**有关。

| 现象 | 根因 | 正确理解 |
|------|------|----------|
| 用「你好」「你是谁」开头，后面变成英文或乱码 | **WikiText-103 为英文维基**；模型从未学习中文接续分布 | **不是 checkpoint 坏了**；应用 **英文** prompt 检验本次实验 |
| 中文后出现问号方块（U+FFFD） | **GPT-2 BPE** 对中文切分与训练时见过的序列差异大 | 同上：改用英文开头 |
| 正文出现 `8 @.@ 4`、`@-@` | WikiText **raw** 语料里小数点、连字符的写法 | 模型模仿语料风格，**非程序错误** |
| `--temperature 0` 时出现 `Mary , Mary , …` 式重复 | **贪心解码**（每步取概率最大 token），易锁进局部重复循环 | **不是 checkpoint 坏了**；检验观感优先用默认 **temperature≈0.8 + top-k**；仅在和调试场景需要完全可复现时用 `0` |
| 多行粘贴命令报错 `command not found: --prompt` | shell 续行少了行尾 `\`，`--prompt` 未被传给 Python | 除最后一行外，每行行尾保留 `\` |

**推荐检验命令（英文开头）：**

```bash
uv run python generate_from_checkpoint.py \
  --checkpoint runs/gpt2_small_wikitext103/checkpoint_best.pt \
  --prompt "The history of London began in"
```

脚本在未加载模型前若检测到中日韩等字符，会向 stderr 打印提示（与上表一致）。

**展开说明与示例：** [`docs/RUN_REPORT_gpt2_small_wikitext103.md`](RUN_REPORT_gpt2_small_wikitext103.md) **第七节**（含参数表与 `@.@` 说明）。生成模块 REQ 中的对照说明见 [`REQ-P2-01_Generate.md`](REQ-P2-01_Generate.md) **第 8 节**。

### 6.5 预训练 vs 对话能力 vs 「世界知识」

本项目 **`train.py`** 做的是 **WikiText 域上的因果语言建模（下一 token 预测）**：产出的是「维基正文风格的续写能力」，**不是**产品意义上的 **Chat 助手**，也**不等价于**通识 **世界知识库**。

**为何不能像「hi, who are you」那样正常对话？**

| 能力 | 当前 checkpoint 大致处在哪 | 缺什么 |
|------|---------------------------|--------|
| 百科式英文续写 | 与训练目标一致 | — |
| 多轮对话、自我介绍、助手口吻 | 基本不具备 | 几乎没在 **对话 / 指令–回答** 数据上训练；无 **指令微调（SFT）**、无常见 **对齐（RLHF / DPO 等）** |
| 可靠事实与广义常识 | 极有限 | 语料仅为英文维基的一小部分；模型规模与训练量远低于大型商用基座 |

**「世界知识」常见从哪里来（可多选叠加）：**

| 途径 | 含义 |
|------|------|
| **更大规模、更多样的预训练** | 书、网页、百科等海量文本 + 更大模型容量，参数中隐含更多事实与用法（成本高）。 |
| **指令微调（SFT）** | 用「用户消息 → 助手回复」类监督数据，学会对话格式与任务跟随；**主要改善交互形态**，不单独等价于无限事实。 |
| **对齐（RLHF / DPO 等）** | 用人类偏好优化回答风格与安全边界；多在 SFT 之后。 |
| **检索增强（RAG）** | 生成前从知识库/文档检索相关内容再组织回答，减轻「全靠参数记事实」的负担。 |

**若在本项目路线上的务实延展（由易到难）：**

1. **对比 baseline**：直接试用开源 Chat 模型或 API，理解「预训练基座」与「对话模型」的差距。  
2. **在现有 checkpoint 上做小规模指令微调**：自备中英文指令–回答 JSON/文本，套对话模板（如 user/assistant 标记），复用 `GPTModel` + 短训练循环（可用更小 LR、可选 LoRA 等参数高效方法——需额外实现或引入库）。预期：**话术与格式会靠近对话**，世界知识仍主要来自当前底座。  
3. **更大预训练或更多语料**：提升语言与事实的上限，时间与算力显著增加。  
4. **应用层 RAG**：若要可控的事实问答，在推理管线外挂检索。

本节与上文 **6.4** 一起看：**6.4** 解决「英文维基模型不要用中文 prompt 误判」；**6.5** 解决「不要用 Chat 产品的预期误判 WikiText 预训练 checkpoint」。

### 6.6 分类微调（P2-02 SMS Spam）

**核心思路**：复用预训练模型的语言理解能力，只训练少量参数完成下游分类任务。

| 步骤 | 做了什么 | 为什么 |
|------|----------|--------|
| 冻结全部参数 | `requires_grad=False` | 防止破坏预训练学到的语言表示（灾难性遗忘） |
| 替换 out_head | `Linear(768, 50257)` → `Linear(768, 2)` | LM head 无用，分类只需 2 个输出 |
| 解冻末层 | 最后 1 个 Transformer block + final_norm | 让末层适配新任务，比只训练 head 效果更好 |
| 取最后 token | `logits[:, -1, :]` | 因果注意力下，最后 token 能「看到」所有前文，信息最完整（与书本第六章一致） |
| 数据平衡 | ham 下采样到与 spam 同量 | 避免模型只学会「全预测 ham」就拿到约 87% 准确率 |

**预期与书上的差异**：我们加载自训练的 GPT-2 Small（163M, WikiText-103），书上用 OpenAI 官方 GPT-2（124M、海量数据），微调准确率可能略低；SMS 任务通常仍可达 REQ 约定的 ≥90% test accuracy。

#### 6.6.1 标签与混淆矩阵（ham=0，spam=1）

评估脚本对 **整集 test**（或任意 `DataLoader`，且 **`shuffle=False`** 以保持与 CSV 行对齐）逐条 `argmax` 得到预测，再与真实标签累计 **混淆矩阵**四格：

|  | 预测 ham | 预测 spam |
|--|----------|-----------|
| **真实 ham** | TN（真负） | FP（假正：把好短信标成垃圾） |
| **真实 spam** | FN（假负：**漏判垃圾**，过滤场景常更敏感） | TP（真正） |

**Accuracy** = (TN+TP) / N。**只看 accuracy** 的问题在于：在 **极度不平衡** 的真实运营商短信里，「全判 ham」也可能准确率很高；因此本项目训练阶段做 **平衡集**，并在 REQ **BL-P2-02-02** 落地后固定给出 **spam 的 Precision / Recall / F1**，单独盯住 **Recall_spam** = TP / (TP + FN) 是否足够。

与常用库的对齐：若使用 `sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0,1])`，得到的 2×2 矩阵左上角依次为 TN、FP、FN、TP（行列为标签 0、1）。

#### 6.6.2 Precision / Recall / F1（直觉）

以 **spam 为正类**：

- **Precision_spam**：模型喊 spam 的样本里，有多少真是 spam（关系到「用户会不会投诉误判」）。
- **Recall_spam**：真实 spam 里有多大比例被抓住（关系到「漏了多少垃圾」）。
- **F1_spam**：二者调和平均，便于单一标量对比模型变体。

对 **ham** 对称地也可定义 Precision_ham / Recall_ham / F1_ham（实现见 `prf1_from_counts`）。**两类 F1 都高**比「只有一个 accuracy」更接近「两边都不翻车」。

#### 6.6.3 为何报告前 reload **best val** checkpoint

训练最后一轮的权重未必是 **验证集上最优** 的那一轮；保存到磁盘的 `checkpoint_best.pt` 才是默认交付物。若在最后一 epoch 上算 test，会与 **`classify_sms.py` 加载的权重** 不一致。因此 **`finetune_classify.py`** 在打混淆矩阵和写 **FN CSV** 之前，从 `checkpoint_best.pt` **重新加载** `model_state_dict`。

#### 6.6.4 FN CSV 与探针清单

- **`runs/<run_name>/test_false_negative_spam.csv`**（训练脚本末段）、或 **`eval_false_negative_spam.csv`**（`eval_classify.py` 默认路径）：列为 `Index, Label, Pred, Text`，只含 **真实 spam、预测 ham**，便于人工归纳话术模式（模板 spam、拼写变体等）。
- **[`docs/probes/classify_spam_probes.json`](probes/classify_spam_probes.json)**：少量固定英文句 + 人工 **`expected`**（ham/spam），用于演示或脚本回归；**不写入训练标签**，也 **不保证** 当前权重一定匹配——与高置信错例（见 [`REPORT_ClassifySpamProbe.md`](REPORT_ClassifySpamProbe.md)）一并用来讨论 **分布外模板**，而非替代 holding-out test。

#### 6.6.5 与单条推理的分工

- **`classify_sms.py`**（P2-03）：product/API 形态，stdin/`--text` → stdout 标签；`--probs` 辅助看置信度。
- **`eval_classify.py`**：离线批量指标与 FN 审计，与 REQ-P2-02 **§4**、**§11** 契约一致。

**其余 backlog**（加权 CE、官方 GPT-2 权重对照、`--smoke` 等）仍见 [`REQ-P2-02_ClassifyFinetune.md`](REQ-P2-02_ClassifyFinetune.md) **§10**。

---

## 7. 文件系统地图（磁盘布局）

```
team-mini-llm/
├── configs/
│   ├── config.json              小模型配置（29M，the-verdict.txt）
│   ├── config_medium.json       GPT-2 Medium 配置（406M，WikiText-103）
│   ├── config_gpt2_small.json   GPT-2 Small 配置（163M，WikiText-103）
│   └── config_classify_spam.json 分类微调配置（SMS Spam，5 epoch）
│
├── src/mini_llm/
│   ├── __init__.py
│   ├── m01_tokenizer/           分词：tiktoken GPT-2 BPE 封装
│   │   ├── __init__.py          模块级函数 (encode_text, decode_token_ids, ...)
│   │   └── tokenizer.py         GPT2Tokenizer 类
│   ├── m02_data_loader/         数据管道：语料加载 + 滑动窗口 + token 缓存
│   │   └── __init__.py          GPTDataset, load_text, train_val_dataloaders
│   ├── m03_attention/           多头因果自注意力
│   │   └── __init__.py          MultiHeadAttention
│   ├── m04_model/               完整 GPT 模型
│   │   └── __init__.py          LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel
│   ├── m05_generate/            自回归生成
│   │   └── __init__.py          generate_text_simple, generate
│   └── m06_classify_finetune/   分类微调数据管道与评估
│       └── __init__.py          SpamDataset；accuracy；混淆矩阵/PRF/FN 导出等
│
├── train.py                     预训练入口（编排层）
├── finetune_classify.py         分类微调入口（末段 best→test 指标 + FN CSV）
├── eval_classify.py             已有分类 checkpoint → test 混淆矩阵 / PRF / FN CSV
├── classify_sms.py              分类 checkpoint → 单行英文短信 → ham / spam
├── generate_from_checkpoint.py   加载 checkpoint 做文本生成（人工检验）
│
├── tests/                       pytest 测试套件
│   ├── test_tokenizer.py
│   ├── test_data_loader.py
│   ├── test_attention.py
│   ├── test_model_forward.py
│   ├── test_imports.py
│   ├── test_classify_finetune.py  分类 Dataset / loss / encode（P2-02/P2-03）
│   └── test_classify_metrics.py   混淆矩阵、PRF、FN CSV、探针 JSON（BL-P2-02-02）
│
├── runs/                        训练产出（gitignore）
│   ├── team_gpt/                小模型实验
│   │   └── data_cache/
│   └── gpt2_medium_wikitext103/ WikiText-103 实验
│       ├── data_cache/
│       │   ├── wikitext103_raw_train.txt  (516MB, L1 文本缓存)
│       │   ├── train_tokens.pt            (859MB, L2 token 缓存)
│       │   └── val_tokens.pt              (45MB,  L2 token 缓存)
│       ├── checkpoint_latest.pt
│       └── checkpoint_best.pt
│
├── docs/                        设计文档与需求文档
│   ├── DOMAIN-KNOWLEDGE.md      ← 本文档
│   ├── REQ-P1-01_Tokenizer.md   ~ REQ-P1-07_GPT2Medium.md
│   ├── REQ-P2-01_Generate.md
│   ├── REQ-P2-02_ClassifyFinetune.md
│   ├── REQ-P2-03_ClassifySmsInfer.md
│   ├── probes/                  SMS 分类回归探针（JSON）
│   │   └── classify_spam_probes.json
│   └── process/                 流程规范（产品/开发/测试/迭代）
│
├── SPEC.md                      API 契约 + 进度看板
├── HARNESS.md                   验收标准 + 闸门定义
├── PROCESS.md                   三角色闭环流程总纲
├── README.md                    项目总览 + 环境设置
├── REFERENCE.md                 原书章节对照表
├── pyproject.toml               依赖：torch, tiktoken, datasets, pandas
└── .cursor/
    └── skills/
        └── team-mini-llm-domain/   Agent Skill：模块边界与领域约定（SKILL.md）
```

---

## 8. 张量契约速查

贯穿整个系统的张量形状约定：

```
B = batch_size (当前 medium: 1)
T = context_length (当前 medium: 1024)
V = vocab_size (50257)
D = emb_dim (当前 medium: 1024)
H = n_heads (当前 medium: 16)

encode_text(str)           → list[int]           长度不定
GPTDataset[i]              → (input[T], target[T])
DataLoader batch           → (input[B,T], target[B,T])
GPTModel.forward([B,T])    → logits[B,T,V]
CE loss                    → scalar
generate(model, [B,T])     → [B, T+new_tokens]

--- 分类微调 ---
SpamDataset[i]             → (token_ids[T], label)       T=max_length
DataLoader batch           → (token_ids[B,T], labels[B])
model(batch)[:, -1, :]     → logits[B, num_classes]      取最后 token
CE loss (classification)   → scalar
accuracy                   → float ∈ [0, 1]
```

---

## 9. 需求演进时间线

```
P1-01  m01_tokenizer         done    tiktoken GPT-2 BPE
P1-02  m02_data_loader       done    滑动窗口 + 三路语料加载
P1-03  m03_attention         done    多头因果自注意力
P1-04  m04_model             done    GPTModel 完整架构
P1-05  train.py              done    基础训练循环
  ── 闸门 M1 ──              done    预训练闭环验收
P2-01  m05_generate          done    贪心 + temperature + top-k
  ── 闸门 M2 ──              done    训练→生成链路验收
P1-06  训练优化               done    MPS / 梯度裁剪 / cosine scheduler / early stop
P1-07  GPT-2 Medium          wip     WikiText-103 raw 训练中
  │
  ├─ WikiText-2 (Run 1)     done    patience=5 过早停，val_loss=5.81
  ├─ WikiText-2 (Run 2)     done    patience=20，val_loss=5.44，过拟合
  └─ WikiText-103 (Run 3)   wip     训练进行中，目标 val_loss < 5.0
  ── GPT-2 Small            done    163M, WikiText-103, val_loss=3.3092
P2-02  分类微调 (SMS Spam)   done    REQ-P2-02；含 BL-P2-02-02 评估与 eval_classify（DOMAIN §6.6 / REQ §11）
P2-03  classify_sms 推理     done    REQ-P2-03；依赖分类 checkpoint
```

---

## 10. AI 协作备忘

在与 AI 对话时，以下信息可以直接引用，无需重复解释：

1. **项目路径**：`/Users/lianggao/MyWorkSpace/001-360/llms_team_work/team-mini-llm`
2. **Python 环境**：`.venv/bin/python`（Python 3.11，uv 管理）
3. **硬件**：M3 Max 36GB，MPS 加速
4. **当前训练**：WikiText-103 + GPT-2 Medium (406M)，config_medium.json
5. **日志**：`train_wt103.log`（需 `PYTHONUNBUFFERED=1` 才能实时看到）
6. **工作流**：一人三角色（产品→开发→测试），先 REQ 再写码再验收
7. **本文档位置**：`docs/DOMAIN-KNOWLEDGE.md` —— 遇到上下文断裂时先读这个
8. **生成检验**：WikiText 英文预训练模型应用 **英文** `prompt`；中文开头会得到英文续写或解码异常（U+FFFD），属域外输入而非 checkpoint 损坏。见本文档 **第 6.4 节**、[`RUN_REPORT_gpt2_small_wikitext103.md`](RUN_REPORT_gpt2_small_wikitext103.md) **第七节**
9. **对话与世界知识**：WikiText 预训练 ≠ Chat 助手；闲聊与广义事实需 SFT / 更大预训练 / RAG 等。见本文档 **第 6.5 节**、运行报告 **第三节 3.5**
10. **Cursor Agent Skill**：与本仓库协作时可启用项目技能 **`team-mini-llm-domain`**（`.cursor/skills/team-mini-llm-domain/SKILL.md`），将 §1.3 边界、术语与生成踩坑压缩为可执行约定。
