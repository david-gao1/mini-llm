# 项目领域知识（DDD 视角）

> **目的**：让人和 AI 在每次对话时共享相同的上下文，消除重复解释。
> 本文档用 DDD（领域驱动设计）的思想组织，不是为了"套概念"，而是为了把零散知识结构化，让任何人（包括未来的自己和 AI）都能快速建立全局认知。
>
> **最后更新**：2026-04-28

---

## 1. 战略全景（Strategic Overview）

### 1.1 领域定义

本项目是一个**从零实现的 GPT 预训练系统**，对照《Build a Large Language Model (From Scratch)》一书，目标是在 M3 Max 36GB 笔记本上完成从 BPE 分词到 GPT-2 Medium（406M 参数）预训练的完整链路。

**不是**：不是微调框架、不是推理服务、不是多卡分布式训练系统。

### 1.2 核心子域

| 子域 | 类型 | 说明 |
|------|------|------|
| **分词** | 支撑子域 | 将原始文本转化为 token 序列，消费 tiktoken 库 |
| **数据管道** | 支撑子域 | 语料获取、缓存、滑动窗口切分、DataLoader 构建 |
| **模型** | 核心子域 | Transformer 架构实现（Attention + FFN + GPTModel） |
| **训练** | 核心子域 | 预训练循环、损失计算、优化器、调度器、checkpoint |
| **生成** | 核心子域 | 自回归文本生成（贪心 / temperature / top-k） |

### 1.3 限界上下文（Bounded Contexts）

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
└──────────────────────────────────────────────────────────────────┘
          │                 │                        │
          ▼                 ▼                        ▼
┌──────────────────────────────────────────────────────────────────┐
│ train.py（项目根目录，编排层 / Application Service）               │
│                                                                  │
│ 职责：读 config → 加载数据 → 构建模型 → 训练循环 → checkpoint     │
│ 不含业务逻辑，只做"胶水"编排                                      │
└──────────────────────────────────────────────────────────────────┘
```

**关键边界规则**：
- 每个 `m0N_xxx` 是一个独立限界上下文，通过 `__init__.py` 暴露公开 API
- 模块间只通过**张量契约**通信（形状、dtype），不依赖内部实现
- `train.py` 是唯一的编排入口，所有模块在这里组装

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
train.py ← 编排所有模块
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

---

## 7. 文件系统地图（磁盘布局）

```
team-mini-llm/
├── configs/
│   ├── config.json              小模型配置（29M，the-verdict.txt）
│   └── config_medium.json       GPT-2 Medium 配置（406M，WikiText-103）
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
│   └── m05_generate/            自回归生成
│       └── __init__.py          generate_text_simple, generate
│
├── train.py                     训练入口（编排层）
├── generate_from_checkpoint.py   加载 checkpoint 做文本生成（人工检验）
│
├── tests/                       pytest 测试套件（24 用例）
│   ├── test_tokenizer.py
│   ├── test_data_loader.py
│   ├── test_attention.py
│   ├── test_model_forward.py
│   └── test_imports.py
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
│   └── process/                 流程规范（产品/开发/测试/迭代）
│
├── SPEC.md                      API 契约 + 进度看板
├── HARNESS.md                   验收标准 + 闸门定义
├── PROCESS.md                   三角色闭环流程总纲
├── README.md                    项目总览 + 环境设置
├── REFERENCE.md                 原书章节对照表
└── pyproject.toml               依赖：torch, tiktoken, datasets
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
