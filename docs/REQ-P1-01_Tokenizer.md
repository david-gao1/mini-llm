# REQ-P1-01：GPT-2 BPE 分词器

**所属**：[SPEC.md](../SPEC.md) → Part I · 模块 01  
**被依赖**：[REQ-P1-02](REQ-P1-02_DataLoader.md)（DataLoader 编码入口）、[REQ-P1-05](REQ-P1-05_Train.md)（训练时词表校验与文本 decode）  
**状态**：✅ 已完成  
**OpenSpec（行为契约）**：[预训练 · `pretraining/spec.md`](../openspec/specs/pretraining/spec.md)

---

## 1. 业务逻辑（读完就知道「要干嘛」）

### 先打个比方

模型只能吃数字，不能吃汉字或英文。**分词器**就像一本「拆字对照表」：把任意句子拆成有限种 **小块**（子词），每块对应一个整数；解码就是倒回去拼成给人看的字符串。  
我们直接用 GPT-2 官方训练好的 **BPE 词表**（通过 tiktoken），不自己从零训练合并规则——省时，也和书上一致。

### 最关键的一句话

> **把任意文本变成一串整数 ID，再能基本无损变回文本**；词表大小 **50257** 必须和 `configs/config.json` 里 `model.vocab_size` **完全一致**，否则后面 Embedding 和输出层会像对不上号的表格一样维度报错。

### 这条 REQ 还约定什么

- **编码**：文本 → 子词 → ID 列表  
- **解码**：ID 列表 → 文本  
- **硬约束**：tokenizer、模型 embedding、LM 头三处 **同一张 50257 行词表**  

从下面 **§1.1** 起是 BPE 原理与粒度对比，可当延伸阅读。

---

## 1.1 BPE 与 vocab_size 原理详解

### 一句话总结

50,257 就是模型的"词汇量"——模型能认识和输出的最小文本单元（子词）的种类数。它决定了：

> Embedding 表的行数（输入侧的查找表）
> 输出层的分类类别数（模型要在 50,257 个选项中选下一个 token）
> 必须全局一致——tokenizer、Embedding、输出层三处的 50,257 必须对齐，否则就像数据库外键约束失败一样，维度不匹配直接崩溃




### 为什么需要 Tokenizer

神经网络只能处理数字，不能直接吃文本。Tokenizer 的职责就是把字符串转成整数数组，本质是一张 **编码查找表**（类比 `HashMap<String, Integer>`）。

### 三种编码粒度对比

| 方案 | 示例 | 字典大小 | 优点 | 缺点 |
|------|------|----------|------|------|
| **字符级** | `"hello"` → `[104,101,108,108,111]` | ~256 | 无 OOV，字典小 | 序列极长，模型处理成本高 |
| **单词级** | `"hello"` → `[15234]` | 几十万 | 序列短 | 字典爆炸，新词无法处理（OOV） |
| **BPE 子词级** ✅ | `"hello"` → `[31373]` | 50,257 | 兼顾序列长度与覆盖率 | 需要预训练合并规则 |

### BPE 合并算法（简化）

```text
1. 初始字典 = 256 个字节（0x00 ~ 0xFF）
2. 在大规模语料中统计最高频的相邻 token 对，例如 "t"+"h" 出现最多
3. 将 "th" 合并为一个新 token，加入字典 → 字典 = 257
4. 重复步骤 2-3，直到字典达到目标大小（GPT-2 合并了 50,000 次）
```

最终效果——自适应压缩编码（类似 Huffman 编码的思路）：

```text
高频词：  "hello"   → [31373]                         # 1 个 token
中频词：  "unhappy" → [403, 34477]                     # "un" + "happy"，2 个 token
罕见组合："Bxzqk"   → [33, 87, 89, 80, 74]            # 退化为字符级，5 个 token
```

### 50,257 的构成

| 组成部分 | 数量 | 说明 |
|----------|------|------|
| 基础字节 token（0x00-0xFF） | 256 | 保证任何字节序列都能编码 |
| BPE 合并产生的子词 | 50,000 | 在 WebText 语料（~40GB）上统计合并 |
| 特殊控制 token | 1 | `<\|endoftext\|>`，标记文档边界 |
| **合计** | **50,257** | GPT-2 全系列统一词表 |

### 在模型中的数据流

```text
         原始文本                  Tokenizer                    Embedding 层                  Transformer              输出层
   "Every effort moves"  ──encode()──→  [6109, 3626, 6100]  ──lookup──→  3个384维向量  ──前向传播──→  [50257维概率]×3
                                                                │                                        │
                                                        50,257 行的查找表                        50,257 路 softmax
                                                    HashMap<int, float[384]>                    "下一个 token 是哪个？"
```

三处 50,257 必须严格对齐：
1. **Tokenizer**：`encode()` 输出的 id 范围 ∈ [0, 50257)
2. **Embedding 表**：`nn.Embedding(50257, emb_dim)` —— 行数 = 50,257
3. **输出分类头**：`nn.Linear(emb_dim, 50257)` —— 类别数 = 50,257

> 类比数据库外键约束：三张表引用同一个主键空间，任何一处不一致就会 "维度不匹配 → 运行时崩溃"。

### emb_dim（嵌入维度）详解

Embedding 层查表取出的每个向量有多长，就由 `emb_dim` 决定。它是 **人为选择** 的超参数，不是计算得来的。

```java
// emb_dim = 384 时（小模型配置）
float[] tokenVector = new float[384];   // 用 384 个浮点数"描述"一个 token

// emb_dim = 1024 时（Medium 配置）
float[] tokenVector = new float[1024];  // 用 1024 个浮点数"描述"一个 token
```

**维度越高，语义表达能力越强，但计算量和内存也越大。**

如果把 token 比作数据库中的一条记录：
- `vocab_size = 50,257` → 表有多少 **行**（多少个不同 token）
- `emb_dim = 384` → 每行有多少 **列**（用多少个特征描述这个 token）

```text
Embedding 表 = float[50257][384] = 50,257 行 × 384 列 = 19.3M 个浮点数
```

#### 业界常见 emb_dim 选择

这些数值没有理论公式，是通过大量实验调出的经验值：

| 模型 | emb_dim | n_heads | 每头维度 | 总参数量 |
|------|---------|---------|----------|----------|
| GPT-2 Small | 768 | 12 | 64 | 124M |
| **GPT-2 Medium** | **1024** | **16** | **64** | **355M** |
| GPT-2 Large | 1280 | 20 | 64 | 774M |
| GPT-2 XL | 1600 | 25 | 64 | 1.5B |
| GPT-3 | 12288 | 96 | 128 | 175B |

注意：GPT-2 全系列不管模型多大，每个注意力头都分到 **64 维**，通过增加头数来扩大总维度。

#### 硬约束：emb_dim 必须被 n_heads 整除

多头注意力（Multi-Head Attention）会把向量均分给每个头处理：

```text
config.json:         emb_dim=384  ÷ n_heads=6  = 64 维/头  ✅
config_medium.json:  emb_dim=1024 ÷ n_heads=16 = 64 维/头  ✅
```

如果不能整除，就无法均分，代码会直接报错。

#### 本项目两套配置对比

| 配置 | emb_dim | 含义 |
|------|---------|------|
| `config.json`（小模型调试） | 384 | 用 384 个特征描述 token，轻量快速 |
| `config_medium.json`（主力训练） | 1024 | 用 1024 个特征描述 token，与 GPT-2 Medium 原版一致 |

### 当前配置的 Embedding 参数效率

| | config.json（小模型） | config_medium.json（Medium） |
|---|---|---|
| **Embedding 参数量** | 50,257 × 384 = **19.3M** | 50,257 × 1,024 = **51.5M** |
| **模型总参数量** | ~38M | ~406M |
| **Embedding 占比** | ~50%（偏高，小模型调试用可接受） | ~12.7%（与 GPT-2 原版一致，合理） |
| **语料覆盖度** | the-verdict.txt 仅用到 ~3,000 种 token | WikiText-2 覆盖 ~30,000+ 种 token |

小模型中约一半参数花在 Embedding 表上且大量行未被训练到，这是小规模调试的已知代价。Medium 配置下比例合理，且保持了与 GPT-2 预训练权重的兼容性（未来可直接 `load_pretrained`）。

---

## 2. 设计思路（怎么做）

**方案**：tiktoken 封装 + 单例模式 + 模块级便捷函数。

**为什么选 tiktoken 而非 HuggingFace tokenizers**：
- tiktoken 是 OpenAI 官方实现，与 GPT-2 词表完全对齐
- 纯 Rust 后端，编解码速度快
- 书本范例也使用 tiktoken，降低对照难度

**为什么封装成类而非直接用 tiktoken**：
- 统一 `allowed_special` 默认值（允许 `<|endoftext|>`），避免下游每次调用都要手动指定
- 提供 `assert_vocab_size` / `vocab_matches_config` 等校验方法，与 config 形成防护闭环
- 模块级函数委托给默认单例，下游写 `from mini_llm.m01_tokenizer import encode_text` 即可

**关键设计决策**：
- `ALLOWED_SPECIAL_DEFAULT = {"<|endoftext|>"}` — 语料中边界符需被正常编码，不应报错
- `allowed_special=set()` 时禁止特殊 token，适用于不含边界符的纯文本场景
- 单例在模块加载时创建，后续零成本复用

---

## 3. 架构定位（在哪里）

```text
                  ┌──────────────────────────────┐
                  │  原始文本 / 语料文件            │
                  └──────────┬───────────────────┘
                             │
                             ▼
                  ┌──────────────────────────────┐
                  │  m01_tokenizer                │
                  │   encode_text() → list[int]   │
                  │   decode_token_ids() → str    │
                  │   vocab_size() → 50257        │
                  └──────────┬───────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     m02_data_loader    train.py       m05_generate
     (tokenize 语料)   (词表校验)     (decode 输出)
```

**上游**：原始文本（语料文件、用户输入）  
**下游**：m02 DataLoader（编码语料）、train.py（词表校验 + 采样 decode）、m05 generate（输出 decode）

---

## 4. 输入 / 输出契约

### 编码

```python
encode_text(text: str, *, allowed_special: set | "all" | None = None) -> list[int]
```

- `text`：待编码的自然语言文本
- `allowed_special`：允许的特殊 token 集合；`None` 时默认允许 `<|endoftext|>`
- 返回：token id 列表，每个值 ∈ [0, 50257)

### 解码

```python
decode_token_ids(token_ids: list[int]) -> str
```

- 返回：还原文本，与 `encode_text` 互逆

### 校验

```python
assert_vocab_size(expected: int) -> None          # 不一致则 raise ValueError
vocab_matches_config(model_vocab_size: int) -> bool  # 软检查
```

---

## 5. 业务规则

| # | 规则 | 说明 | 示例 |
|---|------|------|------|
| R1 | 编码名称 | 固定使用 `tiktoken.get_encoding("gpt2")`，50 257 子词 BPE | — |
| R2 | 往返一致 | `decode(encode(text)) == text`（在 allowed_special 覆盖范围内） | `"hello world"` → `[31373, 995]` → `"hello world"` |
| R3 | 特殊 token | 默认允许 `<|endoftext|>`，使语料边界符可正常编码 | `"text<\|endoftext\|>text"` → 不报错 |
| R4 | 禁止 special | `allowed_special=set()` 时遇 special token 抛异常 | 明确不含边界符的场景 |
| R5 | ID 范围 | 所有返回的 token id ∈ [0, vocab_size) | — |
| R6 | 配置对齐 | `vocab_size()` 必须等于 `config.model.vocab_size`（50257） | 不一致 → `ValueError` |
| R7 | 单例缓存 | 模块加载时创建默认 `GPT2Tokenizer` 实例，后续复用 | — |

---

## 6. 验收标准

| # | 输入 | 预期输出 |
|---|------|---------|
| AC1 | `vocab_size()` | `50257` |
| AC2 | `encode_text("Hello world")` → `decode_token_ids(...)` | `"Hello world"`（往返一致） |
| AC3 | 含 `<\|endoftext\|>` 的文本，默认 allowed_special | 正常编码，不抛异常 |
| AC4 | `vocab_matches_config(50257)` | `True` |
| AC5 | `assert_vocab_size(99999)` | `raise ValueError` |
| AC6 | `encode_text("")` → `decode_token_ids(...)` | `""`（空字符串往返） |
| AC7 | `encode_text(any_text)` 的每个 id | `0 <= id < 50257` |
| AC8 | 含 special 但 `allowed_special=set()` | `raise` 异常 |

单测 8 用例，全部 ✅。

---

## 7. 代码落位与依赖

| 项 | 路径 |
|----|------|
| 模块入口（便捷函数） | `src/mini_llm/m01_tokenizer/__init__.py` |
| 实现类 GPT2Tokenizer | `src/mini_llm/m01_tokenizer/tokenizer.py` |
| 测试 | `tests/test_tokenizer.py` |
| 设计文档 | `docs/m01_tokenizer.md` |
| BPE 原理 | `docs/bpe_principles.md` |
| 依赖库 | `tiktoken >= 0.5.0` |
