# REQ-P1-01：GPT-2 BPE 分词器

**所属**：[SPEC.md](../SPEC.md) → Part I · 模块 01  
**被依赖**：[REQ-P1-02](REQ-P1-02_DataLoader.md)（DataLoader 编码入口）、[REQ-P1-05](REQ-P1-05_Train.md)（训练时词表校验与文本 decode）  
**状态**：✅ 已完成

---

## 1. 业务逻辑（为什么做）

GPT 模型的输入是整数 token id 序列，而非原始文本。需要一个分词器完成两件事：

1. **编码**：将自然语言文本切分为子词（subword）并映射为整数 id
2. **解码**：将模型生成的 id 序列还原为可读文本

团队采用 GPT-2 的 BPE（Byte Pair Encoding）词表，共 50 257 个子词。直接使用 OpenAI 的 tiktoken 库获取预训练词表，避免从零训练 BPE——这与书本的做法一致，重点放在理解 BPE 原理而非重新实现合并算法。

关键约束：`vocab_size` 必须与 `configs/config.json` 中 `model.vocab_size` 严格对齐，否则 Embedding 层维度不匹配会导致运行时崩溃。

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
