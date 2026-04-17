# m01 分词器：设计说明（Tokenizer）

本文说明 `src/mini_llm/m01_tokenizer` 的**设计决策**、**理论落点**与**实现边界**，并与原书 **LLMs-from-scratch 第 2 章**（`ch02/01_main-chapter-code/ch02.ipynb`、`dataloader.ipynb`）及仓库 **[`HARNESS.md`](../HARNESS.md)** 中的 **P1-01** 对齐。

---

## 1. 第 2 章在理论上的落点

语言模型处理的是**离散 token 序列**，不是原始字符串。第 2 章的核心脉络是：

1. **文本 → token id**：用某种子词算法（书中主线为 **Byte Pair Encoding, BPE**）把 UTF-8 文本切成子词单元，并映射到整数 id。
2. **词表大小 `vocab_size`**：id 的取值范围是 `0 … vocab_size - 1`，直接决定 **嵌入层**与**输出层**的最后一维，必须与模型配置一致。
3. **可逆性**：训练与推理需要 **encode（文本→id）** 与 **decode（id→文本）** 配套，且与后续 **滑动窗口 Dataset**（`m02_data_loader`）使用的 id 序列一致。

BPE 在工程上的含义是：在**字符/字节**与**整词**之间取折中，既控制词表规模，又能表示未登录词；GPT-2 所采用的 BPE 由 OpenAI 发布，**词表大小固定为 50257**（含特殊符号等），这是后续所有维度的基准之一。**BPE 算法动机、合并过程与为何采用子词级设计** 见 **[bpe_principles.md](bpe_principles.md)**。

---

## 2. 为什么采用「tiktoken + gpt2 + vocab 50257」

### 2.1 与书本主线一致

原书第 2 章及后续预训练章节（如 ch05）的**参考实现**使用 **`tiktoken.get_encoding("gpt2")`**：与 **OpenAI GPT-2** 官方 BPE **完全一致**。这样：

- 对照 `LLMs-from-scratch` 中的代码与实验时，**token 对齐**，便于排错与复现。
- 团队不必在课程主线内维护一份自研 merge 规则与词表文件，**降低实现与文档双重负担**。

### 2.2 词表 50257 是「编码」的一部分，不是随意常数

`gpt2` 编码在 `tiktoken` 中的实现里，**`n_vocab == 50257`**。该数目来自 GPT-2 BPE 的合并表与词表设计，**不是**本仓库随意选取的超参（**为何是「合并表 + 词表」导出的固定规模**，见 [bpe_principles.md](bpe_principles.md) §3.1、§5）；若改用其他编码名，则 `vocab_size` 会随之变化，必须**整体替换**（tokenizer、config、模型头）。

因此本仓库**显式坚持**：**编码名 `gpt2` ↔ 词表大小 50257 ↔ `configs/config.json` 中 `model.vocab_size`**，三者一致。

### 2.3 为什么用 tiktoken 而不是在 m01 里「从零实现 BPE」

- **主线目标**是打通 **数据 → 模型 → 训练 →（后续）生成**；分词在书中已有成熟库实现，**重复实现 BPE 合并算法**属于原书 **bonus**（`ch02/05_bpe-from-scratch/`），适合扩展学习，而非团队主路径的阻塞项。
- `tiktoken` 与 PyTorch 训练栈兼容良好，**encode/decode** 行为稳定、可测。

若课程强制「手写 BPE」，应在单独分支或模块中实现，并仍满足下文 **与 `vocab_size` 一致** 的契约；主分支可继续以 `tiktoken` 为默认 Harness。

---

## 3. 特殊 token 策略

### 3.1 为何需要单独约定

`tiktoken` 对某些 **特殊字符串**（如 `<|endoftext|>`）的默认行为是：若出现在正文中且未声明，可能报错或行为与预期不符。训练数据若**拼接多段文档**，常用 `<|endoftext|>` 作为**文档边界**标记，因此必须在 **encode 时显式允许**这类 special。

### 3.2 本仓库策略

- **调用方**（如 `m02_data_loader` 中对整段文本 `encode`）使用：

  `enc.encode(text, allowed_special={"<|endoftext|>"})`

  与书中常见写法一致，避免无意中将特殊串当普通文本处理。
- **m01** 提供 **`get_encoding()`**，返回标准 **`tiktoken.Encoding`**，**不**在包内隐藏 `allowed_special` 参数，以便：
  - 数据管线控制「哪些 special 可出现」；
  - 与书中「同一 encode API、不同调用场景」的讲法一致。

若未来增加其它 special（如自定义边界符），应在 **数据与 encode 调用处**统一列出，并在本文档与 **HARNESS** 的契约中补充一行说明。

---

## 4. 与 `config.json` 的契约：`vocab_size`

模型配置中 **`model.vocab_size`** 必须等于所选用分词器的词表大小：

| 项目 | 值（当前策略） |
|------|----------------|
| 编码 | `gpt2` |
| `tiktoken` 的 `n_vocab` | **50257** |
| `configs/config.json` → `model.vocab_size` | **50257**（须与上一致） |

**原因（理论落点）**：嵌入矩阵形状为 `(vocab_size, emb_dim)`，输出 logits 为 `(..., vocab_size)`。若 tokenizer 实际 id 超出 `vocab_size - 1` 或配置小于真实词表，会出现**索引越界**或**无效行从未被训练**；若配置大于真实词表，则**浪费参数**且与 softmax 语义不一致。

**工程约定**：修改编码或自训词表时，**先**确定 `vocab_size`，**再**改 `config.json` 与模型构造，并跑通 **HARNESS P1-01 / P1-04** 相关测试。

---

## 5. 当前策略下「要实现什么」（m01 边界）

在 **坚持 tiktoken + gpt2 + 50257** 的前提下，`m01_tokenizer` **实现的是薄封装**，而非重造 BPE：

| 内容 | 说明 |
|------|------|
| **`ENCODING_NAME`** | 固定为 `"gpt2"`，与书中一致。 |
| **`get_encoding()`** | 返回 `tiktoken.get_encoding("gpt2")`，供全局使用；**encode/decode** 语义与第 2 章一致。 |
| **`vocab_size()`** | 返回 `get_encoding().n_vocab`，供配置校验与模型构建引用（与 `config` 交叉核对）。 |

**不在 m01 内重复实现**：BPE 合并、词表序列化、字节级预处理——均由 `tiktoken` 完成。

**与 m02 的衔接**：`m02_data_loader` 取得 encoding 后对**整段文本**做 `encode` 与滑动窗口；m01 保证**词表与配置一致**，m02 保证**序列长度与 `context_length` 一致**。

---

## 6. 验收与追溯

- 需求条目：**[`HARNESS.md`](../HARNESS.md)** 中 **P1-01**（契约、Harness、通过判据）。
- 原书对照：**`LLMs-from-scratch/ch02/01_main-chapter-code/`**（`ch02.ipynb`、示例语料 `the-verdict.txt`）。

---

## 7. 若偏离当前策略（备忘）

| 变动 | 必须同步 |
|------|----------|
| 改用其它 `tiktoken` 编码名 | 更新 `ENCODING_NAME`、`vocab_size()` 与 `config.json` 中 `model.vocab_size` |
| 自训 BPE | 提供与 `tiktoken.Encoding` 可类比的 **encode/decode** 与确定 **`n_vocab`**，并更新配置与 HARNESS |

以上变动应视为**新 REQ**，在 `HARNESS.md` 中增行或改 P1-01 契约，避免口头约定。
