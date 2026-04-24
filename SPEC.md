# 项目规格与进度追踪

集中记录每个模块的 **API 契约、实现状态、测试覆盖与阻塞项**。
验收标准见 [`HARNESS.md`](HARNESS.md)，流程规范见 [`PROCESS.md`](PROCESS.md)。

> 状态标记：`done` 已完成 · `wip` 进行中 · `todo` 未开始 · `blocked` 被阻塞

---

## 总览看板

| 模块 | REQ | 代码 | 测试 | 阻塞项 |
|------|-----|------|------|--------|
| `m01_tokenizer` | P1-01 | done | done | — |
| `m02_data_loader` | P1-02 | done | done | — |
| `m03_attention` | P1-03 | done | done | — |
| `m04_model` | P1-04 | done | done | — |
| `train.py` | P1-05 | done | todo | 端到端训练未跑（依赖数据下载） |
| `m05_generate` | P2-01 | done | done | — |
| **闸门 M1** | — | — | blocked | 端到端训练未验证 |
| **闸门 M2** | — | — | blocked | 依赖 M1 checkpoint |

---

## P1-01 · `m01_tokenizer`

**源码** `src/mini_llm/m01_tokenizer/__init__.py`
**设计文档** [`docs/m01_tokenizer.md`](docs/m01_tokenizer.md)

### 公开 API

```python
get_encoding() -> tiktoken.Encoding
vocab_size() -> int                         # 50257 (GPT-2 BPE)
encode_text(text, *, allowed_special=None) -> list[int]
decode_token_ids(token_ids: list[int]) -> str
assert_vocab_size(expected: int) -> None    # 不一致则 raise ValueError
vocab_matches_config(model_vocab_size: int) -> bool
```

### 配置依赖

| config.json 字段 | 用途 |
|-------------------|------|
| `model.vocab_size` | `assert_vocab_size()` / `vocab_matches_config()` 校验对齐 |

### 实现状态

`done` — 基于 tiktoken GPT-2 BPE 封装；encode/decode 往返一致；
`ALLOWED_SPECIAL_DEFAULT` 包含 `<|endoftext|>` 以兼容语料边界符。

### 测试覆盖

| 测试文件 | 用例 | 状态 |
|----------|------|------|
| `tests/test_tokenizer.py` | `test_vocab_size_matches_gpt2` | done |
| | `test_encode_decode_roundtrip` | done |
| | `test_endoftext_allowed_in_corpus_string` | done |
| | `test_vocab_matches_config_ok` | done |
| | `test_vocab_mismatch_raises` | done |
| | `test_empty_string_roundtrip` | done |
| | `test_all_ids_in_range` | done |
| | `test_disallow_special_raises` | done |

### 阻塞项

无。tiktoken vocab.bpe 已缓存。

### 已知 TODO

无。

---

## P1-02 · `m02_data_loader`

**源码** `src/mini_llm/m02_data_loader/__init__.py`

### 公开 API

```python
class GPTDataset(Dataset):
    def __init__(self, txt: str, max_length: int, stride: int) -> None
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]   # (input[T], target[T])

load_text(data_cfg: dict, cache_dir: Path | None = None) -> str
create_dataloader(text, batch_size, max_length, stride, shuffle, drop_last, num_workers=0) -> DataLoader
train_val_dataloaders(full_text, train_ratio, model_cfg, train_cfg) -> tuple[DataLoader, DataLoader]
```

**张量契约**：每个 sample 为 `(input[T], target[T])`，其中 `T = context_length`；
batch 后为 `[B, T]`。

### 配置依赖

| config.json 字段 | 用途 |
|-------------------|------|
| `data.url` | 语料下载地址 |
| `data.filename` | 语料文件名 |
| `data.train_ratio` | train/val 切分比例 |
| `data.max_chars` | 截取前 N 字符（null = 全量） |
| `model.context_length` | 滑动窗口长度 `max_length` 与 `stride` |
| `train.batch_size` | DataLoader batch 大小 |

### 实现状态

`done` — 滑动窗口切分、三路语料加载（环境变量 → 同级仓库 → URL 下载）、train/val split。

### 测试覆盖

| 测试文件 | 用例 | 状态 |
|----------|------|------|
| `tests/test_data_loader.py` | `test_dataset_sample_shapes` | done |
| | `test_target_is_shifted_input` | done |
| | `test_dataloader_batch_shape` | done |
| | `test_no_token_id_out_of_range` | done |
| | `test_stride_controls_overlap` | done |
| | `test_short_text_produces_empty_dataset` | done |
| | `test_train_val_split` | done |
| `tests/test_imports.py` | `test_package_importable`（import 检查） | done |

### 阻塞项

无。

### 已知 TODO

无。

---

## P1-03 · `m03_attention`

**源码** `src/mini_llm/m03_attention/__init__.py`

### 公开 API

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False)
    def forward(self, x: Tensor) -> Tensor
    # x: [B, T, d_in] -> [B, T, d_out]
```

**张量契约**：输入 `[B, T, d_in]`，输出 `[B, T, d_out]`；
因果 mask 上三角；`d_out` 必须能被 `num_heads` 整除。

### 配置依赖

| config.json 字段 | 用途 |
|-------------------|------|
| `model.emb_dim` | `d_in` 和 `d_out` |
| `model.context_length` | 因果 mask 尺寸 |
| `model.n_heads` | 头数 |
| `model.drop_rate` | attention dropout |
| `model.qkv_bias` | QKV 线性层是否带 bias |

### 实现状态

`done` — 多头因果自注意力，含上三角 mask + dropout + output projection。

### 测试覆盖

| 测试文件 | 用例 | 状态 |
|----------|------|------|
| `tests/test_attention.py` | `test_output_shape` | done |
| | `test_output_shape_shorter_sequence` | done |
| | `test_causal_mask` | done |
| | `test_gradient_is_finite` | done |
| | `test_d_out_not_divisible_by_heads_raises` | done |
| | `test_qkv_bias` | done |

### 阻塞项

无。

### 已知 TODO

无。

---

## P1-04 · `m04_model`

**源码** `src/mini_llm/m04_model/__init__.py`

### 公开 API

```python
class LayerNorm(nn.Module):    # eps=1e-5, learnable scale+shift
class GELU(nn.Module):         # 近似 GELU (tanh)
class FeedForward(nn.Module):  # Linear(emb_dim, 4*emb_dim) -> GELU -> Linear(4*emb_dim, emb_dim)
class TransformerBlock(nn.Module):
    # pre-norm: LayerNorm -> MHA -> residual -> LayerNorm -> FFN -> residual

class GPTModel(nn.Module):
    def __init__(self, cfg: dict) -> None
    def forward(self, in_idx: Tensor) -> Tensor
    # in_idx: [B, T] (token ids) -> logits: [B, T, V]
```

**张量契约**：输入 `[B, T]` int64 token ids，输出 `[B, T, vocab_size]` float logits。

### 配置依赖

| config.json 字段 | 用途 |
|-------------------|------|
| `model.vocab_size` | token embedding + LM head 维度 |
| `model.context_length` | position embedding 维度 |
| `model.emb_dim` | 隐藏层维度 |
| `model.n_heads` | 注意力头数 |
| `model.n_layers` | Transformer 层数 |
| `model.drop_rate` | embedding / residual dropout |
| `model.qkv_bias` | 传递给 MultiHeadAttention |

### 实现状态

`done` — GPTModel = tok_emb + pos_emb + dropout + N×TransformerBlock + LayerNorm + LM head。

### 测试覆盖

| 测试文件 | 用例 | 状态 |
|----------|------|------|
| `tests/test_model_forward.py` | `test_gpt_forward_shape` | done |
| | `test_generate_step` | done |

### 阻塞项

无。

### 已知 TODO

无。

---

## P1-05 · `train.py`

**源码** `train.py`（项目根目录）

### 公开 API（脚本级函数）

```python
calc_loss_batch(input_batch, target_batch, model, device) -> Tensor   # 单 batch CE loss
calc_loss_loader(data_loader, model, device, num_batches=None) -> float
evaluate_model(model, train_loader, val_loader, device, eval_iter) -> tuple[float, float]
main() -> int   # 入口；--config 指定配置文件
```

**流程**：加载 config → set seed → load text → train/val split → build model →
AdamW 优化 → 训练循环（eval + checkpoint + sample generation）。

**输出**：`runs/<run_name>/checkpoint_latest.pt`（含 model_state_dict / optimizer_state_dict / global_step / epoch / config）。

### 配置依赖

| config.json 字段 | 用途 |
|-------------------|------|
| `seed` | 随机种子 |
| `device` | `"auto"` / `"cpu"` / `"cuda"` |
| `output_dir` | checkpoint 输出根目录 |
| `run_name` | 运行名称（子目录） |
| `train.learning_rate` | AdamW lr |
| `train.weight_decay` | AdamW weight decay |
| `train.num_epochs` | 训练轮数 |
| `train.batch_size` | batch 大小 |
| `train.eval_freq` | 每 N 步打印 eval loss |
| `train.eval_iter` | eval 时取多少 batch |
| `train.checkpoint_every_steps` | 每 N 步存 checkpoint |
| `train.start_context` | 采样生成的起始文本 |

### 实现状态

`done` — 代码完整，包含损失计算、评估、checkpoint 保存、epoch 结束时的文本采样。

### 测试覆盖

| 测试文件 | 用例 | 状态 |
|----------|------|------|
| （无） | — | — |

**缺失**：无单测；L3 端到端验证需手动运行 `uv run python train.py --config configs/config.json`。

### 阻塞项

- 端到端训练依赖语料下载（`the-verdict.txt`）和 tiktoken 词表缓存。
- 在有网环境下运行一次即可解除。

### 已知 TODO

无。

---

## P2-01 · `m05_generate`

**源码** `src/mini_llm/m05_generate/__init__.py`

### 公开 API

```python
generate_text_simple(model, idx, max_new_tokens, context_size) -> Tensor
    # 贪心 argmax 采样；idx: [B, T] -> [B, T + max_new_tokens]

generate(model, idx, max_new_tokens, context_size, *, temperature=1.0, top_k=None) -> Tensor
    # temperature + top-k 采样；temperature<=0 退化为贪心
```

**张量契约**：输入 `idx: [B, T]`，输出 `[B, T + max_new_tokens]`。

### 配置依赖

不直接读 config.json；由调用方（`train.py`）传入 `context_size = model_cfg["context_length"]`。

### 实现状态

`done` — 贪心 (`generate_text_simple`) + temperature/top-k (`generate`) 两种生成模式。

### 测试覆盖

| 测试文件 | 用例 | 状态 |
|----------|------|------|
| `tests/test_model_forward.py` | `test_generate_step` | done |

### 阻塞项

无。L2/L3 验证需要先有 M1 产出的 checkpoint。

### 已知 TODO

无。

---

## 闸门状态

### M1 — 预训练闭环

| # | 前置条件 | 状态 |
|---|----------|------|
| 1 | P1-01 tokenizer 测试通过 | done |
| 2 | P1-02 data_loader batch 形状正确 | done |
| 3 | P1-03 attention 用例通过 | done |
| 4 | P1-04 model forward 形状匹配 | done |
| 5 | `pytest` 全量绿（24/24） | done |
| 6 | `train.py` 跑若干 step，train/val loss 为有限实数 | 未验证 |
| 7 | `runs/team_gpt/checkpoint_latest.pt` 写出 | 未验证 |

**解除路径**：`uv run python train.py --config configs/config.json` 验证 loss 有限 + checkpoint 写出。

### M2 — 训练→生成链路

| # | 前置条件 | 状态 |
|---|----------|------|
| 1 | M1 闸门通过 | blocked |
| 2 | 加载 checkpoint 跑 `generate`，输出非空可 decode 文本 | 未验证 |

**解除路径**：M1 通过后，加载 checkpoint 调用 `generate()` 验证输出。
