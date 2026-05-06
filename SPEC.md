# 项目规格与进度追踪

集中记录每个模块的 **API 契约、实现状态、测试覆盖与阻塞项**。
验收标准见 [`HARNESS.md`](HARNESS.md)，流程规范见 [`PROCESS.md`](PROCESS.md)。

> 状态标记：`done` 已完成 · `wip` 进行中 · `todo` 未开始 · `blocked` 被阻塞

## 与 OpenSpec 的关系

本仓库采用 [OpenSpec](https://openspec.dev/) 的 **轻量目录约定**：

| 层 | 路径 | 回答什么 |
|----|------|----------|
| **行为规格** | [`openspec/specs/`](openspec/specs/) | 对外可验收的 **Purpose / Requirement / Scenario**（RFC 2119 语气） |
| **SPEC（本文件）** | [`SPEC.md`](SPEC.md) | **API 签名、张量形状、配置字段、实现状态、测试表** |
| **需求故事** | [`docs/REQ-*.md`](docs/README.md) | **为何做、业务比方、边界与 backlog** |

当前已落地的首个能力规格：**[指令 SFT · `instruction-sft`](openspec/specs/instruction-sft/spec.md)**。  
引入或修改 **用户/集成方可见行为** 时，优先更新对应 `openspec/specs/**/spec.md`，再同步本文件与 [`HARNESS.md`](HARNESS.md)。

总说明见 [`openspec/README.md`](openspec/README.md)。

---

## SPEC 书写约定

**SPEC 回答什么**：「这个模块 **对外长什么样**（函数/类/形状）、**做到哪了**、**测了什么**、**还被什么卡住**」。  
**SPEC 不代替 REQ**：业务动机、为什么要做、边界故事写在 [`docs/REQ-*.md`](docs/README.md)（各 REQ 的 §1）；SPEC 里用 **一行链接** 指向对应 REQ 即可，避免两处长篇重复、日后改一处漏一处。

| 约定 | 说明 |
|------|------|
| **总览看板** | 新增或下线模块时 **必须** 同步增删一行；**阻塞项**列写 **人话**（例如「须先有 Medium checkpoint」），必要时括号注明 REQ-ID。 |
| **分节结构（每个 REQ 小节）** | 建议顺序：**源码/脚本路径** → **REQ 文档链接** → **公开 API**（签名 + 张量形状）→ **配置依赖表** → **实现状态** → **测试覆盖表** → **阻塞项**。无阻塞写 `—`。 |
| **实现状态** | 先用 **一句话可读摘要**（例如「Small 轨已通；Medium 轨依赖 P1-07」），再写状态标记 `done` / `wip` / `todo`；避免只有单词没有上下文。 |
| **文风** | **契约与形状必须精确**；解释句尽量 **短**。能用表格列清单，就少用大段散文。术语首次出现可括注白话（与 REQ 的深入浅出一致，但以 SPEC 为准绳）。 |
| **Harness** | **验收命令与通过判据** 以 [`HARNESS.md`](HARNESS.md) 为权威；SPEC 可写「见 HARNESS §…」或简短摘要，**不要**复制整张 Harness 表又不维护。 |
| **一并修改** | 改 **公开 API** 或 **实现状态** 时，同一轮提交尽量带上：**SPEC** + 相关 **REQ**（若契约段有写）+ **HARNESS**（若判据变），避免文档三角漂移。 |

REQ 文档的人话与 §1 优先级见 [`docs/process/product-design.md`](docs/process/product-design.md) **§5**。

---

## 总览看板

| 模块 | REQ | 代码 | 测试 | 阻塞项 |
|------|-----|------|------|--------|
| `m01_tokenizer` | P1-01 | done | done | — |
| `m02_data_loader` | P1-02 | done | done | — |
| `m03_attention` | P1-03 | done | done | — |
| `m04_model` | P1-04 | done | done | — |
| `train.py` | P1-05 | done | done | — |
| `train.py` 优化 | P1-06 | done | done | — |
| GPT-2 Medium + 大语料 | P1-07 | wip | todo | —（P1-06 已满足） |
| `m05_generate` | P2-01 | done | done | — |
| `m06_classify_finetune` | P2-02 | done | done | — |
| `classify_sms.py` | P2-03 | done | done | — |
| `m07_instruction_finetune` | P3-01 | done | done | 轨道 B：须先有 Medium checkpoint（P1-07） |
| 指令 SFT 质检 / 监控 / 对照脚本 | P3-02 | todo | todo | 见 [`docs/REQ-P3-02_InstructionSFTEvalAndQuality.md`](docs/REQ-P3-02_InstructionSFTEvalAndQuality.md) |
| **闸门 M1** | — | — | done | — |
| **闸门 M2** | — | — | done | — |

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
train_val_dataloaders(full_text, train_ratio, model_cfg, train_cfg, cache_dir=None) -> tuple[DataLoader, DataLoader]
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

`done` — 滑动窗口切分、语料加载（环境变量 → 同级仓库 → URL / HuggingFace）、train/val split、可选 `cache_dir` 下的 token `.pt` 缓存。

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
| 6 | `train.py` 跑若干 step，train/val loss 为有限实数 | done |
| 7 | `runs/team_gpt/checkpoint_latest.pt` 写出 | done |

已验证：100 epoch 训练完成，train_loss=0.005，checkpoint 正常写出。

### M2 — 训练→生成链路

| # | 前置条件 | 状态 |
|---|----------|------|
| 1 | M1 闸门通过 | done |
| 2 | 加载 checkpoint 跑 `generate`，输出非空可 decode 文本 | done |

已验证：生成文本通顺可 decode，Epoch 30 后输出完整英语句子。

---

## P1-06 · `train.py` 训练优化

**源码** `train.py`（项目根目录）
**REQ 文档** [`docs/REQ-P1-06_TrainOptimize.md`](docs/REQ-P1-06_TrainOptimize.md)

### 优化内容

| # | 优化项 | 说明 |
|---|--------|------|
| 1 | MPS 设备支持 | `_pick_device` 增加 MPS 分支，M3 Max 加速 |
| 2 | 梯度裁剪 | `clip_grad_norm_` 防止梯度爆炸 |
| 3 | 学习率调度 | warmup + cosine annealing 衰减 |
| 4 | Early stopping | val_loss 不降时提前终止 + best checkpoint |
| 5 | 采样多样性 | `print_sample` 改用 temperature + top-k |

### 配置新增字段

| config.json 字段 | 默认值 | 用途 |
|-------------------|--------|------|
| `train.grad_clip` | 1.0 | 梯度裁剪阈值 |
| `train.warmup_ratio` | 0.1 | warmup 占总步数比例 |
| `train.min_lr_ratio` | 0.1 | cosine 衰减到 lr 的下限比例 |
| `train.patience` | 10 | early stopping 容忍次数（0=不启用） |

### 实现状态

`done` — 已实现。

### 测试覆盖

24/24 全绿（未引入新测试，复用现有套件）。

### 阻塞项

无。

---

## P1-07 · GPT-2 Medium + WikiText-103（raw）

**配置** `configs/config_medium.json`（`Salesforce/wikitext` / `wikitext-103-raw-v1`，HuggingFace 下载后缓存为本地 txt）  
**REQ 文档** [`docs/REQ-P1-07_GPT2Medium.md`](docs/REQ-P1-07_GPT2Medium.md)

### 升级内容

| 项 | 原配置（config.json） | 新配置（config_medium.json） |
|----|:---:|:---:|
| 语料 | the-verdict.txt (~20KB) | WikiText-103 raw train（~500MB，HF 缓存至 `data_cache/`） |
| emb_dim | 384 | 1024 |
| n_heads | 6 | 16 |
| n_layers | 6 | 24 |
| context_length | 256 | 1024 |
| 参数量 | ~29M | ~406M（tok_emb 与 out_head 独立，无 weight tying） |
| batch_size | 8 | 1（另 `gradient_accumulation_steps=4`，等效 batch=4） |
| 内存估算 | ~0.7 GB | ~9–10 GB（batch=1；大 batch 易 OOM） |

### 实现状态

`wip` — Medium 配置 + 大语料数据管道与训练已在跑；完整验收见 REQ-P1-07。

### 阻塞项

依赖 P1-06 已完成（MPS / scheduler / early stopping）。无新增阻塞。

---

## P2-02 · `m06_classify_finetune`（SMS ham/spam 微调 → 分类 checkpoint）

**源码** `src/mini_llm/m06_classify_finetune/__init__.py`  
**微调脚本** `finetune_classify.py`  
**REQ 文档** [`docs/REQ-P2-02_ClassifyFinetune.md`](docs/REQ-P2-02_ClassifyFinetune.md)  
**单条短信判别（CLI）** 见 **P2-03** · [`docs/REQ-P2-03_ClassifySmsInfer.md`](docs/REQ-P2-03_ClassifySmsInfer.md)（`classify_sms`：stdout `ham`|`spam`）

### 公开 API（训练 / 评估）

```python
download_and_prepare_spam(data_dir: Path) -> tuple[Path, Path, Path]

class SpamDataset(Dataset):
    def __init__(self, csv_path, max_length=None, pad_token_id=50256)
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]  # (token_ids[T], label)

calc_loss_batch(input_batch, target_batch, model, device) -> Tensor
calc_loss_loader(loader, model, device, num_batches=None) -> float
evaluate_model(model, train_loader, val_loader, device, eval_iter) -> tuple[float, float]
calc_accuracy_loader(loader, model, device, num_batches=None) -> float
```

**张量契约**：每个 sample 为 `(token_ids[max_length], label)`；分类 logits 取 `model(batch)[:, -1, :]`。

### 配置依赖

| config.json 字段 | 用途 |
|-------------------|------|
| `pretrained_checkpoint` | 预训练 checkpoint 路径 |
| `data.data_dir` | SMS 数据缓存目录 |
| `finetune.num_classes` | 分类类别数 |
| `finetune.num_epochs` | 微调轮数 |
| `finetune.batch_size` | 批大小 |
| `finetune.learning_rate` | AdamW lr |
| `finetune.weight_decay` | 权重衰减 |
| `finetune.eval_freq` | 每 N 步评估 loss |
| `finetune.eval_iter` | 评估取 batch 数 |
| `finetune.unfreeze_last_n_blocks` | 解冻末尾 Transformer block 数 |

### 实现状态

`done` — SMS 微调、`eval_classify`、训练末段 test 混淆矩阵 / spam PRF1 / FN CSV 与 HARNESS 判据已满足；可选 backlog 见 REQ-P2-02 **§10**。

### 测试覆盖（训练侧）

| 测试文件 | 用例 | 状态 |
|----------|------|------|
| `tests/test_classify_finetune.py` | `test_spam_dataset_shapes` … `test_calc_loss_batch_finite`（共 6） | done |
| `tests/test_classify_metrics.py` | 混淆 / PRF / `collect_predictions_loader` / FN 导出 / 探针 JSON | done |

推理编码 / checkpoint 加载的 2 个用例归入 **P2-03**。

### 阻塞项

依赖预训练 checkpoint（如 `runs/gpt2_small_wikitext103/checkpoint_best.pt`）。

**可选增强（非阻塞）**：详见 [REQ-P2-02 §10](docs/REQ-P2-02_ClassifyFinetune.md)（`BL-P2-02-02` 已完成；`BL-P2-02-03`～`BL-P2-02-05` 仍待有空实现）。

---

## P2-03 · `classify_sms.py`（单行英文短信 → `ham` / `spam`）

**脚本** [`classify_sms.py`](classify_sms.py)  
**REQ 文档** [`docs/REQ-P2-03_ClassifySmsInfer.md`](docs/REQ-P2-03_ClassifySmsInfer.md)

### 公开 API（模块内推理辅助）

```python
encode_spam_text_for_model(text: str, max_length: int, *, pad_token_id=50256) -> Tensor
    # [1, max_length]

load_spam_classifier_checkpoint(path: Path | str, device: torch.device) -> tuple[nn.Module, dict]
```

### classify_sms.py（CLI）

- **输入**：`--checkpoint`（默认 `runs/spam_classify_phase_b/checkpoint_best.pt`）；`--text` 或 stdin；可选 `--device`、`--max-length`、`--probs`。
- **输出**：stdout 单行 `ham` | `spam`。

### 分类 checkpoint（读取）

完整字段表见 REQ-P2-03 §4；须含分类 `out_head` 与（推荐）`spam_max_length`。

### 实现状态

`done` — 与 REQ-P2-03 一致。

### 测试覆盖

| 测试文件 | 用例 | 状态 |
|----------|------|------|
| `tests/test_classify_finetune.py` | `test_encode_spam_text_matches_dataset_row`、`test_load_spam_classifier_checkpoint_roundtrip` | done |

### 阻塞项

依赖 **P2-02** 产出的分类 checkpoint（非裸预训练 LM）。

---

## Part III（第 7 章 · 指令微调 SFT）

## P3-01 · `m07_instruction_finetune` — Ch7 指令 SFT（双轨 Small / Medium）

**源码** `src/mini_llm/m07_instruction_finetune/__init__.py`  
**脚本** [`finetune_instruction.py`](../finetune_instruction.py)  
**配置** [`configs/config_instruction_small.json`](../configs/config_instruction_small.json)（Small + `smoke_trim`）、[`configs/config_instruction_medium.json`](../configs/config_instruction_medium.json)（全量数据；依赖 Medium checkpoint）  
**REQ 文档** [`docs/REQ-P3-01_Ch07InstructionSFT.md`](docs/REQ-P3-01_Ch07InstructionSFT.md) · **书本对齐细则** [`docs/REQ-P3-01SUB_Ch07InstructionBookAlignment.md`](docs/REQ-P3-01SUB_Ch07InstructionBookAlignment.md)  
**OpenSpec（行为契约）** [`openspec/specs/instruction-sft/spec.md`](openspec/specs/instruction-sft/spec.md)

### 公开 API

```python
format_input(entry: dict) -> str
split_instruction_entries(data, *, train_ratio=0.85, test_ratio=0.1) -> tuple[list, list, list]
download_instruction_json(cache_path: Path, url: str) -> list[dict]

class InstructionDataset(Dataset):
    def __init__(self, data, encode_fn: Callable[[str], list[int]]) -> None

instruction_collate_fn(batch, *, pad_token_id=50256, ignore_index=-100,
    allowed_max_length=None, device=None) -> tuple[Tensor, Tensor]
make_instruction_collate_fn(...) -> Callable  # DataLoader collate_fn
```

### 实现状态

`done`（**轨道 A**：Small + 书本 JSON + `finetune_instruction.py`；**轨道 B**：配置已备，仍依赖 **P1-07** Medium checkpoint）。**DPO** 见 REQ-P3-01 §9。

### 测试覆盖

| 测试文件 | 用例 | 状态 |
|----------|------|------|
| `tests/test_instruction_finetune.py` | `format_input`、划分、`collate` 形状与截断、`InstructionDataset`、本地 JSON | done |

### 阻塞项

- **轨道 B**：依赖 **P1-07** 产出可用 Medium 预训练 checkpoint（[`configs/config_instruction_medium.json`](../configs/config_instruction_medium.json) 内路径须存在）。

---

## P3-02 · 指令 SFT 效果检验、训练监控与质量优化

**REQ** [`docs/REQ-P3-02_InstructionSFTEvalAndQuality.md`](docs/REQ-P3-02_InstructionSFTEvalAndQuality.md)  
**OpenSpec** 规划中需求见 [`openspec/specs/instruction-sft/spec.md`](openspec/specs/instruction-sft/spec.md) **§ Roadmap (REQ-P3-02)**；本条 closure 后应将对应条升格为正式 **Requirement**。  
**依赖** P3-01（训练管线已实现）  
**状态** `todo`：全 val 评估、epoch 末与 best 对齐、正式训练用 JSON 配置、固定 prompt 的双 checkpoint 对照生成等，见 REQ **§4** 阶段 A/B/C。

（不要求新增 `m08` 模块；实现可落在 `finetune_instruction.py`、小脚本与文档。）
