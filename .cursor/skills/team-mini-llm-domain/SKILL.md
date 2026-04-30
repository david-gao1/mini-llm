---
name: team-mini-llm-domain
description: >-
  Enforces team-mini-llm module boundaries, train orchestration, shared terms,
  and checkpoint-generation pitfalls. Use when editing mini_llm (m01–m05),
  train.py, generate_from_checkpoint.py, configs, or docs; when discussing
  WikiText training, token cache, GPTDataset, bounded contexts / 模块边界,
  DOMAIN-KNOWLEDGE, or architecture of this repo.
---

# team-mini-llm 领域与模块边界

## 何时读 canonical 文档

涉及全貌、踩坑、性能细节时，打开仓库根目录 **`docs/DOMAIN-KNOWLEDGE.md`**（优先 **§1.3 模块边界与职责划分**、**§6.4 / §6.5**）。

## 模块边界（改代码前对齐）

| 代码区域 | 职责 | 约定 |
|----------|------|------|
| **m01_tokenizer** | 字符串 ↔ token ID | 不耦合 Dataset / 优化器 |
| **m02_data_loader** | 语料、`(input, target)`、`DataLoader`、可选 **token `.pt` 缓存** | 输出与 `context_length`、batch 对齐 |
| **m03_attention** | `MultiHeadAttention` | **仅由 m04 组装**；不要在 `train.py` 里直接依赖 |
| **m04_model** | `GPTModel`：`[B,T]` → logits `[B,T,V]` | 不读 config 文件、不拉 HuggingFace |
| **m05_generate** | temperature / top-k 自回归采样 | 只调用 `model.forward` |
| **train.py** | 训练循环、eval、checkpoint、梯度累积 | **不实现** Attention / FFN / LayerNorm |
| **generate_from_checkpoint.py** | 加载 ckpt + 推理 | **不塞入**仅训练需要的逻辑 |

**跨模块**：优先只传递 **张量 + shape + dtype**。**预训练编排**以 **`train.py`** 为单一入口，避免复制整套循环。

## 术语（与文档一致）

- **micro-step**：单次 forward + backward（常为 batch_size=1 的一步）。
- **optimizer step**：累积 **`gradient_accumulation_steps`** 次 micro-step 后再 `optimizer.step()`；**`global_step`** 通常指它。
- **effective batch**：`batch_size × gradient_accumulation_steps`。
- **token cache**：`runs/<run_name>/data_cache/train_tokens.pt`、`val_tokens.pt`。

## 生成 / 检验 checkpoint

- **英文 WikiText 模型**：检验用 **英文** `--prompt`；中文开头易产生误判（见 DOMAIN-KNOWLEDGE **§6.4**、`docs/RUN_REPORT_gpt2_small_wikitext103.md` **第七节**）。
- **`--temperature 0`**：贪心 decoding，易出现 **同一短语重复**；不等于「更好」。调试可复现时用；日常观感多用默认温度 + top-k。详见 **`docs/REQ-P2-01_Generate.md` §8**。
- **多行 shell**：除最后一行外每行行尾 **`\`**，否则 `--prompt` 可能被 shell 当成命令。

## 提交前自检

- [ ] 改动是否限制在单一模块边界内，或仅在编排层？
- [ ] 新增跨模块调用是否仍满足 **`[B,T]` / `[B,T,V]`** 等契约？
- [ ] 训练路径是否仍主要经 **`train.py`**？

## 延伸阅读

- [DOMAIN-KNOWLEDGE.md](../../../docs/DOMAIN-KNOWLEDGE.md) — 全文
- [reference.md](reference.md) — 文档索引与 REQ 锚点
