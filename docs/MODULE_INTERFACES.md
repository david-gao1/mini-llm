# 模块接口约定

实现放在 [`src/mini_llm/`](../src/mini_llm/)。分工与文件名见 [TEAM_WORK.md](./TEAM_WORK.md)。对照实现见 [`../../LLMs-from-scratch/pkg/llms_from_scratch/`](../../LLMs-from-scratch/pkg/llms_from_scratch/)（只读）。

## tokenizer（第 2 章）

- BPE / 分词；可与书中一致使用 tiktoken GPT-2，或自研 BPE；输出须与 `vocab_size`（`configs/config.json` 中 `model.vocab_size`）一致。

## data_loader（第 2 章）

- 语料读取、滑动窗口样本、`DataLoader`；`input` / `target` 与训练循环约定对齐（`[B, T]`）。

## GPTModel（第 4 章）

- `cfg`: `vocab_size`, `context_length`, `emb_dim`, `n_heads`, `n_layers`, `drop_rate`, `qkv_bias`
- `forward(in_idx) -> logits` 形状 `[B, T, vocab_size]`

## 训练一步（第 5 章）

- `input_batch`, `target_batch`: `[B, T]`
- `loss = cross_entropy(logits.flatten(0,1), target_batch.flatten())`

## 生成（第 6/7 章）

- 自回归解码；支持 temperature、top-k 等采样策略（见 `generate.py`）。
