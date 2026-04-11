# 模块接口约定

实现放在 [`src/mini_llm/`](../src/mini_llm/)。对照实现见 [`../../LLMs-from-scratch/pkg/llms_from_scratch/`](../../LLMs-from-scratch/pkg/llms_from_scratch/)（只读）。

## GPTModel

- `cfg`: `vocab_size`, `context_length`, `emb_dim`, `n_heads`, `n_layers`, `drop_rate`, `qkv_bias`
- `forward(in_idx) -> logits` 形状 `[B, T, vocab_size]`

## 训练一步

- `input_batch`, `target_batch`: `[B, T]`
- `loss = cross_entropy(logits.flatten(0,1), target_batch.flatten())`

## 数据

- 滑动窗口；tokenizer 可与书中一致使用 tiktoken GPT-2。
