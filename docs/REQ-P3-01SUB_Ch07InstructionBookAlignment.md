# REQ-P3-01SUB：第七章指令 SFT —「跟书走」对齐细则

**所属**：[REQ-P3-01 · Ch7 指令 SFT](REQ-P3-01_Ch07InstructionSFT.md)（本子 REQ 只讲「书上脚本 ↔ 本仓库」映射，不写业务比方）  
**基准**：同级仓库 [`LLMs-from-scratch/ch07/01_main-chapter-code/gpt_instruction_finetuning.py`](../../LLMs-from-scratch/ch07/01_main-chapter-code/gpt_instruction_finetuning.py) 的 **`main()` 主线**。书中的 **`if args.test_mode`**（随机 Tiny GPT）是单机自检分支，**不在本条对照范围内**。

---

## 1. 流水线（四段）

| 段 | 书里 | 本仓库 |
|----|------|--------|
| **① 数据** | `download_and_load_file`；`train_portion`/`test_portion`（默认 85% / 10%，余下 val）；`test_mode` 时三段各取前 10 条 | [`download_instruction_json`](../src/mini_llm/m07_instruction_finetune/__init__.py)；[`split_instruction_entries`](同上)；[`instruction_finetune.smoke_trim`](../configs/config_instruction_small.json)（划分后各段截前 N 条；设 **10** 可对齐书的数目） |
| **② 编码** | `format_input` + `### Response:\n` + `output` → `tiktoken("gpt2").encode` | 同名模板 [`format_input`](同上)；[`InstructionDataset`](同上) + [`encode_text`](../src/mini_llm/m01_tokenizer/__init__.py)（等价 gpt2 编码） |
| **③ Collate** | `batch_max_length = max(len+1)`；尾追加 **50256**；pad；`inputs[:-1]` / `targets[1:]`；pad 位保留 **第一个**监督、其余 **`ignore_index=-100`**；`allowed_max_length` 截断 | [`instruction_collate_fn`](同上)：公式一致 |
| **④ 训练** | `cross_entropy(..., ignore_index)`；DataLoader（train shuffle+drop_last；eval 否）；`AdamW` 全参数；`train_model_simple` | [`calc_loss_batch_instruction`](../finetune_instruction.py)；[`finetune_instruction.py`](../finetune_instruction.py) 里 Loader 与优化器逻辑对齐 |

---

## 2. 刻意不同（不算「没跟书」）

| 点 | 说明 |
|----|------|
| **起点权重** | 书：`download_and_load_gpt2` / OpenAI 权重映射。我们：**[`train.py`](../train.py) 预训练 checkpoint** + `load_state_dict`。 |
| **`allowed_max_length`** | 书 Main 常写 **1024**。我们：默认 **512**，且不得超过 **`model.context_length`**（超限会在脚本里收紧并打 stderr）。 |
| **训练循环** | 书：`train_model_simple` 且可夹 **`generate` 打样**。我们：手写循环 + **eval_freq**，**未**在训练中批量打样；要对齐可自行接 `generate`。 |
| **文末批量导出** | 书：对 test 逐条 `generate` 写 `instruction-data-with-response*.json`。我们：**未实现**；先用 [`generate_from_checkpoint.py`](../generate_from_checkpoint.py)；backlog 见 P3-01 §9 **BL-P3-01-02**。 |

---

## 3. 建议核对顺序

1. **②～③**：模板、`InstructionDataset`、`instruction_collate_fn` —— 决定「训练题」是否与书一致。  
2. **①**：划分比例与 `smoke_trim` 是否满足你对照实验的需要。  
3. **④ + §2**：优化器与 Loader 对齐后，再看权重来源与 §2 里「刻意不同」的几项。

---

## 4. 修订记录

| 日期 | 变更 |
|------|------|
| 2026-05-01 | 从 REQ-P3-01 拆出；表格压缩为四段流水线 + 刻意差异表。 |
