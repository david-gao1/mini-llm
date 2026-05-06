# Pretraining (language modeling)

## Purpose

在配置的英文（或兼容）语料上，自 **BPE 分词** → **固定上下文滑动窗口** → **因果 Transformer（GPT 风格）** → **下一词交叉熵训练** 形成闭环：训练与验证损失可计算，训练权重**须**持久化到 checkpoint，供生成或其它微调复用。  
本规格覆盖 **Harness Part I（P1-01 … P1-05）** 与 **闸门 M1** 的可验收行为；**GPT-2 Medium / 大语料长线（P1-07）** 见 [REQ-P1-07](../../../docs/REQ-P1-07_GPT2Medium.md)，不阻塞 Small 闭环之 **MUST**。

## Non-goals

- **MUST NOT** 在本规格中固定「达到某 WikiText val_loss」；数值目标见运行报告与 [OWNER_CHECKLIST](../../../docs/OWNER_CHECKLIST.md)。  
- **MAY** 使用 smoke / 截断配置做快速 Harness；与「全量训练」的 loss 不可比。

## References

| 文档 | 用途 |
|------|------|
| [HARNESS.md](../../../HARNESS.md) Part I | REQ-ID、Harness 层、M1 判据 |
| [SPEC.md](../../../SPEC.md) · P1-01 … P1-05 | 公开 API、形状、测试文件 |
| [REQ-P1-01](../../../docs/REQ-P1-01_Tokenizer.md) … [REQ-P1-05](../../../docs/REQ-P1-05_Train.md) | 业务与分 REQ 边界 |

---

## Requirements

### Requirement: Token vocabulary alignment

The system **MUST** use a tokenizer whose effective vocabulary size **matches** the configured `model.vocab_size` (GPT-2 BPE convention in this project) so that encode/decode pairs are consistent and invalid-vocab surprises do not occur at model boundaries.

#### Scenario: Round-trip encoding

- **GIVEN** arbitrary text encodable under the project’s BPE settings
- **WHEN** text is encoded to token ids and decoded back
- **THEN** the result **SHALL** match the original text for that round-trip contract (as enforced by automated tests).

---

### Requirement: Causal data windows

The system **MUST** produce training batches where `input` and `target` are integer matrices of shape `[batch, context_length]`, targets are shifted by one position relative to inputs for next-token prediction, and indexing respects `context_length` and stride/split configuration without out-of-range accesses.

#### Scenario: Loader batch shape

- **GIVEN** a valid dataset configuration and `DataLoader` batching
- **WHEN** a batch is drawn
- **THEN** `input` and `target` **SHALL** have the same shape `[B, T]` with `T` equal to configured context length (or documented trim), and **SHALL** satisfy causal LM alignment.

---

### Requirement: Causal self-attention

The model’s attention **MUST** implement causal masking so that position *i* does not attend to positions *>* *i* (autoregressive constraint).

#### Scenario: Attention module tests pass

- **GIVEN** the project’s attention test suite
- **WHEN** `pytest` tests covering causal behavior run
- **THEN** they **SHALL** pass against the reference expectations (shape / mask / known tensors as defined in tests).

---

### Requirement: GPT forward pass contract

The system **SHALL** provide a GPT-style module that maps token indices `[B, T]` to logits `[B, T, V]` where `V` is `model.vocab_size`.

#### Scenario: Forward shape check

- **GIVEN** a batch of token indices matching `[B, T]`
- **WHEN** the model forward pass runs
- **THEN** logits **SHALL** have shape `[B, T, V]` with **V** consistent with configuration.

---

### Requirement: Pretraining loop and checkpoint

The system **MUST** implement a training entrypoint that computes cross-entropy next-token loss, performs evaluation on held-out data at documented intervals, and **SHALL** persist model state (and associated config metadata) to checkpoint files under the configured output directory/run name.

#### Scenario: M1 — short train completes

- **GIVEN** a valid project training configuration (e.g. small/smoke-friendly config)
- **WHEN** the training script runs for the configured steps/epochs
- **THEN** training and validation losses **SHALL** be finite real numbers (not NaN) within the monitored steps
- **AND** **SHALL** write at least one checkpoint to the configured run path (e.g. `checkpoint_latest.pt` / `checkpoint_best.pt` per implementation).

#### Scenario: Automated tests for the pretraining chain

- **GIVEN** the repository dev environment
- **WHEN** `pytest` is run for modules P1-01 … P1-04 and any integration tests prescribed in [HARNESS.md](../../../HARNESS.md)
- **THEN** the relevant tests **SHALL** pass (L0/L1/L2 as applicable).

---

## Roadmap (P1-07)

When [REQ-P1-07](../../../docs/REQ-P1-07_GPT2Medium.md) is closed, **SHALL** add or extend Requirements here for **Medium** architecture + agreed corpus path, without contradicting the Small contract above.
