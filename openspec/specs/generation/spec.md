# Autoregressive text generation

## Purpose

从 **预训练（或其它兼容的）checkpoint** 重建自回归 GPT，对给定 **prompt token 序列** 按温度与 top-k（或等价策略）**续写**新 token，并 **SHALL** 能解码为人类可读文本。  
本规格覆盖 **Harness Part II** 中与 **P2-01**、**闸门 M2** 一致的生成侧行为。

## Non-goals

- **MUST NOT** 保证任意语言或域外 prompt 的「语义正确」；仅保证 **程序可执行、输出可 decode**。  
- **Chat / 指令遵循** 见 [`instruction-sft`](../instruction-sft/spec.md)（P3），不在本文件扩展。

## References

| 文档 | 用途 |
|------|------|
| [HARNESS.md](../../../HARNESS.md) Part II | P2-01、M2 |
| [SPEC.md](../../../SPEC.md) · P2-01、`generate_from_checkpoint.py` | API 与脚本入口 |
| [REQ-P2-01](../../../docs/REQ-P2-01_Generate.md) | 需求故事 |

---

## Requirements

### Requirement: Load checkpoint and generate continuation

The system **SHALL** load model weights and architecture configuration from a training-produced checkpoint file and **SHALL** extend an initial token sequence by autoregressive sampling or greedy decoding per configured `temperature` and `top-k` (when supported).

#### Scenario: M2 — non-empty continuation

- **GIVEN** a checkpoint compatible with the project’s `GPTModel` and a short prompt encoded to token ids
- **WHEN** the generation entrypoint runs with default or documented decoding settings
- **THEN** the output sequence **SHALL** be longer than the input (or explicitly documented as empty only on error)
- **AND** decoded text **SHALL** be non-empty in the success path (M2: 训练 → 生成链路可跑通).

#### Scenario: Decode matches tokenizer

- **GIVEN** generated token ids
- **WHEN** ids are decoded with the same codec used in training
- **THEN** the result **SHALL** be a string without raising in normal operation (subject to special-token policy documented in DOMAIN knowledge).

---

### Requirement: Tests for generation path

The system **MUST** include automated coverage that exercises generation (or generate-suitable helpers) with fixed seeds or small loops where applicable.

#### Scenario: Pytest passes

- **GIVEN** project tests that exercise `mini_llm.m05_generate.generate` (e.g. `tests/test_model_forward.py::test_generate_step`)
- **WHEN** `pytest` runs these tests
- **THEN** they **SHALL** pass.
