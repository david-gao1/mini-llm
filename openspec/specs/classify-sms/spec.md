# SMS spam classification (微调与推理)

## Purpose

在 **预训练 GPT 表征** 之上，用带标签的 **英文短信** 数据训练 **ham/spam 二分类**，导出含分类头与必要元数据（如 `spam_max_length`）的 checkpoint，并提供 **单行短信 → stdout 标签** 的 CLI。  
本规格覆盖 **Harness Part II** 中 **P2-02**、**P2-03** 的可验收行为。

## Non-goals

- **MUST NOT** 在本规格中固定「达到某商业垃圾邮件召回」；阈值与混淆矩阵解读见 [OWNER_CHECKLIST](../../../docs/OWNER_CHECKLIST.md) 与 [REQ-P2-02](../../../docs/REQ-P2-02_ClassifyFinetune.md)。  
- **多标签 / 多类** 扩展不在本文件范围。

## References

| 文档 | 用途 |
|------|------|
| [HARNESS.md](../../../HARNESS.md) Part II | P2-02、P2-03 判据 |
| [SPEC.md](../../../SPEC.md) · P2-02、P2-03 | API、测试文件 |
| [REQ-P2-02](../../../docs/REQ-P2-02_ClassifyFinetune.md)、[REQ-P2-03](../../../docs/REQ-P2-03_ClassifySmsInfer.md) | 业务与 backlog |

---

## Requirements

### Requirement: Classification fine-tuning produces a checkpoint

The system **SHALL** fine-tune (or train from a compatible pretrained LM) to minimize classification loss on labeled SMS data and **SHALL** persist `checkpoint_best.pt` (or equivalent) including classification head weights and metadata required to reproduce inference (e.g. max sequence length for SMS encoding).

#### Scenario: Training completes with reported metrics

- **GIVEN** a valid classify config and pretrained backbone path when required
- **WHEN** the classification fine-tuning script runs to completion
- **THEN** the process **SHALL** terminate without unhandled failure in normal operation
- **AND** **SHALL** report test-set metrics (e.g. accuracy, confusion counts) consistent with [HARNESS.md](../../../HARNESS.md) expectations
- **AND** **SHALL** write a best checkpoint under the configured run directory.

#### Scenario: Test accuracy threshold (demonstration config)

- **GIVEN** the project’s agreed demonstration configuration (e.g. phase_b default path in docs)
- **WHEN** full training completes on the canonical split
- **THEN** test accuracy **SHALL** meet or exceed **90%** as recorded in Harness / REQ (subject to same data and split as documented).

---

### Requirement: Single-message inference CLI

The system **SHALL** provide a CLI that loads a **classification** checkpoint (not a raw LM-only checkpoint without head), encodes one English SMS string, and prints exactly **`ham`** or **`spam`** to standard output.

#### Scenario: CLI runs with default or explicit checkpoint

- **GIVEN** a path to a valid classify checkpoint and a sample English SMS text
- **WHEN** the inference script runs
- **THEN** stdout **SHALL** be `ham` or `spam` (single line label as specified)
- **AND** encoding **SHALL** match the dataset preprocessing contract.

---

### Requirement: Automated tests

The project **MUST** ship tests for classification dataset wiring, forward+loss, metrics, and CLI behavior as listed in SPEC.

#### Scenario: Pytest passes

- **GIVEN** dev environment with test dependencies
- **WHEN** `pytest tests/test_classify_finetune.py` (prescribed cases) and related metric tests run
- **THEN** they **SHALL** pass.
