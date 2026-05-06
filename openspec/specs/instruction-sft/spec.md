# Instruction supervised fine-tuning (SFT)

## Purpose

从已有 **自回归 GPT** 预训练权重出发，在 **书本格式指令数据**（`instruction` / 可选 `input` / `output`）上继续训练，使模型在「带 `### Instruction:` / `### Response:` 模板」的英文文本上预测下一 token；**不得**在 pad 位置上施加交叉熵监督。  
本规格描述 **P3-01（轨道 A）** 已具备的行为；**评测与监控增强（P3-02）** 见 [`docs/REQ-P3-02_InstructionSFTEvalAndQuality.md`](../../../docs/REQ-P3-02_InstructionSFTEvalAndQuality.md)，closure 后再将对应 SHALL 并入本文件。

## Non-goals

- **MUST NOT** 将本能力等同于「可靠事实问答」或 Chat 对齐；小数据与短训下生成质量不在本条目的保证范围内。  
- **MAY** 后续通过 [REQ-P3-01 §9](../../../docs/REQ-P3-01_Ch07InstructionSFT.md) 所列 backlog 扩展 DPO 等；**当前规格不包含**偏好学习。

## References

| 文档 | 用途 |
|------|------|
| [REQ-P3-01](../../../docs/REQ-P3-01_Ch07InstructionSFT.md) | 业务边界、双轨（Small/Medium）、验收草案 |
| [REQ-P3-01SUB](../../../docs/REQ-P3-01SUB_Ch07InstructionBookAlignment.md) | 与书本 `gpt_instruction_finetuning.py` 对齐细则 |
| [HARNESS.md](../../../HARNESS.md) Part III | 命令级 Harness 与通过判据 |
| [SPEC.md](../../../SPEC.md) · P3-01 | 公开 API、配置字段、测试文件路径 |

---

## Requirements

### Requirement: Instruction string template

The system **SHALL** assemble each training example into a single UTF-8 string that includes a fixed preamble, `### Instruction:` with the task text, optional `### Input:`, and `### Response:` followed by the reference output, consistent with the book-aligned template described in REQ-P3-01SUB.

#### Scenario: Optional input omitted

- **GIVEN** a JSON record with empty or missing `input`
- **WHEN** the template is built for encoding
- **THEN** the string **SHALL NOT** introduce a spurious non-empty Input section (format matches the reference implementation in `m07_instruction_finetune`).

---

### Requirement: Padding excluded from loss

The system **MUST** exclude padded positions from the fine-tuning cross-entropy objective using an `ignore_index` (conventionally `-100`) so that the model is not trained to predict padding tokens.

#### Scenario: Batched variable-length samples

- **GIVEN** a batch of encoded samples of unequal length collated into fixed-shape `inputs` and `targets`
- **WHEN** the loss is computed
- **THEN** targets at padding positions **SHALL** be ignored by the cross-entropy reduction (equivalently: no gradient from those positions).

---

### Requirement: Fine-tuning script and checkpoint

The system **SHALL** provide an entrypoint that loads a pretrained GPT checkpoint compatible with [`GPTModel`](../../../SPEC.md), runs instruction SFT, and persists **`checkpoint_best.pt`** (or equivalent best-validation path) under the configured run directory.

#### Scenario: Smoke configuration completes

- **GIVEN** a valid pretrained Small checkpoint (e.g. WikiText-103 `checkpoint_best.pt`) and [`configs/config_instruction_small.json`](../../../configs/config_instruction_small.json)
- **WHEN** the instruction fine-tuning script is executed with that config
- **THEN** the process **SHALL** terminate without NaN loss in normal operation
- **AND** **SHALL** write a best checkpoint file to the run output directory per configuration (`runs/<run_name>/checkpoint_best.pt`).

#### Scenario: Checkpoint carries instruction metadata

- **GIVEN** a successful instruction SFT run
- **WHEN** the best checkpoint is saved
- **THEN** the file **SHALL** include `instruction_meta` (or equivalent) documenting template identity, pad token id, ignore index, and length policy, so inference or future runs can align with training assumptions.

---

### Requirement: Automated tests for data path

The system **MUST** ship automated tests that validate instruction splitting, download/cache behavior (local path), and collate/masking behavior aligned with the book reference.

#### Scenario: Instruction finetune test module passes

- **GIVEN** the project dev environment (`uv sync --extra dev` as applicable)
- **WHEN** `pytest tests/test_instruction_finetune.py` is executed
- **THEN** all tests **SHALL** pass.

---

## Roadmap (normative target: REQ-P3-02)

The following are **not** yet required for P3-01 closure; they **SHALL** be satisfied when [REQ-P3-02](../../../docs/REQ-P3-02_InstructionSFTEvalAndQuality.md) is completed and HARNESS is updated:

- Configurable **full validation-set** loss (or explicit `eval_iter` semantics in logs).
- **Epoch-end** (or unified) best-checkpoint policy vs step-based eval only.
- Documented **paired generation** procedure (same prompt, pretrained vs SFT checkpoint, fixed decoding hyperparameters).

Until then, see REQ-P3-02 and [`docs/OWNER_CHECKLIST.md`](../../../docs/OWNER_CHECKLIST.md) Part III for interim owner checks.
