# OpenSpec（轻量行为规格）

本目录遵循 [OpenSpec](https://openspec.dev/) 的思路：**`specs/`** 描述「系统**对外**应满足的可验收行为」；实现细节、API 签名与模块状态见根目录 [`SPEC.md`](../SPEC.md)，业务动机与故事见 [`docs/REQ-*.md`](../docs/README.md)。

| 路径 | 含义 |
|------|------|
| [`specs/`](specs/) | 现行（已合入主线的）能力规格，按领域分子目录 |
| `changes/` | （预留）进行中变更：`proposal.md`（提案）、`design.md`（设计）、`tasks.md`（任务清单）、delta `specs/` |

新增或修改**可观察行为**时：先改/补对应 `specs/**/spec.md` 中的 **需求**与**场景**，再同步 [`HARNESS.md`](../HARNESS.md) 判据与 SPEC 中的 API 表。维护节奏见 [`docs/process/openspec-workflow.md`](../docs/process/openspec-workflow.md)。

## 能力目录（`specs/`）

| 目录 | Harness / REQ | 说明 |
|------|----------------|------|
| [`pretraining/spec.md`](specs/pretraining/spec.md) | Part I · P1-01 … P1-05、M1 | 分词 → DataLoader → 注意力 → GPT → `train.py` 与 checkpoint |
| [`generation/spec.md`](specs/generation/spec.md) | P2-01、M2 | 自回归续写与加载 checkpoint |
| [`classify-sms/spec.md`](specs/classify-sms/spec.md) | P2-02、P2-03 | 短信 ham/spam 微调与单条命令行推理 |
| [`instruction-sft/spec.md`](specs/instruction-sft/spec.md) | Part III · P3-01、P3-02 路线图 | 指令 JSON 监督微调与后续质检 |
