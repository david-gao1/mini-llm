# OpenSpec（轻量行为规格）

本目录遵循 [OpenSpec](https://openspec.dev/) 的思路：**`specs/`** 描述「系统**对外**应满足的可验收行为」；实现细节、API 签名与模块状态见根目录 [`SPEC.md`](../SPEC.md)，业务动机与故事见 [`docs/REQ-*.md`](../docs/README.md)。

| 路径 | 含义 |
|------|------|
| [`specs/`](specs/) | 现行（已合入主线的）能力规格，按领域分子目录 |
| `changes/` | （预留）进行中变更：proposal / design / tasks / delta specs |

新增或修改**可观察行为**时：先改/补对应 `specs/**/spec.md` 中的 Requirement 与 Scenario，再同步 [`HARNESS.md`](../HARNESS.md) 判据与 SPEC 中的 API 表。
