# changes/

进行中的变更在此按 **`openspec/changes/<change-id>/`** 建目录，典型包含：

| 文件 | 用途 |
|------|------|
| `proposal.md` | 为何做、目标 / 非目标 |
| `design.md` | 技术方案、权衡、模块边界 |
| `tasks.md` | 可勾选实施项 |
| `specs/...`（可选） | 相对 [`specs/`](../specs/) 主规格的 **增量** `spec.md` |

**合并上线后**：把增量吸收进正式 [`specs/`](../specs/)，同步 [`SPEC.md`](../../SPEC.md)、[`HARNESS.md`](../../HARNESS.md)，将目录移至 **`archive/`** 或删除。

## 归档（已合并）

| 变更 ID | 合并 destination |
|---------|------------------|
| [`archive/instruction-sft-p3-02-monitor`](archive/instruction-sft-p3-02-monitor/) | [`specs/instruction-sft/spec.md`](../specs/instruction-sft/spec.md) |

总述见上级 [`README.md`](../README.md) · 流程见 [`docs/process/openspec-workflow.md`](../../docs/process/openspec-workflow.md)。
