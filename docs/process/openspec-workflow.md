# OpenSpec 维护流程（与本仓库配合）

与 [`openspec/README.md`](../../openspec/README.md)、根目录 [`SPEC.md`](../../SPEC.md)、[`HARNESS.md`](../../HARNESS.md) 一起阅读。

---

## 1. 三层分工（先改哪份）

| 层次 | 路径 | 何时动笔 |
|------|------|----------|
| **行为规格** | `openspec/specs/**/spec.md` | 增加/变更 **对外可观察、可验收** 的行为（用户、集成方、Harness 能判断真假） |
| **API / 状态** | `SPEC.md` | 改 **函数签名、张量形状、配置键、实现状态、测试文件表** |
| **业务叙事** | `docs/REQ-*.md` | 改 **动机、比方、边界、分阶段交付、backlog**；REQ 头部应链到对应 OpenSpec |

**原则**：同一条规则 **不要**在 REQ 故事与 OpenSpec 需求里各写一长段且互不同步；**验收句式**以 OpenSpec **需求 + 场景** 为准，REQ 用白话指过去即可。

---

## 2. 措辞约定（中文 spec）

各 `spec.md` 正文使用中文。语气与力度对应关系建议为：

- **必须**：强制（相当于 MUST/SHALL）  
- **应当**：强建议且默认遵守（相当于 SHOULD）  
- **不得**：禁止（相当于 MUST NOT）  
- **可以**：允许、非必选（相当于 MAY）  

场景建议用 **给定 / 当 / 那么**（或 **且**）结构，便于和测试用例对照。

---

## 3. 变更目录 `openspec/changes/`（可选）

进行中的大改（跨多文件、需单独评审）可建：

`openspec/changes/<change-id>/proposal.md`（为何做）  
`design.md`（技术方案）  
`tasks.md`（检查项）  
`specs/...`（相对现行 `openspec/specs/` 的 **增量** 描述）

**合并上线后**：把增量吸收进正式 `openspec/specs/**/spec.md`，同步 `SPEC.md` / `HARNESS.md`，将 change 目录移至 `archive/` 或删除（团队自定）。未启用 CLI 时，**手工维护**同样结构即可。

---

## 4. 最小检查清单（提交前）

1. 行为有变 → **`openspec/specs`** 是否已改 **需求/场景**？  
2. 判据有变 → **`HARNESS.md`** 是否已改？  
3. 对外 API 有变 → **`SPEC.md`** 是否已改？  
4. 业务故事需对齐 → **`docs/REQ-*.md`** 头部或 §1 是否已提一句？  

---

## 5. 能力规格与 REQ 的粗映射

与 [`docs/README.md`](../README.md) **Harness × OpenSpec × REQ 总览**一致；下行便于复制检索。

| OpenSpec 目录 | Harness | 闸门（若有） | 主要 REQ |
|---------------|---------|--------------|----------|
| `pretraining/` | Part I | **M1** | REQ-P1-01 … P1-05（契约主干）；P1-06、P1-07 见各自 REQ + pretraining **路线图** |
| `generation/` | Part II | **M2** | REQ-P2-01 |
| `classify-sms/` | Part II | — | REQ-P2-02、P2-03 |
| `instruction-sft/` | Part III | — | REQ-P3-01、P3-01SUB；**P3-02** 行为已写入 [`openspec/specs/instruction-sft/spec.md`](../../openspec/specs/instruction-sft/spec.md)（归档见 [`openspec/changes/archive/instruction-sft-p3-02-monitor`](../../openspec/changes/archive/instruction-sft-p3-02-monitor/)） |

（映射不是一一穷尽；细分以各 REQ 头部 **OpenSpec（行为契约）** 链为准。）
