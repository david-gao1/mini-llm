# 文档索引

与根目录 [`README.md`](../README.md)、[`HARNESS.md`](../HARNESS.md)、[`PROCESS.md`](../PROCESS.md) 配合阅读。

## 规范体系（SPEC · OpenSpec · REQ · HARNESS）

| 类型 | 位置 | 用途 |
|------|------|------|
| **OpenSpec** | [`openspec/specs/`](../openspec/specs/) | 行为 **目的 / 需求 / 场景**（可与 AI 评审对齐验收口径） |
| **SPEC** | [`SPEC.md`](../SPEC.md) | **API、形状、实现状态、测试覆盖** |
| **REQ** | 下表与 `REQ-*.md` | **业务动机、边界、分阶段交付与 backlog** |
| **HARNESS** | [`HARNESS.md`](../HARNESS.md) | **可执行命令与通过判据** |

引读：[`openspec/README.md`](../openspec/README.md) · 行为规格：[pretraining](../openspec/specs/pretraining/spec.md) · [generation](../openspec/specs/generation/spec.md) · [classify-sms](../openspec/specs/classify-sms/spec.md) · [instruction-sft](../openspec/specs/instruction-sft/spec.md)。

### Harness × OpenSpec × REQ 总览

**验收句式**（必须 / 应当 / 场景）以对应 **`openspec/specs/**/spec.md`** 为准；**命令与数值判据**以 [`HARNESS.md`](../HARNESS.md) 为准；**函数签名与形状**以 [`SPEC.md`](../SPEC.md) 为准；下表用于一眼对齐「闸门 → 规格文件 → 叙事 REQ」。

| Harness | 闸门 | OpenSpec | 主要 REQ-ID |
|---------|------|----------|-------------|
| Part I | **M1** | [`pretraining/spec.md`](../openspec/specs/pretraining/spec.md) | P1-01 … P1-05（**主线契约**）；P1-06 / P1-07 见同 spec **非目标 / 路线图**与各自 REQ |
| Part II | **M2** | [`generation/spec.md`](../openspec/specs/generation/spec.md)（加载 checkpoint 续写） | **P2-01**（依赖 Part I checkpoint） |
| Part II | — | [`classify-sms/spec.md`](../openspec/specs/classify-sms/spec.md) | P2-02（微调产出）、P2-03（`classify_sms` 推理） |
| Part III | — | [`instruction-sft/spec.md`](../openspec/specs/instruction-sft/spec.md) | P3-01、P3-01SUB（对齐细则）；P3-02（Small-only 质检闭环） |

---

## 流程规范

一人全栈（产品 + 开发 + 测试）的闭环流程，总纲见 [`PROCESS.md`](../PROCESS.md)。

| 文档 | 说明 |
|------|------|
| [process/product-design.md](process/product-design.md) | 产品设计流程：从想法到 REQ 条目 |
| [process/openspec-workflow.md](process/openspec-workflow.md) | OpenSpec：行为规格与 SPEC/REQ/HARNESS 的维护顺序、`changes/` 约定 |
| [process/development.md](process/development.md) | 开发流程规范：编码规范、提交规范、代码审查 |
| [process/testing.md](process/testing.md) | 测试流程规范：测试分层、用例编写、回归策略 |
| [process/iteration.md](process/iteration.md) | 迭代与发布流程：版本管理、技术债、复盘 |

## REQ 需求文档

**写法约定**：§1 统一采用「先打个比方 → 最关键的一句话 → （按需）再往下看」，范例 [**REQ-P3-01**](REQ-P3-01_Ch07InstructionSFT.md)；总则见 [process/product-design.md](process/product-design.md) **§5**。每条 REQ 文首应有 **`OpenSpec（行为契约）`**，指向对应 `spec.md`（详见 [process/openspec-workflow.md](process/openspec-workflow.md)）。

| 文档 | REQ | OpenSpec | 状态 |
|------|-----|----------|------|
| [REQ-P1-01_Tokenizer.md](REQ-P1-01_Tokenizer.md) | P1-01 GPT-2 BPE 分词器 | [pretraining](../openspec/specs/pretraining/spec.md) | ✅ 已完成 |
| [REQ-P1-02_DataLoader.md](REQ-P1-02_DataLoader.md) | P1-02 滑动窗口 Dataset 与 DataLoader | [pretraining](../openspec/specs/pretraining/spec.md) | ✅ 已完成 |
| [REQ-P1-03_Attention.md](REQ-P1-03_Attention.md) | P1-03 多头因果自注意力 | [pretraining](../openspec/specs/pretraining/spec.md) | ✅ 已完成 |
| [REQ-P1-04_Model.md](REQ-P1-04_Model.md) | P1-04 GPT 模型 | [pretraining](../openspec/specs/pretraining/spec.md) | ✅ 已完成 |
| [REQ-P1-05_Train.md](REQ-P1-05_Train.md) | P1-05 预训练循环 | [pretraining](../openspec/specs/pretraining/spec.md) | ✅ 已完成 |
| [REQ-P1-06_TrainOptimize.md](REQ-P1-06_TrainOptimize.md) | P1-06 训练优化（过拟合治理 + 设备加速） | [pretraining](../openspec/specs/pretraining/spec.md) | ✅ 已完成 |
| [REQ-P1-07_GPT2Medium.md](REQ-P1-07_GPT2Medium.md) | P1-07 GPT-2 Medium + WikiText（规格见 pretraining **路线图**） | [pretraining](../openspec/specs/pretraining/spec.md) | todo |
| [REQ-P2-01_Generate.md](REQ-P2-01_Generate.md) | P2-01 `m05_generate` / checkpoint 续写；**闸门 M2** | [generation](../openspec/specs/generation/spec.md) | ✅ 已完成 |
| [REQ-P2-02_ClassifyFinetune.md](REQ-P2-02_ClassifyFinetune.md) | P2-02 finetune_classify：SMS ham/spam checkpoint（供 P2-03） | [classify-sms](../openspec/specs/classify-sms/spec.md) | ✅ 已完成；§10 为可选增强 backlog |
| [REQ-P2-03_ClassifySmsInfer.md](REQ-P2-03_ClassifySmsInfer.md) | P2-03 `classify_sms`：单行短信 → stdout | [classify-sms](../openspec/specs/classify-sms/spec.md) | ✅ 已完成 |
| [REQ-P3-01_Ch07InstructionSFT.md](REQ-P3-01_Ch07InstructionSFT.md) | P3-01 Ch7 指令 SFT；Small 底座已实现 | [instruction-sft](../openspec/specs/instruction-sft/spec.md) | ✅ 已完成；Medium 不作为本轮 SFT 底座 |
| [REQ-P3-01SUB_Ch07InstructionBookAlignment.md](REQ-P3-01SUB_Ch07InstructionBookAlignment.md) | **P3-01 子**：书本脚本 ↔ 仓库（细则，与同 OpenSpec 同源） | [instruction-sft](../openspec/specs/instruction-sft/spec.md) | 细则文档 |
| [REQ-P3-02_InstructionSFTEvalAndQuality.md](REQ-P3-02_InstructionSFTEvalAndQuality.md) | P3-02 指令 SFT **质检闭环**：全 val、对照生成、仅评估入口 | [instruction-sft](../openspec/specs/instruction-sft/spec.md) | ✅ 已完成；自动评分 / 批量导出为 backlog |

| 文档 | 说明 |
|------|------|
| [PROJECT_OVERVIEW_EVERYONE.md](PROJECT_OVERVIEW_EVERYONE.md) | 全员读得懂版：项目做了什么、有哪些能力、怎么演示 |
| [DOMAIN-KNOWLEDGE.md](DOMAIN-KNOWLEDGE.md) | DDD 视角项目知识、踩坑汇总（含 checkpoint 生成：**中英文 prompt**、`temperature=0` 贪心易重复、**shell 续行 `\`**） |
| [../.cursor/skills/team-mini-llm-domain/SKILL.md](../.cursor/skills/team-mini-llm-domain/SKILL.md) | Cursor Agent Skill：将边界与约定压缩为可执行清单（与 DOMAIN-KNOWLEDGE 对齐） |
| [RUN_REPORT_gpt2_small_wikitext103.md](RUN_REPORT_gpt2_small_wikitext103.md) | GPT-2 Small × WikiText-103 运行报告；**第七节**为加载 checkpoint 与生成参数说明 |
| [RUN_REPORT_instruction_sft_small.md](RUN_REPORT_instruction_sft_small.md) | 指令微调冒烟配置 `instruction_sft_small`（MPS、smoke_trim=24）一次运行记录与读数说明 |
| [TRAINING_LOG.md](TRAINING_LOG.md) | 训练配置切换与时间线记录 |

## 模块设计文档

| 文档 | 说明 |
|------|------|
| [m01_tokenizer.md](m01_tokenizer.md) | m01 分词器：tiktoken/gpt2/词表契约、特殊 token、理论落点与实现边界 |
| [bpe_principles.md](bpe_principles.md) | BPE 原理：算法、为何采用子词级、与 GPT-2/词表规模的关系 |
