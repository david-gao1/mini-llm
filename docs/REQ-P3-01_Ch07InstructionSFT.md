# REQ-P3-01：第 7 章——教会模型「听指令、写回答」（指令微调 SFT）

**所属**：[SPEC.md](../SPEC.md) → Part III · 指令微调（第 7 章）  
**依赖**：[REQ-P1-04](REQ-P1-04_Model.md)（`GPTModel`）、[REQ-P1-05](REQ-P1-05_Train.md)（Small 预训练 checkpoint）  
**被依赖**：[REQ-P3-02](REQ-P3-02_InstructionSFTEvalAndQuality.md)（效果检验、训练监控与质量优化）  
**状态**：✅ **已实现**（`m07` + `finetune_instruction.py` + 单测；本轮以 Small checkpoint 为底座收口）；DPO → §9 backlog  
**原书对照**：和本仓库 **同级**的 [`../../LLMs-from-scratch/ch07/`](../../LLMs-from-scratch/ch07/) · [`gpt_instruction_finetuning.py`](../../LLMs-from-scratch/ch07/01_main-chapter-code/gpt_instruction_finetuning.py) · [`REFERENCE.md`](../REFERENCE.md)  
**OpenSpec（行为契约）**：[指令 SFT · `instruction-sft/spec.md`](../openspec/specs/instruction-sft/spec.md)

---

## 1. 业务逻辑（读完就知道「要干嘛」）

### 先打个比方

想象你在培训一位只会「接龙写文章」的员工（这就是 **预训练语言模型**）：  
第七章要做的不是考他「这条短信是不是垃圾」（那是 **第六章分类**），而是给他一张 **作业纸**，上面写着：**任务说明 +（可选）补充材料 + 标准答案**。  
多抄几千遍类似的作业，他就会逐渐学会：**看到任务说明 → 按格式写出一段像样的回答**。

这就是 **指令微调（书里叫 instruction finetuning；行业里常叫 SFT，Supervised Fine-Tuning）**：  
仍然是「猜下一个词」，但练习材料换成了「指令 + 回答」拼好的整段话。

### 最关键的一句话

> **每条训练样本** = 一小段固定格式的英文（任务说明 + 可选输入 + 参考答案连在一起）；**训练方式**还是「猜下一个词」；**填空对齐的地方**（补齐长短不一的句子时用的 pad）在算 loss 时要 **跳过**，别让模型去学「无意义的填充符号」。  
> **本轮只做这一种监督学习（SFT）**；**不搞「偏好对战 / DPO」**——那种放到下面 **§9**，以后再议。

### 和第六章有什么不一样（别混）

| | 第六章（短信分类） | 第七章（本条 REQ） |
|--|-------------------|-------------------|
| **输入** | 一条短信 | 一段「指令 + 可选材料」 |
| **输出** | 两个标签之一：`ham` / `spam` | **一小段文字回答**（很多个 token） |
| **模型头上** | 换成 **2 类** 小脑袋 | **不换**成二分类，还是用 **整本词典那么大** 的输出头，一个一个词往外猜 |

### 你们已经拍板的三件事（白话版）

| 约定 | 白话 |
|------|------|
| **跟书走** | 数据模板、划分、collate、loss 等与 **`gpt_instruction_finetuning.py`** **`main()`** 对齐；**细则（四段流水线 + 刻意差异）** 见子文档 [**REQ-P3-01SUB**](REQ-P3-01SUB_Ch07InstructionBookAlignment.md)。实现上使用 **`GPTModel` + `encode_text`**，不复制书里 `previous_chapters`。 |
| **只做 SFT** | 只有「标准答案抄写作业」这一种训练；**不做** DPO、奖励模型、强化学习那一套。 |
| **一条正式跑道** | 用 GPT-2 **Small** 预训练 checkpoint 做底座：先用 `config_instruction_small.json` 冒烟，再用 `config_instruction_train_small.json` 做全划分、多 epoch、全 val 质检。**Medium checkpoint 不作为本轮 SFT 底座**；P1-07 仍是独立预训练实验。 |

冒烟与正式 Small 用的是 **同一套代码和数据管线**，只是 **换数据多少**、**换训练多久**、**换评估口径**。  
**书上每一步 ↔ 仓库符号**：见 [**REQ-P3-01SUB · 「跟书走」对齐细则**](REQ-P3-01SUB_Ch07InstructionBookAlignment.md)。

---

## 2. 设计思路（技术上怎么做——仍可跳着读）

### 书上已经替你验证过的流程（我们照着对齐）

1. **数据文件**：很多条 JSON，每条大致有——要做什么（`instruction`）、可选的补充（`input`，可以空）、参考答案（`output`）。默认用书里的 [`instruction-data.json`](https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json)（也可下载后放到本地缓存，少爬网）。
2. **拼成一整段字**：先写「下面是一条指令……」+ `### Instruction:` ……，若有输入再加 `### Input:` ……，最后加上 `### Response:` 和参考答案。模型练习的就是 **整段连起来的下一个词**。
3. **一批里句子长短不一**：短的要在末尾 **垫（pad）** 到跟最长的一样长，才能叠成矩阵。**垫出来的位置**在「标准答案那一栏」里标记成一个 **不算分的记号**（书里常用 `-100`，叫 `ignore_index`），这样 loss 不会去强迫模型「预测填充符」。
4. **训练**：和普通预训练一样用 **交叉熵** 做下一个词预测；**没有**第六章那种「只有两个类别」的分类头。

### 我们仓库里预期会长什么样（名字可以微调）

- **一个新模块**（例如 `mini_llm.m07_instruction_finetune`）：负责读 JSON、拼字符串、Dataset、以及「把一批样本垫齐并打上不算分记号」的函数。
- **一个新脚本**（例如根目录 `finetune_instruction.py`）：加载 **已有的预训练** `.pt` → 跑训练循环 → 再存一个新的 `.pt`，供后面接 **生成脚本** 试用。
- **两份 Small 配置**：`configs/config_instruction_small.json` 用于冒烟；`configs/config_instruction_train_small.json` 用于全划分、多 epoch 与全 val 质检。里面写 **预训练权重路径**、数据路径或网址、学习率、batch、最长长度等。

---

## 3. 数据怎么流（一图流）

```text
instruction-data.json（很多条「指令+回答」）
        │
        ▼
   拼成整段英文 → 转成 token 编号
        │
        ▼
   按 batch 垫齐 ──► 标记「填充位置别算分」
        │
        ▼
   接上已有的 GPTModel（词表那么大的输出头）
        │
        ▼
   finetune_instruction.py 训练 → runs/…/checkpoint_*.pt
```

---

## 4. 契约要点（实现时要遵守的大方向）

- **Dataset**：取出一条，就应能拿到 **这一条对应的整段 token 编号**（含 Response 部分）。
- **组 batch**：交给模型的要有 **`inputs` 和 `targets`** 两个矩阵；**targets** 里凡是不想让它学的地方（主要是 pad），一律填 **不算分的那个数**。
- **存盘**：checkpoint 里除了权重，最好带上 **这次训练用的最长长度**、模板版本之类信息，免得以后加载推理时对不齐。

（字段级明细表等代码写好后再补，写法可对齐 `finetune_classify.py`。）

---

## 5. 怎样算验收通过（草案）

| 步骤 | 你要跑什么（暂定） | 怎样算过 |
|------|-------------------|----------|
| 单元测试 | `pytest tests/test_instruction_finetune.py` | 垫齐、mask、形状 **和书上逻辑一致**；固定随机种子可复现 |
| 真训练（小配置） | `uv run python finetune_instruction.py --config configs/config_instruction_small.json` | loss 是正常数字、不出现 NaN；**磁盘上真有新 checkpoint** |
| 肉眼看一下 | 用微调后的权重 + 现有生成方式，试 **2～3 条固定指令** | 听起来更像 **在回答问题**，而不是维基百科那种「接着往后编百科」（记在 REPORT 或笔记里即可） |

具体命令以后写进 [`HARNESS.md`](../HARNESS.md)、[`SPEC.md`](../SPEC.md)。

---

## 6. 谁先谁后（依赖）

- 只要你有 **Small** 的预训练 checkpoint（例如 WikiText-103 那条）就能开始并完成本轮验收。
- **Medium** 预训练（P1-07）不作为本轮 SFT 底座，也不阻塞 P3-01 / P3-02。

---

## 7. 书里同一章还有别的文件夹——本条先不做

这些是 **选修**，不算本条 REQ 交付：

| 原书文件夹 | 大概是啥 |
|------------|----------|
| `02_dataset-utilities/` | 数据去重、清洗之类 |
| `04_preference-tuning-with-dpo/` | **偏好学习 / DPO** → 放到 **§9** |
| `06_user_interface/` | 小网页演示 |

---

## 8. 文档索引（做完代码后要回来改的）

[`SPEC.md`](../SPEC.md)、[`HARNESS.md`](../HARNESS.md)、[`docs/README.md`](README.md)、[`REFERENCE.md`](../REFERENCE.md)、[`README.md`](../README.md)。自学问题可写在 [`LEARNING_LOG.md`](LEARNING_LOG.md)。  
行为契约见文首 **OpenSpec（行为契约）**。  
**质检与优化闭环**（全 val、对照生成、checkpoint 策略等）：[**REQ-P3-02**](REQ-P3-02_InstructionSFTEvalAndQuality.md)。

---

## 9. 以后再做的：偏好 / DPO（Backlog）

> **再打个比方**：**SFT** = 老师给你标准答案，你照着背写法。**DPO** = 给你两篇作文，说「这篇比那篇好」，你学会更喜欢哪一种风格。**本条 REQ 不做后半种**，免得验收界限模糊。

| 编号 | 内容 | 书上可参考 |
|------|------|------------|
| **BL-P3-01-01** | 偏好数据 + DPO 训练 | `ch07/04_preference-tuning-with-dpo/` |
| **BL-P3-01-02** | 自动或半自动评测脚本 | `01_main-chapter-code/ollama_evaluate.py` 等 |

---

## 10. 修订记录

| 日期 | 变更 |
|------|------|
| 2026-05-17 | 决策更新：Medium checkpoint 不作为本轮 SFT 底座；P3 以 Small-only 跑道收口。 |
| 2026-04-30 | 初稿：参考书数据/脚本、仅 SFT、Small/Medium 双轨；DPO → §9。 |
| 2026-04-30 | 全文改写成更易读的表述（比方、表格白话、术语后置）。 |
| 2026-05-05 | §8：链至 [**REQ-P3-02**](REQ-P3-02_InstructionSFTEvalAndQuality.md)；**被依赖**更新为 P3-02。 |
| 2026-05-06 | 头部增加 **OpenSpec** 链至 [`openspec/specs/instruction-sft/spec.md`](../openspec/specs/instruction-sft/spec.md)。 |
