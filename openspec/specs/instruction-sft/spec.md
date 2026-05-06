# 指令监督微调（SFT）

## 目的

从已有 **自回归 GPT** 预训练权重出发，在 **书本格式指令数据**（`instruction` / 可选 `input` / `output`）上继续训练，使模型在「带 `### Instruction:` / `### Response:` 模板」的英文文本上预测下一 token；**不得**在填充（pad）位置上对交叉熵损失计分。  
本规格描述 **P3-01（轨道 A）** 已具备的行为；**评测与监控增强（P3-02）** 见 [`docs/REQ-P3-02_InstructionSFTEvalAndQuality.md`](../../../docs/REQ-P3-02_InstructionSFTEvalAndQuality.md)，该项关闭后应将对应 **必须** 条款并入本文件。

## 非目标

- **不得**将本能力等同于「可靠事实问答」或与 Chat 类产品对齐；数据量很少、训练很短时，生成质量**不在**本规格保证范围内。  
- **可以**后续通过 [REQ-P3-01 §9](../../../docs/REQ-P3-01_Ch07InstructionSFT.md) 所列 backlog 扩展 DPO 等；**当前规格不包含**偏好学习。

## 参阅文档

| 文档 | 用途 |
|------|------|
| [REQ-P3-01](../../../docs/REQ-P3-01_Ch07InstructionSFT.md) | 业务边界、双轨（Small/Medium）、验收草案 |
| [REQ-P3-01SUB](../../../docs/REQ-P3-01SUB_Ch07InstructionBookAlignment.md) | 与书本 `gpt_instruction_finetuning.py` 对齐细则 |
| [HARNESS.md](../../../HARNESS.md) Part III | 命令级 Harness 与通过判据 |
| [SPEC.md](../../../SPEC.md) · P3-01 | 公开 API、配置字段、测试文件路径 |

---

## 需求

### 需求：指令字符串模板

系统 **应当**将每条训练样本拼成单一 UTF-8 字符串，包含固定引导语、`### Instruction:` 与任务正文、可选的 `### Input:`、以及 `### Response:` 与参考答案全文，与 REQ-P3-01SUB 所述书本模板一致。

#### 场景：省略可选 input

- **给定** JSON 条目中 `input` 为空或缺失  
- **当**为编码构建模板字符串时  
- **那么**字符串 **不得** 多出一段无依据的非空 `Input` 区（格式与 `m07_instruction_finetune` 参考实现一致）。

---

### 需求：填充位不计入损失

系统 **必须**在微调交叉熵中使用 `ignore_index`（约定 `-100`）排除填充位置，使模型不被训练去「预测填充 token」。

#### 场景：变长样本成批

- **给定**一批长度不等的已编码样本，经 collate 得到固定形状的 `inputs` 与 `targets`  
- **当**计算损失时  
- **那么**处于填充位置的 target **应当**被交叉熵归约忽略（等价地：这些位置无梯度）。

---

### 需求：微调入口与 checkpoint

系统 **应当**提供入口程序：加载与 [`GPTModel`](../../../SPEC.md) 兼容的预训练 checkpoint，执行指令 SFT，并在配置的 run 目录下持久化 **`checkpoint_best.pt`**（或等价的「验证最优」路径）。

#### 场景：冒烟配置跑通

- **给定**合法的 Small 预训练 checkpoint（如 WikiText-103 的 `checkpoint_best.pt`）与 [`configs/config_instruction_small.json`](../../../configs/config_instruction_small.json)  
- **当**以该配置执行指令微调脚本时  
- **那么**正常情形下流程 **应当**结束且损失中不出现 NaN  
- **且** **应当**按配置将最优 checkpoint 写入 run 输出目录（如 `runs/<run_name>/checkpoint_best.pt`）。

#### 场景：checkpoint 携带指令元数据

- **给定**一次成功的指令 SFT  
- **当**写入最优 checkpoint 时  
- **那么**文件 **应当**包含 `instruction_meta`（或等价字段），记录模板标识、pad token id、`ignore_index`、长度策略等，以便推理或后续训练与当时假设对齐。

---

### 需求：数据通路的自动化测试

系统 **必须**提供自动化测试，校验指令数据划分、下载/缓存（本地路径）及与书本参考一致的 collate/掩码行为。

#### 场景：指令微调测试模块通过

- **给定**项目开发环境（如适用则 `uv sync --extra dev`）  
- **当**执行 `pytest tests/test_instruction_finetune.py` 时  
- **那么**全部用例 **应当**通过。

---

## 路线图（REQ-P3-02 归并目标）

下列项在 **P3-01 关闭时仍非强制**；在 [REQ-P3-02](../../../docs/REQ-P3-02_InstructionSFTEvalAndQuality.md) 完成且 HARNESS 更新后 **必须**满足：

- 可配置的 **整体验证集** loss（或在日志中明确 `eval_iter` 抽样语义）。  
- **按 epoch 结束**（或统一）的最优 checkpoint 策略，相对「仅按步评估」。  
- 成文规定的 **成对生成** 流程（相同提示语、预训练 vs SFT 权重、固定解码超参）。

在此之前，过渡期的负责人自检见 REQ-P3-02 与 [`docs/OWNER_CHECKLIST.md`](../../../docs/OWNER_CHECKLIST.md) Part III。
