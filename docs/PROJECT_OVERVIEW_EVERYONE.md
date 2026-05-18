# team-mini-llm 全员读得懂版说明

这份文档给所有组员看，不要求先懂深度学习、PyTorch 或 Transformer。  
目标是让大家能回答三个问题：

1. 这个项目到底做了什么？
2. 现在有哪些能力可以展示？
3. 如果老师或同学问起来，我们怎么解释？

---

## 1. 一句话说明

`team-mini-llm` 是我们小组从零实现的一个迷你 GPT 语言模型项目。

它不是调用 ChatGPT API，也不是直接改官方教程代码，而是我们自己把一条 LLM 训练链路拆出来实现：

```text
文本 → 分词 → 训练数据 → 注意力 → GPT 模型 → 预训练 → 生成 → 分类微调 → 指令微调
```

旁边的 `LLMs-from-scratch` 是教程仓库，只作为参考资料；真正的代码、实验、测试、报告都在 `team-mini-llm`。

---

## 2. 这个项目能做什么？

### 能力 A：把文本变成 token，再变回来

模型不能直接读中文或英文句子，它只能处理数字。

我们实现了 GPT-2 BPE 分词封装：

```text
"Hello world" → [15496, 995] → "Hello world"
```

这部分对应：

- 代码：`src/mini_llm/m01_tokenizer/`
- 文档：`docs/REQ-P1-01_Tokenizer.md`

可以这样解释：

> 分词器就像模型的字典，把人类文字翻译成模型能处理的编号。

---

### 能力 B：把长文本切成训练样本

语言模型训练时做的是“根据前文猜下一个 token”。

所以我们把一大段文本切成很多固定长度的小窗口：

```text
input : The history of London
target: history of London began
```

target 比 input 整体右移一位，模型就是靠这个学习“下一个词是什么”。

这部分对应：

- 代码：`src/mini_llm/m02_data_loader/`
- 文档：`docs/REQ-P1-02_DataLoader.md`

可以这样解释：

> DataLoader 像出题老师，把文章切成一张张“根据前面猜后面”的练习题。

---

### 能力 C：实现 GPT 的核心结构

我们实现了 GPT 里最核心的几块：

- 多头因果自注意力：每个 token 只能看见自己和前面的 token，不能偷看未来。
- TransformerBlock：注意力层 + 前馈网络 + 残差连接 + LayerNorm。
- GPTModel：输入 token 编号，输出每个位置上“下一个 token”的打分。

这部分对应：

- 注意力：`src/mini_llm/m03_attention/`
- GPT 模型：`src/mini_llm/m04_model/`
- 文档：`docs/REQ-P1-03_Attention.md`、`docs/REQ-P1-04_Model.md`

可以这样解释：

> GPT 的核心不是“背答案”，而是学会在一段上下文里判断哪些词更重要，然后预测下一个词。

---

### 能力 D：从零预训练一个 GPT-2 Small 级别模型

我们用英文 WikiText-103 数据，从零训练了一个 GPT-2 Small 规模的模型。

关键结果：

| 项目 | 结果 |
|------|------|
| 模型规模 | 约 163M 参数 |
| 训练数据 | WikiText-103 英文语料 |
| 训练步数 | 54,967 steps |
| 训练耗时 | 约 16 小时 39 分 |
| 最好验证损失 | `val_loss = 3.3092` |
| 产物 | `runs/gpt2_small_wikitext103/checkpoint_best.pt` |

这说明模型已经学到了一定的英文维基风格续写能力。

这部分对应：

- 训练脚本：`train.py`
- 配置：`configs/config_gpt2_small.json`
- 报告：`docs/RUN_REPORT_gpt2_small_wikitext103.md`

可以这样解释：

> 预训练就是让模型大量练习“猜下一个词”。loss 从高到低，说明它猜得越来越准。

---

### 能力 E：加载 checkpoint 做英文续写

训练完成后，我们可以加载保存好的模型权重，让它接着一段英文继续写。

示例：

```bash
uv run python generate_from_checkpoint.py \
  --checkpoint runs/gpt2_small_wikitext103/checkpoint_best.pt \
  --prompt "The history of"
```

注意：这个模型不是 ChatGPT，不是问答助手。  
它更像“英文维基百科续写器”。

适合的输入：

```text
The history of London began in
Science fiction is
The city of
```

不适合的输入：

```text
你好
你是谁？
法国首都是哪里？
```

因为它主要在英文 WikiText 上训练，中文和聊天问答都不是它的训练任务。

这部分对应：

- 生成模块：`src/mini_llm/m05_generate/`
- 推理脚本：`generate_from_checkpoint.py`
- 文档：`docs/REQ-P2-01_Generate.md`

可以这样解释：

> 它会续写，不一定会回答问题；这是预训练语言模型和聊天助手的区别。

---

### 能力 F：垃圾短信分类

我们还做了第 6 章风格的分类微调：让模型判断一条英文短信是正常短信 `ham` 还是垃圾短信 `spam`。

示例：

```bash
uv run python classify_sms.py --text "Thanks see you tomorrow"
```

输出可能是：

```text
ham
```

再试一条：

```bash
uv run python classify_sms.py --text "URGENT FREE prize call now"
```

输出目标是：

```text
spam
```

这部分对应：

- 训练脚本：`finetune_classify.py`
- 推理脚本：`classify_sms.py`
- 评估脚本：`eval_classify.py`
- 代码：`src/mini_llm/m06_classify_finetune/`
- 文档：`docs/REQ-P2-02_ClassifyFinetune.md`、`docs/REQ-P2-03_ClassifySmsInfer.md`

可以这样解释：

> 预训练模型学过英文文本规律；分类微调是在这个基础上再教它做“二选一判断”。

---

### 能力 G：指令微调，也就是让模型更像在“按要求回答”

预训练模型只会续写，不会天然听指令。

所以我们做了第 7 章风格的指令微调 SFT。训练数据长这样：

```text
Below is an instruction that describes a task.

### Instruction:
Translate the following to Spanish: Good morning.

### Response:
Buenos días.
```

训练目标仍然是“猜下一个 token”，但文本格式变成了“任务说明 + 标准回答”。

这部分对应：

- 数据模块：`src/mini_llm/m07_instruction_finetune/`
- 训练脚本：`finetune_instruction.py`
- 正式 Small 配置：`configs/config_instruction_train_small.json`
- 仅评估脚本：`eval_instruction_loss.py`
- 对照生成脚本：`compare_instruction_generate.py`
- 文档：`docs/REQ-P3-01_Ch07InstructionSFT.md`、`docs/REQ-P3-02_InstructionSFTEvalAndQuality.md`

本轮决策：

- 指令微调只以 GPT-2 Small checkpoint 为底座。
- Medium checkpoint 不作为本轮指令微调底座。
- 自动评分、批量导出 JSON 等属于后续 backlog。

可以这样解释：

> SFT 像给模型看很多“题目 + 标准答案”的作业纸，让它学会看到类似格式时往回答方向续写。

---

## 3. 当前项目完成度

| 阶段 | 内容 | 状态 |
|------|------|------|
| P1 | 分词、数据加载、注意力、GPT 模型、预训练 | 已完成 |
| P2 | 文本生成、SMS 分类微调、单条短信推理 | 已完成 |
| P3-01 | 指令微调数据管线与训练脚本 | 已完成 |
| P3-02 | 指令微调质检：全 val、对照生成、仅评估入口 | 已完成 |
| P1-07 | GPT-2 Medium + 更大规模预训练实验 | 独立实验，非 P3 底座 |

一句话：

> 主线能力已经闭环：能训练、能保存 checkpoint、能生成、能分类、能做指令 SFT，并且有测试和报告支撑。

---

## 4. 怎么证明不是只写了代码？

项目里有三类证据。

### 证据 1：自动化测试

全量测试命令：

```bash
uv run pytest -q
```

测试覆盖：

- tokenizer
- data loader
- attention
- model forward
- generate
- classify finetune
- classify metrics
- instruction finetune

### 证据 2：训练产物

重要 checkpoint：

| 路径 | 说明 |
|------|------|
| `runs/gpt2_small_wikitext103/checkpoint_best.pt` | GPT-2 Small 预训练最好权重 |
| `runs/spam_classify_phase_b/checkpoint_best.pt` | SMS 分类推荐演示权重 |
| `runs/instruction_sft_small/checkpoint_best.pt` | 指令微调冒烟权重 |

### 证据 3：运行报告与验收文档

| 文档 | 看什么 |
|------|--------|
| `docs/RUN_REPORT_gpt2_small_wikitext103.md` | 预训练跑了多久、loss 降到多少、生成效果如何 |
| `docs/RUN_REPORT_instruction_sft_small.md` | 指令微调冒烟训练日志怎么读 |
| `docs/OWNER_CHECKLIST.md` | 项目负责人怎么一步步验收 |
| `SPEC.md` | 每个模块做到哪了、测试覆盖如何 |
| `HARNESS.md` | 每个阶段用什么命令证明通过 |

---

## 5. 如果要演示，推荐顺序

### 演示 1：跑测试，证明工程没坏

```bash
uv run pytest -q
```

### 演示 2：预训练模型英文续写

```bash
uv run python generate_from_checkpoint.py \
  --checkpoint runs/gpt2_small_wikitext103/checkpoint_best.pt \
  --prompt "The history of"
```

### 演示 3：垃圾短信分类

```bash
uv run python classify_sms.py --text "Thanks see you tomorrow"
uv run python classify_sms.py --text "URGENT FREE prize call now"
```

### 演示 4：指令微调 val loss 复评

```bash
uv run python eval_instruction_loss.py \
  --config configs/config_instruction_small.json \
  --checkpoint runs/instruction_sft_small/checkpoint_best.pt
```

预期能看到类似：

```text
val_loss_sampled=...
val_loss_full=...
```

### 演示 5：预训练 vs 指令微调对照生成

```bash
uv run python compare_instruction_generate.py \
  --pretrained runs/gpt2_small_wikitext103/checkpoint_best.pt \
  --sft runs/instruction_sft_small/checkpoint_best.pt \
  --prompt-file docs/prompts/instruction_compare_sample.txt
```

---

## 6. 这个项目不是什么？

为了避免误解，下面这些不是本项目当前目标：

| 误解 | 正确说法 |
|------|----------|
| 这是 ChatGPT | 不是。它是教学规模的 GPT 训练项目。 |
| 它能稳定中文聊天 | 不能。主要训练语料是英文 WikiText。 |
| 它能回答所有事实问题 | 不能。预训练模型主要是续写，不是检索问答系统。 |
| 它用了官方 GPT-2 权重 | 不是主线。主线是我们自己从零训练的 checkpoint。 |
| Medium 是指令微调底座 | 不是。本轮 P3 只用 Small 底座收口。 |
| 它已经做了 DPO / RLHF | 没有。偏好学习是 backlog。 |

---

## 7. 每个组员可以怎么讲自己的部分？

| 分工 | 可以这样讲 |
|------|------------|
| 分词 / 数据 | 我负责把原始文本变成 token，并切成模型训练用的 input/target。 |
| 注意力 | 我负责实现因果自注意力，让模型只能看前文，不能偷看未来。 |
| GPT 模型 | 我负责把 embedding、attention、feed-forward、layer norm 组装成完整 GPT。 |
| 训练 | 我负责 loss、优化器、学习率调度、checkpoint、训练日志。 |
| 生成 | 我负责从 checkpoint 加载模型，并用 greedy / temperature / top-k 生成文本。 |
| 分类微调 | 我负责把预训练模型改造成短信 ham/spam 分类器。 |
| 指令微调 | 我负责把 instruction JSON 变成训练样本，并做 SFT 与质检。 |
| 整合 / 汇报 | 我负责把模块、测试、报告、OpenSpec、PPT 串起来，形成可验收项目。 |

---

## 8. 最短汇报稿

如果只能讲一分钟，可以这样说：

> 我们做的是一个从零实现的迷你 GPT 项目，不是直接调用现成大模型。项目从 GPT-2 BPE 分词开始，依次实现了数据加载、因果多头注意力、GPT 模型、预训练循环、checkpoint 保存和加载生成。我们用 WikiText-103 从零训练了一个约 163M 参数的 GPT-2 Small 模型，best val loss 达到 3.3092。基于这个 checkpoint，我们还完成了两个下游任务：SMS 垃圾短信分类，以及第 7 章风格的指令微调 SFT。现在项目能演示英文续写、ham/spam 分类、指令微调复评和预训练/SFT 对照生成。所有核心模块都有 pytest 测试，项目用 SPEC、HARNESS、REQ、OWNER_CHECKLIST 和运行报告记录验收口径。

---

## 9. 想继续深入看哪里？

| 想了解 | 推荐读 |
|--------|--------|
| 项目整体技术 README | `README.md` |
| 每个模块做到哪 | `SPEC.md` |
| 怎么验收 | `HARNESS.md`、`docs/OWNER_CHECKLIST.md` |
| 预训练结果 | `docs/RUN_REPORT_gpt2_small_wikitext103.md` |
| 指令微调结果 | `docs/RUN_REPORT_instruction_sft_small.md` |
| 所有文档索引 | `docs/README.md` |
