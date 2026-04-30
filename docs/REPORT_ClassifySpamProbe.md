# REPORT：SMS Spam 分类 — 探针误判与高置信 FN 分析

**性质**：运行分析报告（非 REQ 验收文档）  
**日期上下文**：2026-04（与仓库当期训练/推理链一致）  
**关联需求**：[REQ-P2-02_ClassifyFinetune.md](REQ-P2-02_ClassifyFinetune.md) · [REQ-P2-03_ClassifySmsInfer.md](REQ-P2-03_ClassifySmsInfer.md)  
**相关 backlog**：REQ-P2-02 [§10 可选增强](REQ-P2-02_ClassifyFinetune.md)（BL-P2-02-02 / -04 等）

---

## 1. 背景与目标

项目在 GPT-2 Small 兼容架构（本仓库 `GPTModel` + WikiText-103 预训练 checkpoint）上，按 REQ-P2-02 完成 **SMS 垃圾短信二分类微调**：标签 `ham=0`、`spam=1`。训练可产出 **`runs/spam_classify/`**（[`config_classify_spam.json`](../configs/config_classify_spam.json)，解冻末 1 block）与 **`runs/spam_classify_phase_b/`**（[`config_classify_spam_phase_b.json`](../configs/config_classify_spam_phase_b.json)，解冻末 2 block）。**文档与脚本默认演示路径**为 **`runs/spam_classify_phase_b/checkpoint_best.pt`**（对照结论见 §7.3「对照实验说明了什么」）；单条推理见 REQ-P2-03 [`classify_sms.py`](../classify_sms.py)。

数据管线来自 **UCI SMS Spam Collection**：原始极度不平衡（ham 远多于 spam），实现上对 ham **随机下采样**至与 spam 等量后再 **70% / 10% / 20%** 划分 train / val / test，`random_state=123` 固定（见 `mini_llm.m06_classify_finetune.download_and_prepare_spam`）。

---

## 2. 方法与实现要点（与误判分析相关）

| 要点 | 说明 |
|------|------|
| 表征与分类头 | 冻结绝大部分 Transformer；替换语言建模头为 `Linear(emb_dim → 2)`；默认 **仅解冻最后 1 个 block + final_norm**（[`configs/config_classify_spam.json`](../configs/config_classify_spam.json) · `unfreeze_last_n_blocks: 1`） |
| 序列编码 | GPT-2 BPE（tiktoken），过长截断，**右侧 pad**，`pad_token_id=50256` |
| 分类 logits | `model(batch)[:, -1, :]`，即序列 **最后一个位置** 的 logits（与书本第六章一致） |
| 损失 | 对最后一个位置的 logits 做 `cross_entropy` |
| checkpoint | 按 **验证集 accuracy** 保存 `checkpoint_best.pt`；训练结束报告 **test accuracy** |

配置默认值：`num_epochs=5`，`batch_size=8`，AdamW `lr=5e-5`，`weight_decay=0.1`。

---

## 3. 已观察到的运行结果

| 项目 | 结果 |
|------|------|
| 训练 | 成功；规模量级约 train ≈1045 / val ≈149 / test ≈300（与平衡后划分一致） |
| 聚合指标 | Test accuracy 约 **96%**；best 验证集 accuracy 约 **95%**（依 checkpoint 保存逻辑） |
| 推理命令示例 | （历史记录）基线权重：`classify_sms.py --checkpoint runs/spam_classify/checkpoint_best.pt`；**当前默认演示**见 §8（`spam_classify_phase_b`） |
| 预测 | **ham** |
| 概率 | **P(ham)=0.8341，P(spam)=0.1659**（`--probs` 详见 REQ-P2-03：概率走 stderr，stdout 单行标签） |

---

## 4. 问题陈述（从指标到单句）

- **聚合层面**：高 test accuracy 表示在 **固定测试集** 上大部分样本预测正确，**不保证**任意人工构造或域外英文短语都正确。
- **本句层面**：该文案为典型「中奖/领奖」垃圾话术模板；人类标注应倾向 **spam**。模型却以 **约 83% 置信度** 判为 **ham**，属于 **高置信错误（confident wrong）**，不可归因于决策边界上的偶然抖动。
- **业务含义**：属于 **漏判 spam（False Negative）** 风险。REQ-P2-02 §10 已注明：上线场景往往更敏感于 FN。

---

## 5. 可能原因分析（待数据验证）

1. **训练分布 vs 探针短语**：UCI 中 spam 多为真实采集短信，与教科书式英文模板 **分布不一致**；模型学到的是数据集中的边界，未必覆盖未见模板。
2. **仅靠 accuracy**：平衡集上 accuracy 与 spam 质量有关，但 **不等价于 spam recall**；少见措辞可能欠拟合。
3. **微调容量**：仅解冻末 1 block，可调空间有限；未见措辞泛化可能不足。
4. **Pooling 约定**：`[:, -1, :]` 与右填充一致，训练/推理自洽；仍存在与其他 pooling（如最后非 pad、masked mean）对照实验的空间（架构级，非当前 REQ 必做）。
5. **checkpoint 选择**：best 由 **val accuracy** 决定；val 集较小可能使某些错误模式在选模阶段未被充分反映——需 **混淆矩阵 / per-class F1** 佐证。

---

## 6. 结论

- **流水线正确性**：训练与推理、标签映射、`--probs` 行为与源码一致；该误判来自 **模型在该输入下的决策**，而非明显的 CLI 误用。
- **评估口径**：BL-P2-02-02 已在代码中落地（混淆矩阵、spam/ham PRF、FN CSV、`eval_classify.py`）；上线决策仍建议结合探针与业务阈值，不单看 accuracy。

---

## 7. 改进计划

### 7.1 原则

先 **量化定位**（是否系统性漏判 spam），再 **小步调超参与数据**，最后考虑 **架构/预训练对照**；每步有可重复命令与指标闸门。

### 7.2 阶段 A：观测与指标（优先）

| 序号 | 动作 | 目的 |
|------|------|------|
| A1 | test（或 val+test）输出 **混淆矩阵** TN/FP/FN/TP | 看清 FN/FP |
| A2 | 报告 **spam** 的 precision / recall / F1（ham 可选） | 对齐上线关注点 |
| A3 | 导出 **FN 样本**（预测 ham、真实 spam）抽样人工浏览 | 归纳错误形态 |
| A4 | 固定 **回归探针句**（含本条 WINNER、若干 UCI 原句） | 防回归 |

**落地**：实现 **BL-P2-02-02**（`finetune_classify.py` 末尾或独立 `eval_classify.py`）。

### 7.3 阶段 B：训练与数据试错

#### 对照实验的业务逻辑（写给 Java 背景的 Owner）

**我们要解决的业务问题**：短信分类器在实际使用里最「疼」的是 **漏判垃圾（FN）**——真 spam 被判成 ham，等于放行了一条骚扰或诈骗话术。此前出现过「整体验证指标很好看，但某条典型 spam 话术仍高置信判 ham」的情况（见本文 §3），所以需要一种 **成本低、可重复** 的手段探索「能不能在不推倒重写的前提下，再多挤出一点 spam 捕获能力」。

**对照实验在做什么（用人话说）**：  
用 **同一套训练脚本、同一批数据、同一种保存 best 的规则**，只改 **「有多少层神经网络允许在本次任务里继续学习」**——跑两轮训练，得到 **两个 checkpoint**，再在 **同一张测试卷（test.csv）** 上打分，对比 **漏判数量 FN** 和 **spam 召回 Recall**。这就像：

- **同一套业务代码库**，打两个 **不同 JVM 启动参数或 Spring Profile** 的包（这里是「解冻层数」不同）；  
- **同一套集成测试套件**（这里是固定的 test 划分）；  
- **产物是两个可部署的 artifact**：`runs/spam_classify/` 与 `runs/spam_classify_phase_b/`，**互不覆盖**，便于并排 diff。

**两条分支分别代表什么**：

| 分支 | 配置 | 业务含义（直觉） |
|------|------|------------------|
| **基线** | `config_classify_spam.json`，解冻末 **1** 个 Transformer block | 「尽量少动预训练底座」，训练快、不易过拟合，但对难样本可能欠表达。 |
| **阶段 B** | `config_classify_spam_phase_b.json`，解冻末 **2** 个 block | 「允许多一层适配下游」，表达能力略升，可能更好抓某些 spam 模板，也可能在小数据上更容易 **过拟合**（验证集好、测试集未必好）。 |

**不算「上线决策」，算「实验记录」**：  
本对照 **不要求** phase_b 一定优于基线；**要求**的是：跑完后把两边的 **混淆矩阵 / spam 的 P/R/F1 / FN 条数**记下来（或贴进运行笔记），便于半年后回看「当时我们尝试过加大可训练深度，结论是 A 还是 B」。若 phase_b 明显变差，回滚到基线 checkpoint 即可——没有数据库迁移、没有接口破坏性变更，只是换一个 `.pt` 文件。

**你可能关心的工程差异**：解冻层变多 → **每次反向传播要更新的参数更多**，单 step 可能略慢、显存/内存略涨；epoch 数不变时，总耗时会上去一点——类比「多打开了几层模块的 hot reload / 更重的 bean 图」，仍在同一进程同一入口脚本内完成。

---

**对照配置（已落盘）**：[`configs/config_classify_spam_phase_b.json`](../configs/config_classify_spam_phase_b.json) — `unfreeze_last_n_blocks: 2`，`run_name: spam_classify_phase_b`（写出到 `runs/spam_classify_phase_b/`，不覆盖默认 `spam_classify`）。

```bash
# 基线（末 1 块解冻）
uv run python finetune_classify.py --config configs/config_classify_spam.json

# 阶段 B（末 2 块解冻）
uv run python finetune_classify.py --config configs/config_classify_spam_phase_b.json

# 分别评估（对比 spam R / FN）
uv run python eval_classify.py --checkpoint runs/spam_classify/checkpoint_best.pt
uv run python eval_classify.py --checkpoint runs/spam_classify_phase_b/checkpoint_best.pt
```

| 序号 | 动作 |
|------|------|
| B1 | `unfreeze_last_n_blocks: 2`（见上 phase_b 配置）或谨慎增加 epoch |
| B2 | 学习率小幅网格（如 `3e-5`～`8e-5`），仍可复制 phase_b 改 `learning_rate` + 换新 `run_name` |
| B3 | 规则级数据增强（spam，需谨慎） |
| B4 | **BL-P2-02-04**：全量 ham + 类权重 CE，与下采样对照 |

#### 对照实验运行记录（一次实际跑数，便于复盘）

**环境**：同一 `data_cache/sms_spam` 划分、`seed=123`；预训练 `runs/gpt2_small_wikitext103/checkpoint_best.pt`；设备 MPS；训练约 **55s**（phase_b）。

| 指标 | 基线 `runs/spam_classify/checkpoint_best.pt` | 阶段 B `runs/spam_classify_phase_b/checkpoint_best.pt` |
|------|-----------------------------------------------|--------------------------------------------------------|
| 可训练参数占比（日志打印） | （本次未重训基线；历史约 5–6%） | **11.4%**（解冻末 2 block） |
| Test accuracy | **93.33%** | **96.33%** |
| 混淆矩阵 TN / FP / FN / TP | 144 / 5 / **15** / 136 | 147 / 2 / **9** / 142 |
| spam Recall（R） | 0.9007 | **0.9404** |
| spam F1 | 0.9315 | **0.9627** |
| FN CSV 条数（test 漏判 spam） | **15** | **9** |
| 探针：`WINNER!! You have been selected...` | （此前 **基线** 报告：ham，P(ham)≈0.83） | **spam**，`P(spam)=0.6300` |

#### 对照实验说明了什么（结论归档）

1. **假设得到支持**：漏判与「微调深度不够」一致——仅把可训练深度从末 **1** 个 block 加到 **2** 个，在同一数据与随机种子下，**test 上 FN 减少、spam Recall/F1 上升**，说明模型有更多容量把决策边界推向「少漏 spam」一侧。
2. **整体指标与探针同向**：不仅表格数字变好，原先 **高置信 ham** 的模板句在 phase_b 下改为 **spam**（置信中等），与「减少 FN」的业务目标一致。
3. **不是普遍定理**：只在 **当前划分、当前预训练 checkpoint、当前 lr/epoch** 下成立；换数据或解冻更多层可能 **过拟合** 或变差，故仍需每次记录 `eval_classify` 输出。
4. **工程类比（Java）**：同一应用两套 **profile**，一套少放开可写模块、一套多放开一层；用 **同一套集成测试（test.csv）** 比失败用例数；phase_b 这轮 **失败用例更少**，故仓库把 **默认 artifact** 指到 phase_b，避免后人演示仍拿旧 profile。
5. **不能推出的结论**：不等于上线安全；不等于英文域外话术全覆盖；**FP（误杀 ham）** 仍需业务权衡。

**文档与脚本默认**：[`classify_sms.py`](../classify_sms.py)、[`eval_classify.py`](../eval_classify.py) 及本文 §8 示例均以 **`runs/spam_classify_phase_b/checkpoint_best.pt`** 为默认路径；基线路径 **`runs/spam_classify/...`** 保留用于对照复现。**注意**：磁盘上基线 `.pt` 若来自不同训练时刻，`eval_classify` 数字可能与早期对话记录不完全一致；并排对比以当场打印为准。

### 7.4 阶段 C：工程体验（可选）

- `classify_sms.py`：`--json` 单行输出等，稳定「标签 + 概率」顺序（REQ-P2-03 backlog 已提及）。
- 文档强调：**聚合指标 ≠ 单句万能**。

### 7.5 阶段 D：架构与预训练（中长期）

- 最后非 pad token / masked mean pooling 与 `[:, -1, :]` 对比。
- **BL-P2-02-03**：换「预训练底座」（OpenAI 官方 GPT-2 Small vs 本仓库 WikiText 自训），其余 SMS 微调流程不变，并排看 test 指标；需权重对齐代码，见 [REQ-P2-02 §10.1](REQ-P2-02_ClassifyFinetune.md)。

### 7.6 阶段 A 落地状态（已实现）

- **`classify_sms.py` / `eval_classify.py`**：**默认 `--checkpoint`** = **`runs/spam_classify_phase_b/checkpoint_best.pt`**（对照结论见 §7.3「对照实验说明了什么」）；基线路径 **`runs/spam_classify/...`** 仍可用于对照。`eval_classify` 另将 FN 默认写入 **checkpoint 同目录**下的 `eval_false_negative_spam.csv`。
- **`mini_llm.m06_classify_finetune`**：`collect_predictions_loader`、`confusion_counts_binary_spam`、`prf1_from_counts`、`export_false_negative_spam_csv`、`format_classification_eval_lines`。
- **`finetune_classify.py`**：训练结束后加载 **best val** 权重（而非最后一 epoch），在 **test** 上打印混淆矩阵与 spam/ham 的 P/R/F1，并将 FN（真实 spam、预测 ham）导出至 ``runs/<run_name>/test_false_negative_spam.csv``。
- **探针清单**：[`docs/probes/classify_spam_probes.json`](probes/classify_spam_probes.json)，供人工或脚本对照 `classify_sms`。

### 7.7 建议节奏（后续）

- **阶段 B**：B1–B2 + 探针记录的定量对照。
- **有空**：B3/B4、C、D 按优先级插队。

---

## 8. 复现实验命令备忘

```bash
# 单独评估（混淆矩阵、spam PRF1、导出 FN CSV）——默认推荐权重
uv run python eval_classify.py \
  --checkpoint runs/spam_classify_phase_b/checkpoint_best.pt

# 基线对照（须存在 runs/spam_classify/checkpoint_best.pt）
# uv run python eval_classify.py --checkpoint runs/spam_classify/checkpoint_best.pt

# 单条推理 + 概率（stderr）
uv run python classify_sms.py \
  --checkpoint runs/spam_classify_phase_b/checkpoint_best.pt \
  --text 'WINNER!! You have been selected for a free prize call now' \
  --probs

# 若需同一终端内顺序稳定的合并输出（示例）
uv run python classify_sms.py \
  --checkpoint runs/spam_classify_phase_b/checkpoint_best.pt \
  --text 'WINNER!! You have been selected for a free prize call now' \
  --probs 2>&1 | tee classify_out.txt
```

---

*本报告随实验进展可迭代修订；修订时请在本节脚注或 Git 历史中保留观测数值与 checkpoint 路径。*
