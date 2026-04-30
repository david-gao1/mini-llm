# REPORT：SMS Spam 分类 — 探针误判与高置信 FN 分析

**性质**：运行分析报告（非 REQ 验收文档）  
**日期上下文**：2026-04（与仓库当期训练/推理链一致）  
**关联需求**：[REQ-P2-02_ClassifyFinetune.md](REQ-P2-02_ClassifyFinetune.md) · [REQ-P2-03_ClassifySmsInfer.md](REQ-P2-03_ClassifySmsInfer.md)  
**相关 backlog**：REQ-P2-02 [§10 可选增强](REQ-P2-02_ClassifyFinetune.md)（BL-P2-02-02 / -04 等）

---

## 1. 背景与目标

项目在 GPT-2 Small 兼容架构（本仓库 `GPTModel` + WikiText-103 预训练 checkpoint）上，按 REQ-P2-02 完成 **SMS 垃圾短信二分类微调**：标签 `ham=0`、`spam=1`，产出 `runs/spam_classify/checkpoint_best.pt`，并由 REQ-P2-03 的 [`classify_sms.py`](../classify_sms.py) 做单条推理。

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
| 推理命令示例 | `uv run python classify_sms.py --checkpoint runs/spam_classify/checkpoint_best.pt --text 'WINNER!! You have been selected for a free prize call now' --probs` |
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

| 序号 | 动作 |
|------|------|
| B1 | `unfreeze_last_n_blocks: 2` 或谨慎增加 epoch |
| B2 | 学习率小幅网格（如 `3e-5`～`8e-5`），仍以 val 选 best |
| B3 | 规则级数据增强（spam，需谨慎） |
| B4 | **BL-P2-02-04**：全量 ham + 类权重 CE，与下采样对照 |

### 7.4 阶段 C：工程体验（可选）

- `classify_sms.py`：`--json` 单行输出等，稳定「标签 + 概率」顺序（REQ-P2-03 backlog 已提及）。
- 文档强调：**聚合指标 ≠ 单句万能**。

### 7.5 阶段 D：架构与预训练（中长期）

- 最后非 pad token / masked mean pooling 与 `[:, -1, :]` 对比。
- **BL-P2-02-03**：OpenAI GPT-2 Small 与本仓库预训练对照（工程量大）。

### 7.6 阶段 A 落地状态（已实现）

- **`mini_llm.m06_classify_finetune`**：`collect_predictions_loader`、`confusion_counts_binary_spam`、`prf1_from_counts`、`export_false_negative_spam_csv`、`format_classification_eval_lines`。
- **`finetune_classify.py`**：训练结束后加载 **best val** 权重（而非最后一 epoch），在 **test** 上打印混淆矩阵与 spam/ham 的 P/R/F1，并将 FN（真实 spam、预测 ham）导出至 ``runs/<run_name>/test_false_negative_spam.csv``。
- **`eval_classify.py`**：对已有 checkpoint 单独跑同上评估（默认 `<data_dir>/test.csv`，FN 默认写入 checkpoint 同目录下的 `eval_false_negative_spam.csv`）。
- **探针清单**：[`docs/probes/classify_spam_probes.json`](probes/classify_spam_probes.json)，供人工或脚本对照 `classify_sms`。

### 7.7 建议节奏（后续）

- **阶段 B**：B1–B2 + 探针记录的定量对照。
- **有空**：B3/B4、C、D 按优先级插队。

---

## 8. 复现实验命令备忘

```bash
# 单独评估（混淆矩阵、spam PRF1、导出 FN CSV）
uv run python eval_classify.py \
  --checkpoint runs/spam_classify/checkpoint_best.pt

# 单条推理 + 概率（stderr）
uv run python classify_sms.py \
  --checkpoint runs/spam_classify/checkpoint_best.pt \
  --text 'WINNER!! You have been selected for a free prize call now' \
  --probs

# 若需同一终端内顺序稳定的合并输出（示例）
uv run python classify_sms.py \
  --checkpoint runs/spam_classify/checkpoint_best.pt \
  --text 'WINNER!! You have been selected for a free prize call now' \
  --probs 2>&1 | tee classify_out.txt
```

---

*本报告随实验进展可迭代修订；修订时请在本节脚注或 Git 历史中保留观测数值与 checkpoint 路径。*
