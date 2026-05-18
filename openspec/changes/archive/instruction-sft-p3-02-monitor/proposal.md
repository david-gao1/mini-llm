# 提案：指令 SFT 监控与对照（P3-02）

## 背景

P3-01 已交付「能训、能存盘、能冒烟」；训练中仍存在以下 **可观测行为缺口**（见 REQ-P3-02 Q-2～Q-3）：

- 日志中的 `val_loss` 实为 **抽样 batch**，易被误读为全验证集指标。
- **epoch 末**验证优于中途 eval 时，**未必触发** `checkpoint_best.pt` 更新。
- 主观对照生成缺少 **固定解码参数** 下的并排脚本。

## 目标（对外）

1. 日志区分 **`val_loss_sampled`** 与 **`val_loss_full`**。  
2. 配置支持按步评估时 val **扫满**（`eval_val_batches: null`）或 `eval_iter: null`（极小数据）。  
3. **`epoch_val_full`**（默认开）：epoch 末用 full val 参与 **best** 比较并可覆盖保存。  
4. **不写权重**的 val 评估 CLI（`--eval-val-only` / `eval_instruction_loss.py`）。  
5. **双 checkpoint** 对照生成（`compare_instruction_generate.py`）。

## 非目标（本轮不做）

- 自动 LLM 评分、批量导出带参考答案 JSON（仍为路线图 backlog）。
- 改变 causal LM 或 collate 的数学定义。

## 验收

以 [`specs/instruction-sft/spec.md`](../../../specs/instruction-sft/spec.md) **需求 / 场景**与 **Tasks** 小节为准；命令判据见根目录 [`HARNESS.md`](../../../../HARNESS.md) Part III。
