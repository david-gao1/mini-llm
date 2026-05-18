# 设计：P3-02 监控与对照

## 决策摘要

| 话题 | 选择 | 备选（未采纳原因） |
|------|------|---------------------|
| 全 val loss | `calc_loss_loader_instruction(..., num_batches=None)` 扫完整 Loader | 另维护 Streaming CE（不必要复杂） |
| train vs val 抽样脱钩 | 新增 `eval_val_batches`，默认跟随 `eval_iter` | 仅 `eval_iter`：无法「train 快、val 准」 |
| best 口径 | 单一标量 `best_val_loss`，可被抽样或 full 更新 | 双 checkpoint 文件（增加磁盘与 SPEC 复杂度） |
| 对照生成 | `subprocess` 两次 `generate_from_checkpoint.py` | 直接 import 生成（易漂移 argparse） |

## 模块交互

```text
finetune_instruction.py
  ├─ m07：Dataset / collate / ignore_index
  ├─ m04：GPTModel
  └─ torch.save(checkpoint_best.pt)

eval_instruction_loss.py  ──► run_eval_val_only()  （与 main 共用 collate 路径）

compare_instruction_generate.py  ──► generate_from_checkpoint.py ×2
```

## 配置解析

`_resolve_eval_batch_limits(i_cfg)`：

- `eval_iter is None` → `(None, None)`（train/val 按步均 full）。
- 否则 `train_cap = int(eval_iter)`；若存在键 `eval_val_batches`：`None` → val full，`int` → cap。

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| `eval_val_batches: null` + 大 val → 按步极慢 | 文档标明；可调大 `eval_freq` |
| `best_val_loss` 抽样与 full 混比 | spec **设计与取舍 §4** 写明；严口径用全程 full val |

## 与正式规格的关系

行为契约与任务门禁已迁入 [`specs/instruction-sft/spec.md`](../../../specs/instruction-sft/spec.md) 的 **技术设计（Design）**、**实施任务（Tasks）**、**需求** 各节；本文档为归档快照。
