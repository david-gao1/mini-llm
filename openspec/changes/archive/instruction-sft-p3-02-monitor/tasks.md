# 任务清单 · instruction-sft-p3-02-monitor

合并主线时勾选；与 [`specs/instruction-sft/spec.md`](../../../specs/instruction-sft/spec.md) **实施任务与门禁**同步。

## 代码

- [x] `finetune_instruction.py`：`eval_val_batches`、`eval_iter` null、`_resolve_eval_batch_limits`
- [x] `finetune_instruction.py`：按步日志 `val_loss_sampled`；epoch 末 `val_loss_full`
- [x] `finetune_instruction.py`：`epoch_val_full` 与 epoch 末 save
- [x] `finetune_instruction.py`：`run_eval_val_only` + `--eval-val-only` / `--eval-checkpoint`
- [x] `eval_instruction_loss.py` 薄封装
- [x] `compare_instruction_generate.py` + `docs/prompts/instruction_compare_sample.txt`
- [x] `configs/config_instruction_train_small.json`
- [x] `tests/test_instruction_finetune.py`：`test_resolve_eval_batch_limits`

## 规格与 Harness

- [x] `openspec/specs/instruction-sft/spec.md`：需求 + 设计 + 任务 + 归档链接
- [x] `SPEC.md` / `HARNESS.md` / `README.md` 同步
- [x] `docs/OWNER_CHECKLIST.md` Part III 扩充
- [x] `docs/REQ-P3-02` 进度段落

## 阶段 A 文档

- [x] `RUN_REPORT_instruction_sft_small.md` / REQ / OWNER_CHECKLIST 给出 Small-only 检验口径（A1）
- [x] 根 README 显式链 REQ-P3-02，且说明 Medium 不作为本轮 SFT 底座（A2）

## Backlog（非本变更阻塞）

- [ ] 自动评分 / 批量导出 JSON（见 spec **路线图**）
