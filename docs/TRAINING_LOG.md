# 训练运行记录（模型 / 配置切换）

便于回溯「当时在跑哪套 config、日志文件名、为何切换」。

---

## 2026-04-29

| 动作 | 说明 |
|------|------|
| **停止** | 终止当时在跑的 **`config_medium.json`**（`gpt2_medium_wikitext103`）训练进程，以便切换 GPT-2 Small。 |
| **切换** | 主实验改为 **GPT-2 Small 量级**：`configs/config_gpt2_small.json`，`run_name=gpt2_small_wikitext103`。架构 **12×768×12**，WikiText-103 raw，`num_epochs=2`。 |
| **数据缓存** | 从 `runs/gpt2_medium_wikitext103/data_cache/` 复制 `train_tokens.pt` / `val_tokens.pt` 至 `runs/gpt2_small_wikitext103/data_cache/`（同源语料与 split，避免重复 tokenize）。 |
| **日志文件** | `train_gpt2_small.log`（项目根目录）。 |

启动命令（示例）：

```bash
cd /path/to/team-mini-llm
nohup env PYTHONUNBUFFERED=1 "$(pwd)/.venv/bin/python" -u "$(pwd)/train.py" \
  --config "$(pwd)/configs/config_gpt2_small.json" \
  > "$(pwd)/train_gpt2_small.log" 2>&1 &
tail -f train_gpt2_small.log
```

---

## 2026-04-30（补充）— GPT-2 Small 本轮跑完

| 项 | 值 |
|----|-----|
| **val_loss（best）** | **3.3092**（`checkpoint_best.pt`） |
| **步数 / 耗时** | **54967** steps，约 **16h 39m** |
| **运行报告** | [`RUN_REPORT_gpt2_small_wikitext103.md`](RUN_REPORT_gpt2_small_wikitext103.md) |

