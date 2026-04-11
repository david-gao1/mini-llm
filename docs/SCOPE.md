# 项目范围（冻结）

分工、负责人与交付物文件名见 **[TEAM_WORK.md](./TEAM_WORK.md)**（与下列阶段一一对应）。

## 必做（约第 1–2 周）

| 阶段 | 对应书本章节 | 完成标准 | 交付物（代码） |
|------|----------------|----------|----------------|
| 数据与分词 | 第 2 章 | BPE、文本加载、滑动窗口 DataLoader，与 tokenizer 对齐 | `tokenizer.py`、`data_loader.py` |
| 注意力 | 第 3 章 | Self-Attention、因果掩码、Multi-Head | `attention.py` |
| 模型搭建 | 第 4 章 | TransformerBlock、完整 GPT，前向可出 logits | `model.py` |
| 预训练 | 第 5 章 | train/val loss、checkpoint、可恢复训练 | `train.py` |

## 第 3 周（二选一）

- **A** 第 6 章式分类微调；**B** 第 7 章式指令微调。详见 [WEEK3.md](./WEEK3.md)。
- **生成与采样**：文本生成，top-k / temperature 等（与第 6/7 章任务衔接），交付物 `generate.py`。见 [TEAM_WORK.md](./TEAM_WORK.md)。

## 整合与汇报

- 组长：合并分支、端到端跑通、演示、PPT、报告（见 [TEAM_WORK.md](./TEAM_WORK.md)）。

## 不做（当期）

附录 D/E 全套、Llama/Qwen bonus、多卡 DDP（除非环境已就绪）。
