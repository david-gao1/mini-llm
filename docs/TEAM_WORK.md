# 项目工作内容与分工

章节任务与交付物如下；合并与验收由组长统筹（见最后一行）。

| 章节 | 负责人 | 任务 | 交付物 |
|------|--------|------|--------|
| 第 2 章：文本处理与分词 | 同学 A | 实现 BPE、数据加载、滑动窗口 | `tokenizer.py`、`data_loader.py` |
| 第 3 章：注意力机制 | 同学 B | 实现 Self-Attention、因果掩码、Multi-Head | `attention.py` |
| 第 4 章：GPT 模型搭建 | 同学 C | 搭建 TransformerBlock、完整 GPT | `model.py` |
| 第 5 章：预训练与损失 | 同学 D | 训练循环、损失函数、评估 | `train.py` |
| 第 6/7 章：生成与微调 | 同学 E | 文本生成、采样策略（top-k / temperature） | `generate.py` |
| 全组整合 + 汇报 | 组长 | 合并代码、跑通流程、做 PPT、写报告 | 完整项目 + 演示 |

实现代码均放在 [`src/mini_llm/`](../src/mini_llm/)；根目录 [`train.py`](../train.py) 为训练入口脚本（由同学 D 主责，整合时与数据、模型模块对接）。

接口约定见 [MODULE_INTERFACES.md](./MODULE_INTERFACES.md)，范围见 [SCOPE.md](./SCOPE.md)。组长排期、依赖与 Harness 约定见 [PROJECT_GOVERNANCE.md](./PROJECT_GOVERNANCE.md)。
