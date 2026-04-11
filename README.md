# 微型 LLM — 团队主项目（从零实现）

本目录与官方书本仓库 **`LLMs-from-scratch` 同级**，互不嵌套，避免与上游混淆。

```
001-360/
├── LLMs-from-scratch/     # 原书配套代码（勿改乱；仅作阅读 / 对照）
└── team-mini-llm/         # 本仓库：你们手写的实现与实验
```

## 环境

在书本仓库中安装可编辑包（提供 `llms_from_scratch` 供**对照**或可选 import）：

```bash
cd ../LLMs-from-scratch
pip install -e .
```

回到本目录，将 `src` 加入路径后开发：

```bash
cd /path/to/team-mini-llm
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
python train.py --help
```

## 文档与代码

| 路径 | 说明 |
|------|------|
| [`docs/`](docs/) | 范围、环境、模块接口约定 |
| [`configs/config.json`](configs/config.json) | 超参与数据相关配置 |
| [`src/mini_llm/`](src/mini_llm/) | 你们实现的包（attention / model / data / generate） |
| [`train.py`](train.py) | 训练入口（实现模块后在此串联） |
| [`REFERENCE.md`](REFERENCE.md) | 如何对照隔壁书本仓库的章节与路径 |

## 对照跑通（可选）

若需先用官方包跑通流程，可在书本仓库内使用其自带章节脚本（见 [`REFERENCE.md`](REFERENCE.md)），**不要**把主实现长期依赖在 `import llms_from_scratch` 上。
