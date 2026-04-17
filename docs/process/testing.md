# 测试流程规范

从"代码写完"到"确认可交付"的完整流程。目标：**用自动化测试守住质量底线，让重构有信心**。

---

## 1. 流程总览

```
代码已提交
    ↓
[T1] 确认测试覆盖 —— 新代码是否有对应测试？
    ↓
[T2] 运行测试 —— 分层执行，快的先跑
    ↓
[T3] 边界探索 —— 主动找破绽
    ↓
[T4] 回归验证 —— 确保没有破坏已有功能
    ↓
[T5] 记录结果 —— 通过/失败/发现的问题
    ↓
通过 → 标记 REQ 完成
失败 → 回到开发角色修复
```

---

## 2. 测试分层

与 `HARNESS.md` 中的分层体系一致：

| 层级 | 范围 | 速度 | 运行频率 | 示例 |
|------|------|------|----------|------|
| **L0** | 纯函数/单模块 | 秒级 | 每次提交 | `test_tokenizer.py` 中的编解码往返测试 |
| **L1** | 单模块 + 配置 | 秒级 | 每次提交 | 构造 config，验证模型前向形状 |
| **L2** | 模块链 | 十秒级 | 每个 REQ 完成时 | loader → model → loss 的集成测试 |
| **L3** | 端到端 | 分钟级 | 每个闸门前 | `train.py` 完整训练 + checkpoint 生成 |

**原则**：L0/L1 多写、快跑、频繁跑；L2/L3 精写、关键节点跑。

---

## 3. 测试用例编写规范

### 3.1 文件组织

```
tests/
├── test_tokenizer.py       # 对应 m01_tokenizer
├── test_data_loader.py     # 对应 m02_data_loader
├── test_attention.py       # 对应 m03_attention
├── test_model_forward.py   # 对应 m04_model
├── test_generate.py        # 对应 m05_generate
├── test_train.py           # 对应 train.py
└── test_imports.py         # 包导入健全性检查
```

### 3.2 命名规范

```python
def test_<模块>_<行为>_<条件>():
    """测试 <什么模块> 在 <什么条件下> 的 <什么行为>。"""
```

示例：

```python
def test_tokenizer_roundtrip_ascii():
    """encode 后 decode 应还原原始 ASCII 文本。"""

def test_model_forward_output_shape():
    """GPTModel 前向输出形状应为 [B, T, vocab_size]。"""

def test_dataloader_batch_shape_matches_config():
    """DataLoader 输出的 input/target 形状应与 config 中 context_length 一致。"""
```

### 3.3 测试结构（AAA 模式）

每个测试遵循 **Arrange → Act → Assert** 结构：

```python
def test_model_forward_output_shape():
    # Arrange: 准备输入和配置
    cfg = {
        "vocab_size": 50257,
        "context_length": 16,
        "emb_dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = GPTModel(cfg)
    batch = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]))

    # Act: 执行被测行为
    logits = model(batch)

    # Assert: 验证结果
    assert logits.shape == (2, cfg["context_length"], cfg["vocab_size"])
```

### 3.4 固定随机种子

涉及随机性的测试必须固定种子，保证可重复：

```python
import torch

def test_something_with_randomness():
    torch.manual_seed(42)
    # ... 测试逻辑 ...
```

### 3.5 测试独立性

- 每个测试函数**独立运行**，不依赖其他测试的执行顺序或副作用
- 不依赖外部文件（测试数据内联或用 fixture 生成）
- 不依赖网络（离线可跑）

---

## 4. 边界探索清单

主动寻找代码的薄弱点：

### 4.1 通用边界

| 边界类型 | 测试什么 |
|----------|----------|
| **空输入** | 空字符串、空 tensor、batch_size=0 |
| **最小输入** | 单字符、单 token、batch_size=1、context_length=1 |
| **边界值** | vocab_size 边界的 token id（0 和 vocab_size-1） |
| **类型错误** | 传入错误类型时是否有清晰的错误信息 |
| **配置不一致** | vocab_size 与 tokenizer 不匹配时是否报错 |

### 4.2 本项目特有边界

| 模块 | 关注点 |
|------|--------|
| **tokenizer** | 特殊 token 处理、非 ASCII 文本、超长文本 |
| **data_loader** | stride > context_length、文本长度 < context_length、最后一个不完整窗口 |
| **attention** | n_heads 不整除 emb_dim、context_length=1 时的掩码 |
| **model** | 梯度是否有限（无 NaN/Inf）、dropout=0 vs dropout>0 |
| **train** | 1 step 训练后 loss 是否有限、checkpoint 文件是否可加载 |
| **generate** | temperature=0（贪心）、top_k=1、生成长度=0 |

---

## 5. 回归测试策略

### 5.1 什么时候跑回归

| 场景 | 跑什么 |
|------|--------|
| 改了某个模块的内部实现 | 该模块的 L0/L1 + 全量 L0/L1 |
| 改了模块的公开 API | 全量测试 |
| 改了 config.json | 全量测试 |
| 闸门验收前 | 全量测试 + L3 端到端 |

### 5.2 运行命令

```bash
# 快速回归：只跑 L0/L1（秒级）
uv run pytest tests/ -v -k "not slow"

# 全量回归
uv run pytest tests/ -v

# 端到端（标记为 slow 的测试）
uv run pytest tests/ -v -k "slow"

# 单个模块
uv run pytest tests/test_tokenizer.py -v
```

### 5.3 标记慢测试

```python
import pytest

@pytest.mark.slow
def test_train_one_epoch():
    """端到端训练一个 epoch，验证 loss 收敛。"""
    ...
```

在 `pyproject.toml` 或 `pytest.ini` 中注册标记：

```toml
[tool.pytest.ini_options]
markers = [
    "slow: 端到端测试，运行时间较长",
]
```

---

## 6. 测试结果记录

### 6.1 通过时

在 HARNESS.md 对应的 REQ 条目旁标注状态（或用 Issue/commit 追踪）：

```
| P1-03 | `m03_attention` | ... | ✅ 2026-04-12 |
```

### 6.2 失败时

创建 Bug 条目，格式：

```markdown
### BUG: <简述>

- **发现于**: REQ P1-03 测试阶段
- **现象**: attention 输出形状为 [B, T, n_heads] 而非 [B, T, emb_dim]
- **复现**: `uv run pytest tests/test_attention.py::test_attention_output_shape -v`
- **预期**: 输出形状 [B, T, emb_dim]
- **严重程度**: 阻塞（影响下游 P1-04）
```

Bug 修复后，对应的测试用例**保留**作为回归测试。

---

## 7. 与 AI 协作的测试模式

当你对 AI 说"我现在是测试角色"或"帮我测试这个模块"时，AI 应该：

| AI 行为 | 说明 |
|---------|------|
| **对照 HARNESS** | 先看 REQ 的通过判据，确认测试目标 |
| **补充边界用例** | 主动提出你可能遗漏的边界情况 |
| **运行测试** | 帮你执行 pytest 并分析结果 |
| **定位问题** | 测试失败时帮你分析根因 |
| **不改实现** | 测试阶段只改测试代码，不改实现代码（除非是明确的 bug fix） |
| **建议改进** | 发现代码质量问题时记录为 TODO 或 Bug，不直接改 |

---

## 8. 检查清单（测试角色完成时）

- [ ] 每个 REQ 都有对应的测试用例
- [ ] 测试覆盖了正常路径和关键边界
- [ ] 全量测试通过（无新增失败）
- [ ] 发现的 Bug 已记录（含复现步骤）
- [ ] 测试用例命名清晰，结构规范（AAA 模式）
- [ ] 慢测试已标记 `@pytest.mark.slow`
- [ ] 测试不依赖外部资源（网络、特定文件路径）
