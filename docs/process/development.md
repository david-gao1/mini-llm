# 开发流程规范

从"拿到 REQ"到"代码可提交"的完整流程。目标：**写出可维护的代码，而不是能跑就行的代码**。

---

## 1. 流程总览

```
拿到 REQ 条目
    ↓
[D1] 设计 —— 想清楚怎么实现，画出模块关系
    ↓
[D2] 编码 —— 按规范写代码
    ↓
[D3] 自测 —— 开发者自己先跑一遍测试
    ↓
[D4] 提交 —— 小步提交，message 说清楚
    ↓
[D5] 自审 —— 用 AI 做 code review
    ↓
移交测试
```

---

## 2. D1 · 设计阶段

动手写代码之前，花 5~15 分钟想清楚：

### 2.1 要回答的问题

| 问题 | 写在哪 |
|------|--------|
| 新增/修改哪些文件？ | commit 计划 |
| 公开 API 是什么？（函数签名、输入输出） | 模块的 docstring 或 `docs/` 下的设计文档 |
| 与上下游模块的接口怎么对齐？ | 对照 HARNESS 中的契约 |
| 有没有需要特别注意的边界情况？ | 测试用例的种子 |

### 2.2 设计文档

- **简单改动**（< 50 行）：不需要单独文档，commit message 说清楚即可
- **中等改动**（新函数/新类）：在对应模块的 `docs/` 文档中补充一节
- **大改动**（新模块/架构调整）：在 `docs/` 下新建设计文档，格式参照 `docs/m01_tokenizer.md`

---

## 3. D2 · 编码规范

### 3.1 项目结构约定

```
src/mini_llm/
├── m01_tokenizer/     # 每个模块一个目录
│   └── __init__.py    # 实现写在 __init__.py 中
├── m02_data_loader/
│   └── __init__.py
├── ...
tests/
├── test_tokenizer.py  # 测试文件与模块对应
├── test_model_forward.py
├── ...
```

### 3.2 Python 编码规范

| 规则 | 说明 |
|------|------|
| **命名** | 模块/变量用 `snake_case`，类用 `PascalCase`，常量用 `UPPER_SNAKE` |
| **类型标注** | 公开函数必须有类型标注；内部函数建议有 |
| **docstring** | 公开函数/类必须有 docstring，说明参数、返回值、异常 |
| **import 顺序** | 标准库 → 第三方 → 本项目，各组之间空一行 |
| **行长** | 不超过 100 字符 |
| **不要** | 不要 `from xxx import *`；不要裸 `except`；不要硬编码魔法数字 |

### 3.3 接口契约优先

```python
# 好：接口清晰，契约明确
def create_dataloader(
    text: str,
    encoding: tiktoken.Encoding,
    context_length: int,
    stride: int,
    batch_size: int,
) -> DataLoader:
    """创建滑动窗口 DataLoader。

    Args:
        text: 原始文本
        encoding: tiktoken 编码器
        context_length: 上下文窗口长度
        stride: 滑动步长
        batch_size: 批大小

    Returns:
        DataLoader，每个 batch 包含 input [B, T] 和 target [B, T]
    """
```

```python
# 坏：参数含义不明，没有类型，没有文档
def make_dl(txt, enc, cl, s, bs):
    ...
```

### 3.4 配置管理

- 所有超参数从 `configs/config.json` 读取，**不在代码中硬编码**
- 新增配置项时，同步更新 config 文件和对应文档
- 配置变更视为需求变更，需要更新 HARNESS

### 3.5 错误处理

```python
# 好：明确的错误信息，帮助定位问题
if cfg["model"]["vocab_size"] != encoding.n_vocab:
    raise ValueError(
        f"config.model.vocab_size ({cfg['model']['vocab_size']}) "
        f"!= encoding.n_vocab ({encoding.n_vocab}); "
        f"请确保 tokenizer 与 config 一致"
    )
```

---

## 4. D3 · 自测

代码写完后，**开发者自己先跑一遍**，不要直接扔给"测试角色"：

```bash
# 跑与当前模块相关的测试
uv run pytest tests/test_xxx.py -v

# 跑全量测试，确保没有破坏其他模块
uv run pytest -v

# 如果改了 train.py，跑一次短训练验证
uv run python train.py --config configs/config.json
```

自测通过的标准：
- [ ] 相关测试全部 PASS
- [ ] 全量测试无新增 FAIL
- [ ] 无 warning（或 warning 已知且可接受）

---

## 5. D4 · 提交规范

### 5.1 Commit Message 格式

```
<类型>(<范围>): <简述>

<详细说明（可选）>
```

**类型**：

| 类型 | 用途 |
|------|------|
| `feat` | 新功能 |
| `fix` | 修复 bug |
| `refactor` | 重构（不改变行为） |
| `test` | 增加/修改测试 |
| `docs` | 文档变更 |
| `chore` | 构建/工具/配置变更 |

**示例**：

```
feat(m03_attention): 实现因果掩码的 MultiHeadAttention

- 支持 n_heads 参数，与 config.json 中 model.n_heads 对齐
- 输出形状 [B, T, emb_dim]，满足 P1-03 契约
- 关联 REQ: P1-03
```

### 5.2 提交粒度

| 好 | 坏 |
|----|----|
| 一个 commit 做一件事 | 一个 commit 改了 5 个不相关的文件 |
| `feat(m03): 实现 CausalAttention` | `更新了一些代码` |
| 先 refactor 再 feat，分两个 commit | 重构和新功能混在一起 |

### 5.3 什么时候提交

- 一个函数/类写完并自测通过 → 提交
- 一个 REQ 的某个子步骤完成 → 提交
- 要切换到另一个任务之前 → 提交当前进度
- **不要**：攒一天的代码一次性提交

---

## 6. D5 · 自审（AI Code Review）

提交前，让 AI 帮你审查代码：

### 审查清单

| 维度 | 检查项 |
|------|--------|
| **契约** | 函数签名是否与 HARNESS 中的契约一致？ |
| **命名** | 变量/函数名是否清晰、一致？ |
| **边界** | 空输入、极端值、类型错误是否处理？ |
| **耦合** | 是否依赖了不该依赖的模块内部实现？ |
| **重复** | 是否有可以提取的公共逻辑？ |
| **性能** | 是否有明显的性能问题？（不必过早优化，但别写 O(n³) 的低级错误） |
| **文档** | docstring 是否与实现一致？ |

---

## 7. 与 AI 协作的开发模式

当你对 AI 说"我现在是开发角色"或直接给出编码任务时，AI 应该：

| AI 行为 | 说明 |
|---------|------|
| **对照 REQ** | 先确认要实现的 REQ 是什么，契约是什么 |
| **遵守编码规范** | 类型标注、docstring、命名规范一个不少 |
| **写测试** | 实现代码的同时写对应的测试 |
| **小步交付** | 不一口气写 500 行，而是分模块逐步实现 |
| **解释设计决策** | 对非显而易见的实现选择给出理由 |
| **不改需求** | 发现需求有问题时提出来，但不自作主张改 |

---

## 8. 技术债标记

发现代码中有"先凑合"的地方，用 `TODO` 标记并注明原因：

```python
# TODO(P1-05): 当前 eval 只算 train loss，后续需加 val loss（REQ P1-05 完整实现时处理）
```

格式：`TODO(<关联REQ或责任人>): <描述>`

定期清理：每个迭代结束时，review 所有 TODO，决定是修复还是转为正式 REQ。

---

## 9. 检查清单（开发角色完成时）

- [ ] 代码符合编码规范（命名、类型标注、docstring）
- [ ] 接口与 HARNESS 中的契约一致
- [ ] 相关测试已编写并通过
- [ ] 全量测试无新增失败
- [ ] commit message 格式正确，粒度合理
- [ ] 如果新增了配置项，config.json 已更新
- [ ] 如果改了公开 API，设计文档已更新
- [ ] 代码中的 TODO 已标注关联 REQ
