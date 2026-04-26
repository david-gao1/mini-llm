# REQ-P2-01：自回归文本生成

**所属**：[SPEC.md](../SPEC.md) → Part II · 模块 05  
**依赖**：[REQ-P1-04](REQ-P1-04_Model.md)（GPTModel 前向推理）  
**被依赖**：[REQ-P1-05](REQ-P1-05_Train.md)（训练中的采样展示）、闸门 M2（加载 checkpoint 生成文本）  
**状态**：✅ 已完成

---

## 1. 业务逻辑（为什么做）

模型训练完成后，最终目的是**生成文本**。给定一个 prompt（如 `"Every effort moves you"`），模型需要自回归地逐 token 续写。但"怎么选下一个 token"有不同策略：

- **贪心（argmax）**：永远选概率最高的 token，输出确定、但容易重复无趣
- **温度采样**：temperature 越高越随机，越低越保守
- **Top-k 过滤**：只从概率最高的 k 个 token 中采样，避免低概率 token 的噪声

训练阶段用贪心采样快速验证模型是否在学（`print_sample`）；推理阶段用 temperature + top-k 获得多样性更好的文本。

---

## 2. 设计思路（怎么做）

**方案**：两个函数分离两种场景。

**为什么分 `generate_text_simple` 和 `generate` 两个函数**：
- `generate_text_simple` 是训练中的快速验证工具，逻辑最简（纯 argmax），与书本 ch04 完全一致
- `generate` 是完整推理入口，支持 temperature / top-k，与书本 ch05 对应
- 职责分离，调用方按需选择

**为什么用 `@torch.no_grad()` 装饰器而非 with 块**：
- 装饰器包裹整个函数，避免遗漏
- 生成函数 100% 不需要梯度，语义更清晰

**为什么 top-k 用 `torch.topk` + `masked_fill` 而非排序**：
- `topk` 是 O(N log k)，比完整排序 O(N log N) 更快
- `masked_fill` 将低概率位置置 `-inf`，softmax 后自然为 0

**关键设计决策**：
- `temperature <= 0` 退化为 argmax（与贪心行为一致）
- 不实现 `eos_id` 早停（当前训练语料无明确结束符需求，保持简洁）
- 上下文截取 `idx[:, -context_size:]`，确保不超过 position embedding 范围

---

## 3. 架构定位（在哪里）

```text
     ┌────────────────────────────────────────────────────┐
     │  调用方                                             │
     │  ├─ train.py print_sample()  → generate_text_simple│
     │  └─ 推理脚本 / M2 验证       → generate             │
     └─────────────────────┬──────────────────────────────┘
                           │
                           ▼
     ┌────────────────────────────────────────────────────┐
     │  m05_generate                                      │
     │                                                    │
     │  for _ in range(max_new_tokens):                   │
     │      idx_cond = idx[:, -context_size:]             │
     │      logits = model(idx_cond)[:, -1, :]            │
     │      next_token = sample(logits)                   │
     │      idx = cat(idx, next_token)                    │
     │                                                    │
     │  return idx  [B, T + max_new_tokens]               │
     └─────────────────────┬──────────────────────────────┘
                           │
                           ▼
     ┌────────────────────────────────────────────────────┐
     │  m01_tokenizer.decode_token_ids()                  │
     │  → 可读文本                                         │
     └────────────────────────────────────────────────────┘
```

**上游**：已训练的 GPTModel（来自 m04）+ 起始 token ids  
**下游**：decode 为人类可读文本（由 m01 tokenizer 完成）

---

## 4. 输入 / 输出契约

### generate_text_simple（贪心）

```python
generate_text_simple(
    model: nn.Module,
    idx: Tensor,           # [B, T] 起始序列
    max_new_tokens: int,
    context_size: int,
) -> Tensor               # [B, T + max_new_tokens]
```

### generate（temperature + top-k）

```python
generate(
    model: nn.Module,
    idx: Tensor,           # [B, T]
    max_new_tokens: int,
    context_size: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> Tensor               # [B, T + max_new_tokens]
```

### 内部辅助

```python
_sample_next(logits: Tensor, temperature: float, top_k: int | None) -> Tensor
# logits: [B, V] → [B, 1]
```

---

## 5. 业务规则

| # | 规则 | 说明 | 示例 |
|---|------|------|------|
| R1 | 上下文截取 | `idx[:, -context_size:]`，不超过 pos_emb 范围 | context_size=64 |
| R2 | 取最后一步 | `logits[:, -1, :]` 作为下一 token 的分布 | 只关注序列末端 |
| R3 | 贪心 | `torch.argmax(logits, dim=-1)` | 确定性输出 |
| R4 | Temperature | `logits / temperature`，控制分布锐度 | T=0.8 更保守，T=1.5 更随机 |
| R5 | Temperature ≤ 0 | 退化为 argmax | 等价贪心 |
| R6 | Top-k | 只保留前 k 个概率最高的 token，其余填 `-inf` | k=25 |
| R7 | Multinomial | softmax 后 `torch.multinomial` 按概率采样 | 非确定性 |
| R8 | 逐步拼接 | `idx = cat(idx, next_token)` 每步增长 1 | 自回归 |
| R9 | no_grad | 整个函数无梯度计算 | 推理优化 |

---

## 6. 验收标准

| # | 输入 | 预期输出 |
|---|------|---------|
| AC1 | `generate_text_simple(model, [B,T], 10, 64)` | `shape == [B, T+10]` |
| AC2 | 输出的每个 token id | ∈ [0, vocab_size) |
| AC3 | `decode_token_ids(output.tolist())` | 非空可读字符串 |
| AC4 | `temperature=0` 的 `generate` | 与 `generate_text_simple` 行为一致 |
| AC5 | `top_k=1` 的 `generate` | 等价贪心（只有一个候选） |
| AC6 | **闸门 M2**：加载 checkpoint 后 generate | 输出非空可 decode 文本 |

单测 1 用例 ✅（`test_generate_step`）。L2/L3 验证需 M1 checkpoint。

---

## 7. 代码落位与依赖

| 项 | 路径 |
|----|------|
| 模块实现 | `src/mini_llm/m05_generate/__init__.py` |
| 测试 | `tests/test_model_forward.py`（`test_generate_step`） |
| 依赖模块 | `mini_llm.m04_model`（GPTModel 前向） |
| 依赖库 | `torch >= 2.0.0` |
