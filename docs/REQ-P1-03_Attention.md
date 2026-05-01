# REQ-P1-03：多头因果自注意力

**所属**：[SPEC.md](../SPEC.md) → Part I · 模块 03  
**被依赖**：[REQ-P1-04](REQ-P1-04_Model.md)（TransformerBlock 内的注意力层）  
**状态**：✅ 已完成

---

## 1. 业务逻辑（读完就知道「要干嘛」）

### 先打个比方

全班每人都能「回头看」前面的同学，但 **不许偷看后面还没交卷的同桌**——这就是 GPT 的 **因果**：写第 *t* 个字时只能用到前 *t−1* 个字。**多头**就像好几个小组，有的盯语法衔接，有的盯重复词，最后在黑板前汇总。

### 最关键的一句话

> **每个位置只能聚合自己和更早位置的信息**（靠 **因果掩码** 挡住「未来」）；再并行跑 **多个注意力头**，输出交给后面的 Transformer。**不加掩码** 训练会作弊；**只一头** 表达能力偏弱。

---

## 2. 设计思路（怎么做）

**方案**：单个 `MultiHeadAttention` 类，QKV 各一个线性投影 + 输出投影，与书本 ch03/ch04 最终版一致。

**为什么不拆成 SingleHead + MultiHeadWrapper**：
- 书中 ch03 用 `SelfAttention_v1/v2` → `CausalAttention` → `MultiHeadAttentionWrapper` 逐步演进，是教学用
- 生产代码直接实现合并版 MHA 更高效（一次矩阵乘 vs 多次小矩阵乘）

**为什么用 register_buffer 存 mask 而非动态构造**：
- mask 形状固定（`context_length × context_length`），注册为 buffer 随模型自动搬运到正确设备
- 推理时按实际序列长度截取 `mask[:T, :T]`，兼容变长输入

**关键设计决策**：
- `d_out` 必须被 `num_heads` 整除，否则 `assert` 报错
- `qkv_bias` 默认 `False`（与 GPT-2 原版一致），可配置
- Dropout 作用于注意力权重（非 logits），与书本一致

---

## 3. 架构定位（在哪里）

```text
     ┌─────────────────────────────────────┐
     │  TransformerBlock (m04_model)        │
     │                                     │
     │   x ──→ LayerNorm ──→ ┌───────────┐│
     │                       │ MultiHead  ││
     │                       │ Attention  ││  ← 本模块
     │                       │ (m03)      ││
     │                       └─────┬─────┘│
     │   x ──────────────────────+ │       │
     │                       residual      │
     │                           │         │
     │         LayerNorm ──→ FeedForward   │
     │                       residual      │
     └─────────────────────────────────────┘
```

**上游**：TransformerBlock 传入的归一化后隐藏状态 `[B, T, emb_dim]`  
**下游**：TransformerBlock 的残差连接

---

## 4. 输入 / 输出契约

```python
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,          # 输入维度
        d_out: int,         # 输出维度（须被 num_heads 整除）
        context_length: int, # 因果 mask 最大长度
        dropout: float,     # 注意力权重 dropout 率
        num_heads: int,     # 注意力头数
        qkv_bias: bool = False,
    ) -> None

    def forward(self, x: Tensor) -> Tensor
    # x: [B, T, d_in] → [B, T, d_out]
```

---

## 5. 业务规则

| # | 规则 | 说明 | 示例 |
|---|------|------|------|
| R1 | QKV 投影 | 三个 `nn.Linear(d_in, d_out, bias=qkv_bias)` | d_in=128, d_out=128 |
| R2 | 多头分割 | `head_dim = d_out // num_heads`，reshape 为 `[B, heads, T, head_dim]` | 128 / 4 = 32 |
| R3 | 缩放点积 | `Q @ K^T / sqrt(head_dim)`，再 softmax | 除以 √32 ≈ 5.66 |
| R4 | 因果 mask | 上三角填 `-inf`，softmax 后未来位置权重为 0 | token 3 不能看 token 4、5 |
| R5 | Dropout | 作用于 softmax 后的注意力权重 | `drop_rate=0.1` |
| R6 | 输出投影 | 多头合并后经 `nn.Linear(d_out, d_out)` | 恢复原维度 |
| R7 | 维度整除 | `d_out % num_heads != 0` → AssertionError | 128 % 3 → 报错 |
| R8 | 变长截取 | `mask[:T, :T]` 支持 T < context_length | 推理时序列递增 |

---

## 6. 验收标准

| # | 输入 | 预期输出 |
|---|------|---------|
| AC1 | `x: [2, 64, 128]`，4 heads | 输出形状 `[2, 64, 128]` |
| AC2 | `x: [2, 10, 128]`（T < context_length） | 输出形状 `[2, 10, 128]` |
| AC3 | 提取注意力权重矩阵 | `weights[i, j] == 0` 当 `j > i`（因果性） |
| AC4 | 反向传播一步 | 所有梯度为有限实数（无 NaN / Inf） |
| AC5 | `d_out=128, num_heads=3` | `AssertionError` |
| AC6 | `qkv_bias=True` | W_query 等线性层 `bias is not None` |

单测 6 用例，全部 ✅。

---

## 7. 代码落位与依赖

| 项 | 路径 |
|----|------|
| 模块实现 | `src/mini_llm/m03_attention/__init__.py` |
| 测试 | `tests/test_attention.py` |
| 依赖库 | `torch >= 2.0.0` |
