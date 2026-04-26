# REQ-P1-04：GPT 模型（TransformerBlock + GPTModel）

**所属**：[SPEC.md](../SPEC.md) → Part I · 模块 04  
**依赖**：[REQ-P1-03](REQ-P1-03_Attention.md)（MultiHeadAttention）  
**被依赖**：[REQ-P1-05](REQ-P1-05_Train.md)（训练循环）、[REQ-P2-01](REQ-P2-01_Generate.md)（自回归生成）  
**状态**：✅ 已完成

---

## 1. 业务逻辑（为什么做）

前三个模块分别实现了分词、数据加载、注意力机制，但这些组件还是散装的。需要将它们组装成完整的 GPT 模型——具体来说：

- **Token Embedding** + **Position Embedding**：将离散 id 映射为连续向量，并注入位置信息
- **N 层 TransformerBlock**：每层含注意力 + 前馈网络 + 残差连接 + 层归一化
- **LM Head**：将隐藏状态映射回词表维度，输出 logits 供 softmax / argmax 使用

这是整个项目的核心模块。输入 token id `[B, T]`，输出 logits `[B, T, vocab_size]`，上接数据、下接训练和生成。

---

## 2. 设计思路（怎么做）

**方案**：Pre-Norm Transformer（LayerNorm 在注意力/FFN 之前），与 GPT-2 原版和书本 ch04 一致。

**为什么选 Pre-Norm 而非 Post-Norm**：
- GPT-2/3 均采用 Pre-Norm，训练更稳定
- 书本 ch04 采用此结构，对照方便

**为什么用近似 GELU 而非 PyTorch 内置**：
- 书本显式实现了 tanh 近似版，便于理解公式
- 与 GPT-2 原版的 GELU 近似一致

**为什么自定义 LayerNorm 而非用 nn.LayerNorm**：
- 教学目的：显式展示均值/方差计算与可学习 scale/shift
- `unbiased=False`（分母为 N 而非 N-1），与书本一致

**关键设计决策**：
- `self.cfg = cfg` 保存在 GPTModel 中，训练脚本可通过 `model.cfg` 读取配置
- LM Head **不带 bias**（`bias=False`），与 GPT-2 原版一致
- FeedForward 的隐藏层维度固定为 `4 * emb_dim`（标准 Transformer 设计）

---

## 3. 架构定位（在哪里）

```text
     ┌────────────────────────────────────────────────────────┐
     │  GPTModel                                              │
     │                                                        │
     │  in_idx [B, T] ──→ tok_emb + pos_emb ──→ drop_emb     │
     │                                            │           │
     │                   ┌────────────────────────┘           │
     │                   ▼                                    │
     │            ┌─────────────────┐                         │
     │            │ TransformerBlock │ × n_layers              │
     │            │  ├─ LayerNorm    │                         │
     │            │  ├─ MHA (m03)   │                         │
     │            │  ├─ residual    │                         │
     │            │  ├─ LayerNorm    │                         │
     │            │  ├─ FeedForward │                         │
     │            │  └─ residual    │                         │
     │            └─────────────────┘                         │
     │                   │                                    │
     │                   ▼                                    │
     │            final_norm ──→ out_head ──→ logits [B,T,V]  │
     └────────────────────────────────────────────────────────┘
```

**上游**：m02 DataLoader 提供的 token id batch `[B, T]`  
**下游**：train.py 计算 cross_entropy loss；m05 generate 取 logits 采样

---

## 4. 输入 / 输出契约

```python
class GPTModel(nn.Module):
    def __init__(self, cfg: dict) -> None
    def forward(self, in_idx: Tensor) -> Tensor
    # in_idx: [B, T] int64 token ids → logits: [B, T, vocab_size] float
```

cfg 字典需包含：`vocab_size`、`context_length`、`emb_dim`、`n_heads`、`n_layers`、`drop_rate`、`qkv_bias`。

### 子组件

```python
class LayerNorm(nn.Module)      # emb_dim → emb_dim
class GELU(nn.Module)           # 无参数，逐元素激活
class FeedForward(nn.Module)    # emb_dim → 4*emb_dim → emb_dim
class TransformerBlock(nn.Module)  # emb_dim → emb_dim
```

---

## 5. 业务规则

| # | 规则 | 说明 | 示例 |
|---|------|------|------|
| R1 | Token Embedding | `nn.Embedding(vocab_size, emb_dim)` | 50257 × 128 |
| R2 | Position Embedding | `nn.Embedding(context_length, emb_dim)` | 64 × 128 |
| R3 | Embedding Dropout | 两者相加后施加 `Dropout(drop_rate)` | 0.1 |
| R4 | Pre-Norm | LayerNorm 在 MHA / FFN **之前** | 非 Post-Norm |
| R5 | 残差连接 | `x = x + sublayer(norm(x))` | 梯度直通 |
| R6 | FFN 扩展比 | 隐藏层 = `4 * emb_dim` | 128 → 512 → 128 |
| R7 | GELU 近似 | `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))` | tanh 版 |
| R8 | LayerNorm | 沿最后维度，`unbiased=False`，可学习 scale/shift | eps=1e-5 |
| R9 | LM Head | `nn.Linear(emb_dim, vocab_size, bias=False)` | 128 → 50257 |
| R10 | 堆叠层数 | `n_layers` 个 TransformerBlock 顺序连接 | 2 层 |

---

## 6. 验收标准

| # | 输入 | 预期输出 |
|---|------|---------|
| AC1 | `in_idx: [2, 64]` int64 | `logits.shape == (2, 64, 50257)` 且 dtype float |
| AC2 | 单步 generate：取 `logits[:, -1, :]` 的 argmax | 得到有效 token id ∈ [0, 50257)，拼接后 shape `[2, 65]` |

单测 2 用例，全部 ✅。

---

## 7. 代码落位与依赖

| 项 | 路径 |
|----|------|
| 模块实现 | `src/mini_llm/m04_model/__init__.py` |
| 测试 | `tests/test_model_forward.py` |
| 依赖模块 | `mini_llm.m03_attention`（MultiHeadAttention） |
| 依赖库 | `torch >= 2.0.0` |
| 配置文件 | `configs/config.json` → `model.*` 字段 |
