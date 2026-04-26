# REQ-P1-06：训练优化（过拟合治理 + 设备加速）

**所属**：[SPEC.md](../SPEC.md) → Part I · train.py 优化  
**依赖**：[REQ-P1-05](REQ-P1-05_Train.md)（train.py 基础训练循环已跑通）  
**被依赖**：闸门 M1/M2 已通过；本 REQ 为训练质量与效率提升  
**状态**：✅ 已完成

---

## 1. 业务逻辑（为什么做）

M1/M2 闸门已通过，训练链路跑通了——但训练结果暴露了严重的**过拟合**问题：

| 指标 | 现象 |
|------|------|
| train_loss | 10.8 → **0.005**（模型把 ~20KB 语料完全背下来了） |
| val_loss | 6.1 → **7.2**（Epoch 30 后持续上升，泛化越来越差） |
| 生成文本 | Epoch 30 后完全固定为同一段原文复述，一字不变 |

此外还有两个工程问题：

- **M3 Max 白跑 CPU**：当前 `_pick_device` 只认 CUDA，Mac 的 MPS 加速完全没用上
- **贪心采样无多样性**：`print_sample` 用纯 argmax，模型一旦收敛就永远输出同一段

这些问题不影响"能跑通"，但影响"跑得好"。作为教学项目，治理过拟合的过程本身就是重要的学习内容。

---

## 2. 设计思路（怎么做）

**方案**：5 个优化手段集中改 train.py + config，不动 m01-m05 任何模块代码。

### 为什么选这 5 个手段

| 手段 | 为什么选 | 为什么不选其他 |
|------|---------|--------------|
| MPS 设备 | M3 Max 硬件白白浪费，一行代码提速 3-5x | — |
| 梯度裁剪 | 防止梯度爆炸，训练更稳定，几乎无成本 | — |
| 学习率调度 | 固定 lr 在后期仍然很大，持续推向过拟合 | 不选手动分段调参（不够通用） |
| Early stopping | val_loss 最优点在 Epoch 30，后面 70 epoch 全是浪费 | 不选减小模型（教学目的要保持模型规模） |
| Temperature 采样 | 纯 argmax 看不出模型真实能力 | 不选 nucleus/top-p（当前 top-k 已够用） |

### 为什么不换更大语料 / 减小模型

当前是教学项目，目的是理解训练技术。保持 29M 参数 + 20KB 语料不变，通过**正则化和训练策略**来治理过拟合，比换数据更有学习价值。

### 关键设计决策

- 所有新增参数都有合理默认值，不传也能跑（向后兼容）
- Early stopping 默认 `patience=10`，设 0 可关闭
- Best checkpoint 独立于 latest checkpoint，不互相覆盖
- Cosine scheduler 使用 PyTorch 内置 `CosineAnnealingLR`，不自己造轮子

---

## 3. 架构定位（在哪里）

```text
     ┌────────────────────────────────────────────────────────┐
     │  train.py 改动点                                        │
     │                                                        │
     │  _pick_device()                                        │
     │    └─ 新增: CUDA > MPS > CPU               ← ① MPS    │
     │                                                        │
     │  训练循环 inner loop:                                    │
     │    loss.backward()                                     │
     │    clip_grad_norm_(max_norm)                 ← ② 裁剪  │
     │    optimizer.step()                                    │
     │    scheduler.step()                          ← ③ 调度  │
     │                                                        │
     │  评估时:                                                │
     │    if val_loss < best → save best.pt         ← ④ best  │
     │    if patience 耗尽 → early stop             ← ④ 早停  │
     │                                                        │
     │  print_sample():                                       │
     │    generate(temperature=0.8, top_k=25)       ← ⑤ 采样  │
     └────────────────────────────────────────────────────────┘

     ┌────────────────────────────────────────────────────────┐
     │  configs/config.json 新增字段                           │
     │    train.grad_clip / warmup_ratio / min_lr_ratio       │
     │    train.patience                                      │
     │    drop_rate: 0.1→0.2 / weight_decay: 0.1→0.2         │
     └────────────────────────────────────────────────────────┘
```

**改动范围**：仅 `train.py` + `configs/config.json`。m01-m05 零改动。

---

## 4. 输入 / 输出契约

### config.json 新增字段

```python
train.grad_clip: float      # 梯度裁剪阈值（默认 1.0）
train.warmup_ratio: float   # warmup 占总步数比例（默认 0.1）
train.min_lr_ratio: float   # cosine 衰减下限 = lr * min_lr_ratio（默认 0.1）
train.patience: int          # early stopping 容忍次数（默认 10，0=不启用）
```

所有字段可选，不传则使用默认值（向后兼容旧配置）。

### config.json 调整字段

```python
model.drop_rate: 0.1 → 0.2       # 加大 dropout
train.weight_decay: 0.1 → 0.2    # 加大权重衰减
train.learning_rate: 5e-4 → 3e-4 # 配合 scheduler 略降起始 lr
train.num_epochs: 100 → 50       # 配合 early stopping 不需要跑这么久
```

### 输出新增

- `runs/<run_name>/checkpoint_best.pt` — val_loss 最优时的 checkpoint（与 `checkpoint_latest.pt` 共存）

---

## 5. 业务规则

| # | 规则 | 说明 | 示例 |
|---|------|------|------|
| R1 | MPS 设备 | auto 模式优先级：CUDA > MPS > CPU | M3 Max → MPS |
| R2 | 梯度裁剪 | `clip_grad_norm_` 在 `backward()` 之后、`step()` 之前 | max_norm=1.0 |
| R3 | 总步数计算 | `total_steps = num_epochs * len(train_loader)` | 用于 scheduler |
| R4 | Warmup 阶段 | 前 `warmup_ratio * total_steps` 步，lr 从 0 线性升到 `learning_rate` | 前 10% step |
| R5 | Cosine 衰减 | warmup 后，lr 从 `learning_rate` 余弦衰减到 `lr * min_lr_ratio` | 衰减到 3e-5 |
| R6 | Best checkpoint | 每次评估时若 `val_loss < best_val_loss`，保存 `checkpoint_best.pt` 并更新 best | 独立于 latest |
| R7 | Early stopping | 连续 `patience` 次评估 val_loss 未刷新 best → 打印提示并 break | patience=10 |
| R8 | Patience=0 | 不启用 early stopping，训练跑满 num_epochs | 向后兼容 |
| R9 | 采样多样性 | `print_sample` 用 `generate(temperature=0.8, top_k=25)` 替代 `generate_text_simple` | 每次有变化 |
| R10 | 向后兼容 | 新增 config 字段全部可选，不传用默认值 | 旧 config 可直接跑 |

---

## 6. 验收标准

| # | 输入 / 场景 | 预期输出 |
|---|-------------|---------|
| AC1 | M3 Max 上运行，config `device="auto"` | 日志显示设备为 MPS |
| AC2 | 训练 50 epoch | val_loss 不再从 Epoch 30 后持续上升（曲线拐点后趋于平稳或被 early stop） |
| AC3 | 检查 `runs/team_gpt/` | 同时存在 `checkpoint_latest.pt` 和 `checkpoint_best.pt` |
| AC4 | `checkpoint_best.pt` 对应的 val_loss | 为全程最低值 |
| AC5 | 设置 `patience=10`，val_loss 持续不降 | 训练提前终止，打印 early stopping 提示 |
| AC6 | 设置 `patience=0` | 训练跑满 num_epochs，不触发 early stopping |
| AC7 | 连续两个 epoch 的 `print_sample` 输出 | 文本不完全相同（temperature 采样带随机性） |
| AC8 | 使用旧 config（无新增字段） | 正常运行，使用默认值 |
| AC9 | `drop_rate=0.2` + `weight_decay=0.2` 训练 | train_loss 与 val_loss 的差距比之前小 |

---

## 7. 代码落位与依赖

| 项 | 路径 |
|----|------|
| 训练脚本（改动） | `train.py` |
| 配置文件（改动） | `configs/config.json` |
| Latest checkpoint | `runs/team_gpt/checkpoint_latest.pt` |
| Best checkpoint（新增） | `runs/team_gpt/checkpoint_best.pt` |
| 依赖模块 | 无新增（仍为 m01-m05） |
| 依赖库 | 无新增（`torch.optim.lr_scheduler` 为 torch 内置） |
