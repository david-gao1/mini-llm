# REQ-P1-05：预训练循环（train.py）

**所属**：[SPEC.md](../SPEC.md) → Part I · 模块 05  
**依赖**：[REQ-P1-01](REQ-P1-01_Tokenizer.md)（词表校验 + decode）、[REQ-P1-02](REQ-P1-02_DataLoader.md)（数据加载）、[REQ-P1-04](REQ-P1-04_Model.md)（GPTModel）、[REQ-P2-01](REQ-P2-01_Generate.md)（采样展示）  
**被依赖**：闸门 M1（预训练闭环）、闸门 M2（生成链路的前置 checkpoint）  
**状态**：✅ 已完成  
**OpenSpec（行为契约）**：[预训练 · `pretraining/spec.md`](../openspec/specs/pretraining/spec.md)

---

## 1. 业务逻辑（读完就知道「要干嘛」）

### 先打个比方

零件齐了，还差一条 **流水线**：接上优化器当电机，让原材料(batch)一遍遍流过「前向 → 算错题(loss) → 反向改参数」，定时称重量(loss 曲线)、抽样验货(打印几句续写)，并把最佳状态装箱(checkpoint)。

### 最关键的一句话

> **`train.py` 负责预训练闭环**：读配置 → 造 DataLoader → 多轮迭代更新权重 → 盯 train/val loss → **写出 checkpoint**。**闸门 M1** 就靠它证明「数据进得去、loss 是正常数、盘上有存档」。

---

## 2. 设计思路（怎么做）

**方案**：单文件脚本 `train.py` + JSON 配置 + argparse。

**为什么不用 PyTorch Lightning / HuggingFace Trainer**：
- 教学项目，目的是理解训练循环的每一步
- 依赖最小化：只需 torch + tiktoken
- 100 行代码足够覆盖完整流程

**为什么用 JSON 配置而非硬编码**：
- 一处修改，全局生效（学习率、epoch、模型维度等）
- 便于对比不同超参的实验

**为什么 checkpoint 保存 optimizer 和 config**：
- 支持断点续训（虽然当前未实现 resume）
- config 随 checkpoint 走，确保加载时配置一致

**关键设计决策**：
- 损失函数：标准 `cross_entropy`，logits 与 target 展平后计算
- global_step 从 0 开始，`optimizer.step()` 后 +1
- 每 epoch 结束时调用 `generate`（temperature / top-k）打印生成样本（P1-06 起与 `generate_text_simple` 贪心模式区分），直观观察训练效果
- 词表不一致时打 warning 而非直接报错，留给开发者判断

---

## 3. 架构定位（在哪里）

```text
     ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
     │ config.json  │    │ m01_tokenizer│    │ m05_generate │
     │ (超参配置)    │    │ (词表校验)    │    │ (采样展示)   │
     └──────┬──────┘    └──────┬───────┘    └──────┬──────┘
            │                  │                   │
            ▼                  ▼                   ▼
     ┌──────────────────────────────────────────────────────┐
     │  train.py                                            │
     │                                                      │
     │  load config → set seed → load text                  │
     │       → train_val_dataloaders (m02)                  │
     │       → GPTModel (m04) + AdamW                       │
     │       → 训练循环:                                     │
     │           forward → cross_entropy → backward → step  │
     │           eval → checkpoint → print_sample            │
     │                                                      │
     │  输出: runs/team_gpt/checkpoint_latest.pt            │
     └──────────────────────────────────────────────────────┘
```

**上游**：所有 m01–m05 模块 + `configs/config.json`  
**下游**：M1 闸门验证、M2 闸门（checkpoint 供 generate 加载）

---

## 4. 输入 / 输出契约

### 输入

```bash
uv run python train.py --config configs/config.json
```

### 输出

- **标准输出**：训练过程中的 loss 日志 + 每 epoch 的生成样本
- **Checkpoint 文件**：`runs/<run_name>/checkpoint_latest.pt`

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "global_step": int,
    "epoch": int,
    "config": dict,  # 完整 config 快照
}
```

### 脚本级函数

```python
calc_loss_batch(input_batch, target_batch, model, device) -> Tensor
calc_loss_loader(data_loader, model, device, num_batches=None) -> float
evaluate_model(model, train_loader, val_loader, device, eval_iter) -> tuple[float, float]
main() -> int  # 0=成功，1=失败
```

---

## 5. 业务规则

| # | 规则 | 说明 | 示例 |
|---|------|------|------|
| R1 | 损失函数 | `cross_entropy(logits.flatten(0,1), target.flatten())` | 标准 CE |
| R2 | 空 DataLoader | `len(loader)==0` → 返回 `float("nan")` | 文本太短时 |
| R3 | 优化器 | AdamW，lr 和 weight_decay 从 config 读取 | lr=3e-4, wd=0.1 |
| R4 | 评估模式 | `model.eval()` + `torch.no_grad()` 下计算 eval loss | 避免 dropout 影响 |
| R5 | 评估频率 | 每 `eval_freq` 步打印 train/val loss | 每 5 步 |
| R6 | Checkpoint | 每 `checkpoint_every_steps` 步 + 训练结束时保存 | 每 50 步 |
| R7 | 文本采样 | 每 epoch 结束用 `generate` 生成至多 50 token（具体 temperature/top-k 以 `train.py` 为准；当前为 temperature=0.8, top_k=25） | `start_context` 配置项起始 |
| R8 | 设备选择 | `"auto"` → CUDA 优先，其次 MPS，否则 CPU（与 P1-06 一致） | — |
| R9 | 词表校验 | 启动时检查 tokenizer vs config 词表，不一致打 warning | 非致命 |
| R10 | 随机种子 | `random` + `torch.manual_seed` + CUDA seed | seed=123 |

---

## 6. 验收标准

| # | 输入 | 预期输出 |
|---|------|---------|
| AC1 | `uv run python train.py --config configs/config.json` | 正常运行，无报错退出 |
| AC2 | 训练若干 step 后的 loss | 为有限实数（非 NaN / Inf） |
| AC3 | 检查 `runs/team_gpt/checkpoint_latest.pt` | 文件存在，可被 `torch.load` 读取 |
| AC4 | checkpoint 内容 | 包含 `model_state_dict`、`optimizer_state_dict`、`global_step`、`epoch`、`config` |
| AC5 | 每 epoch 结束的生成文本 | 非空字符串（内容不要求有意义，但须可 decode） |
| AC6 | **闸门 M1 第 6 项** | train/val loss 为有限实数 |
| AC7 | **闸门 M1 第 7 项** | checkpoint 正确写出 |

**当前状态**：无单测；端到端验证需手动运行。

---

## 7. 代码落位与依赖

| 项 | 路径 |
|----|------|
| 训练脚本 | `train.py`（项目根目录） |
| 配置文件 | `configs/config.json` |
| Checkpoint 输出 | `runs/team_gpt/checkpoint_latest.pt` |
| 数据缓存 | `runs/team_gpt/data_cache/` |
| 依赖模块 | `m01_tokenizer`、`m02_data_loader`、`m04_model`、`m05_generate` |
| 依赖库 | `torch >= 2.0.0`、`tiktoken >= 0.5.0` |
| 语料 | `the-verdict.txt`（需网络或本地文件） |

### 阻塞项

- 端到端训练依赖语料下载（`the-verdict.txt`）和 tiktoken 词表缓存
- 在有网环境下运行一次即可解除
