# Owner Checklist：项目负责人验收手册

**读者**：项目负责人（有工程经验、正在学习 ML 理论）  
**用途**：拿到一个 REQ「完成」的交付后，**自己跑命令、看输出、判断是否合格**——不依赖开发者口头说「过了」  
**与 SPEC/HARNESS 的关系**：SPEC 定义 API 契约，HARNESS 定义闸门；本文档定义 **你作为 Owner 的「验收动作 + 判断准则」**

---

## 怎么用这份文档

1. 找到对应 REQ 的章节
2. **逐条跑命令**（可直接复制粘贴到终端）
3. 对照「看什么 / 怎么判断」，自己做出 ✅ 或 ❌ 的结论
4. 遇到 ❌ 或拿不准的，记下关键输出行，拿来讨论

> **约定**：所有命令默认在 `team-mini-llm/` 目录下执行。

---

## Part I — 预训练闭环

### P1-01 · m01_tokenizer（分词器）

**一句话**：把英文字符串变成整数列表（token ID），再变回来，信息不丢。

| # | 你跑什么 | 你看什么 | 怎么判断 |
|---|---------|---------|---------|
| 1 | `uv run pytest tests/test_tokenizer.py -q` | 最后一行 `N passed` | 全部 passed、无 FAILED |
| 2 | 在 Python 里试：`from mini_llm.m01_tokenizer import encode_text, decode_token_ids; ids = encode_text("Hello world"); print(ids); print(decode_token_ids(ids))` | 先打印一串整数，再打印回 `Hello world` | 往返一致，没有乱码或丢字 |

**Java 类比**：相当于一个 `Codec<String, List<Integer>>`，`encode → decode` 是幂等的。

---

### P1-02 · m02_data_loader（数据加载器）

**一句话**：把一大段文本切成固定长度的滑动窗口，每个窗口输出 `(input, target)` 对，target 比 input 右移一位。

| # | 你跑什么 | 你看什么 | 怎么判断 |
|---|---------|---------|---------|
| 1 | `uv run pytest tests/test_data_loader.py -q` | `N passed` | 全部 passed |

**Java 类比**：类似一个 `Iterator<Pair<int[], int[]>>`，窗口大小由 `context_length` 决定，步长由 `stride` 决定。

---

### P1-03 · m03_attention（注意力机制）

**一句话**：每个 token 能「看到」它之前（含自身）的所有 token，但看不到之后的——这叫因果注意力。

| # | 你跑什么 | 你看什么 | 怎么判断 |
|---|---------|---------|---------|
| 1 | `uv run pytest tests/test_attention.py -q` | `N passed` | 全部 passed |

**直觉**：想象你在读一本书，读到第 5 个词时，你只能用前 5 个词的信息来理解它——不能偷看后面。注意力机制就是在模型里实现这个约束。

---

### P1-04 · m04_model（GPT 模型）

**一句话**：把上面三块组装成一个完整的「预测器」——输入一段 token，输出每个位置上「下一个 token 最可能是什么」的打分表。

| # | 你跑什么 | 你看什么 | 怎么判断 |
|---|---------|---------|---------|
| 1 | `uv run pytest tests/test_model_forward.py -q` | `N passed` | 全部 passed |

**Java 类比**：一个 `Function<int[B][T], float[B][T][V]>`。输入 `[batch_size, 序列长度]` 的 token 索引，输出 `[batch_size, 序列长度, 词表大小]` 的分数。

---

### P1-05 · train.py（预训练）

**一句话**：用大量英文文本教模型「猜下一个词」，猜得越准 loss 越低。

| # | 你跑什么 | 你看什么 | 怎么判断 |
|---|---------|---------|---------|
| 1 | 看 `runs/gpt2_small_wikitext103/` 目录下是否有 `checkpoint_best.pt` | 文件存在 | ✅ 说明训练跑完且保存了最佳权重 |
| 2 | 训练日志里找 `val_loss` 最小值 | 数值 | **< 4.0 算合格**；我们实际跑到了约 **3.31**（PPL≈27）。如果 > 5.0 说明模型还没学好 |

**判断 loss 的直觉**：

| val_loss | PPL（≈ e^loss） | 含义 |
|----------|----------------|------|
| > 10 | > 20000 | 模型在瞎猜 |
| 5–6 | 150–400 | 学到了一些，但远不够 |
| 3–4 | 20–55 | **对 WikiText 来说算合格**（我们在这个区间） |
| < 3 | < 20 | 很好（通常需要更多数据或更大模型） |

> PPL（困惑度）= e^loss。可以理解为「模型在每个位置上平均在 PPL 个候选词之间纠结」。PPL=27 ≈ 模型把选项缩小到 27 个里挑一个，比 50257 个词表好太多了。

---

### P1-06 · 训练优化（MPS / cosine / early stop）

| # | 你跑什么 | 你看什么 | 怎么判断 |
|---|---------|---------|---------|
| 1 | 训练日志开头找 `Device:` | `mps`（你的 M3 Max） | 如果是 `cpu` 说明 MPS 没生效，训练会慢很多 |

---

### P2-01 · m05_generate（文本生成）

**一句话**：给模型一句开头，它一个 token 一个 token 地「续写」下去。

| # | 你跑什么 | 你看什么 | 怎么判断 |
|---|---------|---------|---------|
| 1 | `uv run python generate_from_checkpoint.py --checkpoint runs/gpt2_small_wikitext103/checkpoint_best.pt --prompt "The city of"` | 输出的英文续写 | **能读懂、像英文、有维基百科风格** → ✅。如果是乱码或重复 → 有问题 |
| 2 | 同上，改 `--prompt "你好"` | 可能乱码或英文 | **正常**——模型只在英文上训练过，中文开头属于域外输入，不算 bug |

**注意**：这个模型 **不是** ChatGPT。它只会「接着往下写」，不会「回答问题」。如果你输入 "What is the capital of France?"，它可能续写一篇维基风格的段落，但不会直接回答 "Paris"。

---

## Part II — 分类微调

### P2-02 · finetune_classify（SMS ham/spam → 分类 checkpoint）

**REQ 状态**：✅ **P2-02 已完成**（验收仍按下表自检；§10 backlog 为可选后续）。

**一句话**：在预训练模型上面加一个「二选一」的分类头，用短信数据教它区分正常短信（ham）和垃圾短信（spam）。

#### 训练验收

| # | 你跑什么 | 你看什么 | 怎么判断 |
|---|---------|---------|---------|
| 1 | `uv run python finetune_classify.py --config configs/config_classify_spam.json` | 训练日志 | 无报错、正常结束 |
| 2 | 日志末尾找 `Test accuracy` | 百分比 | **≥ 90% 合格**（我们实际约 96%） |
| 3 | 日志末尾找混淆矩阵那几行 | `TN=? FP=? / FN=? TP=?` | **FN 越小越好**（FN = 真垃圾被放过去了） |
| 4 | 日志里找 `Recall_spam`（`R=` 后面的数） | 0 到 1 之间的小数 | **≥ 0.90 合格**；< 0.80 说明漏判严重，需要讨论 |
| 5 | 检查 **`runs/spam_classify_phase_b/checkpoint_best.pt`**（推荐演示） | 文件存在 | ✅（须跑过 `config_classify_spam_phase_b.json`） |
| 6 | 检查 `runs/spam_classify_phase_b/test_false_negative_spam.csv` | 训练末尾导出存在 | ✅；基线目录 `runs/spam_classify/` 仅在做对照时出现 |
| 7 | `uv run python eval_classify.py`（可省略 `--checkpoint`，脚本默认 phase_b） | 混淆矩阵 + PRF | 须已有 `runs/spam_classify_phase_b/checkpoint_best.pt` |

> **两种训练产出**：`config_classify_spam.json` → `runs/spam_classify/`（基线）；**演示默认**用 `config_classify_spam_phase_b.json` → `runs/spam_classify_phase_b/`（见 REPORT §7.3）。

#### 混淆矩阵怎么读（4 个格子）

```
                  模型说 ham    模型说 spam
真的是 ham          TN ✅         FP ⚠️
真的是 spam         FN ❌         TP ✅
```

**你最该盯的是 FN**（左下角）：真垃圾，模型没拦住。类比 Java 里的「漏网异常」——异常发生了但没被 catch 住。

- **FN = 0**：完美，所有垃圾都被抓到了
- **FN = 3–5（总共 150 条 spam 测试）**：约 97% recall，很好
- **FN > 15**：recall < 90%，需要讨论改进

#### Precision / Recall / F1 速记

| 指标 | 问的是什么 | Java 类比 |
|------|----------|----------|
| **Precision** | 模型喊 spam 的里面，有多少真是 spam？ | 类似「命中率」——你 catch 到的异常里，有多少是真的异常而不是误报？ |
| **Recall** | 真实 spam 里，有多大比例被模型抓住？ | 类似「覆盖率」——所有真异常里，你 catch 了多少？ |
| **F1** | Precision 和 Recall 的调和平均 | 两个都要高才行；有一个低了 F1 就会被拉下来 |

**在垃圾短信场景里 Recall 更重要**——漏放一条垃圾（FN）比误杀一条正常短信（FP）后果更严重。

#### 离线复评（不用重新训练）

| # | 你跑什么 | 你看什么 | 怎么判断 |
|---|---------|---------|---------|
| 1 | `uv run python eval_classify.py` 或 `--checkpoint runs/spam_classify_phase_b/checkpoint_best.pt` | 混淆矩阵 + PRF | 与「刚才用 phase_b 训练结束」打印一致；**基线对照**时改用 `runs/spam_classify/checkpoint_best.pt` |

#### 可选：阶段 B 对照（解冻 2 块，REQ §10 BL-P2-02-06）

| # | 你跑什么 | 你看什么 | 怎么判断 |
|---|---------|---------|---------|
| 1 | `uv run python finetune_classify.py --config configs/config_classify_spam_phase_b.json` | 日志末尾 spam **R**、**FN** | 写出 `runs/spam_classify_phase_b/checkpoint_best.pt`，**不覆盖**基线目录 |
| 2 | `uv run python eval_classify.py`（默认已是 phase_b）或显式 `--checkpoint runs/spam_classify_phase_b/checkpoint_best.pt` | 与基线 `spam_classify` 并排对比 | Recall_spam ↑ 且 FN ↓ → 更好抓 spam；若变差 → 讨论 lr / epoch |

---

### P2-03 · classify_sms.py（一行英文短信 → stdout ham / spam）

**一句话**：给它一条英文短信，它告诉你 ham 还是 spam。

| # | 你跑什么 | 你看什么 | 怎么判断 |
|---|---------|---------|---------|
| 1 | `uv run python classify_sms.py --text 'Thanks see you tomorrow'`（可省略 `--checkpoint`，默认 phase_b） | `ham` | ✅ |
| 2 | `uv run python classify_sms.py --text 'URGENT FREE prize call now'` | `ham` 或 `spam` | 预期 spam；判 ham 时对照权重是否为 phase_b |
| 3 | 加 `--probs` 看概率 | `P(ham)=0.xx P(spam)=0.xx` | 两个加起来 ≈ 1.0；如果 spam 的概率 > 0.5 但输出 ham → **那才是 bug** |

**单条推理 vs 整集指标的关系**：整集 96% 准确率 ≠ 每条都对。就像你的 Java 单元测试覆盖率 96% ≠ 没有 bug。单条试错是 **探针**，不能替代 `eval_classify` 的系统评估。

---

### 单元测试（全套）

| # | 你跑什么 | 你看什么 | 怎么判断 |
|---|---------|---------|---------|
| 1 | `uv run pytest -q` | 最后一行 | **全部 passed、0 failed** |

这相当于你 Java 项目里的 `mvn test`——不需要理解每个测试的内部逻辑，**红绿灯足矣**。

---

## 方案 B：当前已完成交付的验收问答

以下是你现在就可以回答的问题。**不是考试**，答不上来的地方恰好是你该花 10 分钟搞懂的点。

### Q1（训练结果）
> 你跑完 `finetune_classify.py` 后，日志里报告 test accuracy = 96%。这个数字是在「哪些数据」上算的？为什么不用训练数据算？

<details>
<summary>参考</summary>

test 集（约 300 条）——训练时模型**从未见过**这些短信。如果用训练数据算，模型可能「背答案」而非「学规律」，得到虚高的准确率。这和你做 Java 单元测试时不能拿训练数据做断言是一个道理——你要用模型没见过的数据来评价它。

</details>

### Q2（混淆矩阵）
> 如果混淆矩阵显示 TN=140, FP=2, FN=12, TP=146，你最该关注哪个数字？为什么？

<details>
<summary>参考</summary>

**FN=12**。意味着 12 条真实垃圾短信被放过去了（模型说是 ham）。在垃圾过滤场景里，漏放垃圾比误杀正常短信更糟。FP=2 只是 2 条正常短信被误标成垃圾，用户可能投诉但风险更小。Recall_spam = 146/(146+12) ≈ 92.4%，还算可以但不算出色。

</details>

### Q3（checkpoint 选择）
> 训练跑了 5 个 epoch，最后一个 epoch 的 val accuracy 是 94%，但 epoch 3 的 val accuracy 是 95%。最终 checkpoint 保存的是哪个？为什么？

<details>
<summary>参考</summary>

**epoch 3 的**。`checkpoint_best.pt` 按**验证集**上的最高 accuracy 保存。最后一 epoch 未必最好（可能开始过拟合了）。这就像你的 CI 构建——你部署的是通过所有测试的那个版本，不一定是最新一次 commit。

</details>

### Q4（单条推理）
> `classify_sms.py --probs` 输出 `P(ham)=0.83 P(spam)=0.17`，最终预测是 `ham`。如果你认为这条短信应该是 spam，这说明什么？

<details>
<summary>参考</summary>

模型以 83% 置信度判错了——这是一个**高置信的错误**（confident wrong），比「50/50 判错」更值得关注。可能原因：训练数据里没见过这种模板、模型容量不够、或微调层数太少。**不是 CLI 的 bug**，是模型能力边界的问题。

</details>

### Q5（eval 脚本）
> `eval_classify.py` 和 `finetune_classify.py` 末尾的指标应该一样吗？什么情况下会不一样？

<details>
<summary>参考</summary>

**在路径一致的前提下一样**：对 **同一个** `checkpoint_best.pt`（同一 `run_name` 目录）+ 同一 `test.csv`、`shuffle=False`，`eval_classify` 与训练末尾打印应一致。若你改了默认 `--checkpoint`（例如切到 phase_b）而训练日志来自另一目录，比较的应是 **各自目录下** 的那份权重，而不是混用。

</details>

---

*本文档随 REQ 增加而更新；新增 REQ 时请同步补充对应章节。*
