#!/usr/bin/env python3
"""指令微调（SFT）入口：预训练 LM checkpoint + 书本格式 JSON → 训练 → checkpoint_best.pt。

用法
----
uv run python finetune_instruction.py --config configs/config_instruction_small.json

覆盖权重路径：
uv run python finetune_instruction.py --config ... --checkpoint runs/gpt2_small_wikitext103/checkpoint_best.pt

只评估验证集 CE（抽样 + 全量），不写盘：
uv run python finetune_instruction.py --config configs/config_instruction_small.json \\
  --eval-val-only --eval-checkpoint runs/instruction_sft_small/checkpoint_best.pt

亦可用薄封装 ``eval_instruction_loss.py``（同上逻辑）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
# 脚本在仓库根目录，包在 src/mini_llm；直接 python 运行时需要把 src 放进 path。
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def calc_loss_batch_instruction(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    ignore_index: int,
) -> torch.Tensor:
    # 与预训练相同：每个位置预测「下一个 token」。targets 里 pad 位已标成 ignore_index，不参与 loss。
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
        ignore_index=ignore_index,
    )


def calc_loss_loader_instruction(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None,
    ignore_index: int,
) -> float:
    # 估 loss 用：num_batches 非空时只扫前若干个 batch，加快中途打印；完整 epoch 摘要仍用同一套评估逻辑。
    total = 0.0
    n = len(data_loader)
    if n == 0:
        return float("nan")
    limit = n if num_batches is None else min(num_batches, n)
    model.eval()
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= limit:
                break
            total += calc_loss_batch_instruction(
                input_batch, target_batch, model, device, ignore_index
            ).item()
    model.train()
    return total / limit


def _resolve_eval_batch_limits(i_cfg: dict) -> tuple[int | None, int | None]:
    """返回 (train_batches, val_batches)：None 表示扫完整 DataLoader。

    - ``eval_iter`` 为 JSON ``null``：train/val 在按步评估时均扫全集（慢；适合小数据）。
    - ``eval_iter`` 为 int：train 使用该上限；val 默认相同，可被 ``eval_val_batches`` 覆盖
      （``null`` = 全 val；int = 前若干个 batch）。
    """
    raw_iter = i_cfg.get("eval_iter")
    if raw_iter is None:
        return None, None
    train_cap = int(raw_iter)
    if "eval_val_batches" not in i_cfg:
        return train_cap, train_cap
    raw_val = i_cfg["eval_val_batches"]
    if raw_val is None:
        return train_cap, None
    return train_cap, int(raw_val)


def _save_instruction_best(
    best_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt: dict,
    cfg: dict,
    global_step: int,
    epoch: int,
    best_val_loss: float,
    instruction_meta: dict,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": ckpt["config"],
            "instruction_finetune_config": cfg,
            "global_step": global_step,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "instruction_meta": instruction_meta,
        },
        best_path,
    )


def run_eval_val_only(config_path: Path, eval_ckpt_path: Path) -> int:
    """加载 SFT（或预训练）checkpoint，在同一套 val DataLoader 上打印抽样与全量 val loss。"""
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    if not eval_ckpt_path.is_file():
        print(f"Checkpoint not found: {eval_ckpt_path}", file=sys.stderr)
        return 1

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 123))
    _set_seed(seed)

    from mini_llm.m01_tokenizer import encode_text
    from mini_llm.m04_model import GPTModel
    from mini_llm.m07_instruction_finetune import (
        IGNORE_INDEX_DEFAULT,
        PAD_TOKEN_ID_DEFAULT,
        InstructionDataset,
        download_instruction_json,
        make_instruction_collate_fn,
        split_instruction_entries,
    )

    data_cfg = cfg["data"]
    i_cfg = cfg["instruction_finetune"]
    cache_path = _ROOT / Path(data_cfg["cache_path"])
    url = str(data_cfg["url"])
    entries = download_instruction_json(cache_path, url)
    train_ratio = float(data_cfg.get("train_ratio", 0.85))
    test_ratio = float(data_cfg.get("test_ratio", 0.1))
    train_data, val_data, test_data = split_instruction_entries(
        entries, train_ratio=train_ratio, test_ratio=test_ratio
    )
    smoke_trim = i_cfg.get("smoke_trim")
    if smoke_trim is not None:
        k = int(smoke_trim)
        train_data = train_data[:k]
        val_data = val_data[:k]
        test_data = test_data[:k]

    device = _pick_device(str(cfg.get("device", "auto")))
    allowed_max_length = i_cfg.get("allowed_max_length")
    if allowed_max_length is not None:
        allowed_max_length = int(allowed_max_length)
    pad_token_id = int(i_cfg.get("pad_token_id", PAD_TOKEN_ID_DEFAULT))
    ignore_index = int(i_cfg.get("ignore_index", IGNORE_INDEX_DEFAULT))

    collate = make_instruction_collate_fn(
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        allowed_max_length=allowed_max_length,
        device=device,
    )
    torch.manual_seed(seed)
    val_ds = InstructionDataset(val_data, encode_text)
    batch_size = int(i_cfg["batch_size"])
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
        num_workers=0,
    )

    try:
        ckpt_eval = torch.load(eval_ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt_eval = torch.load(eval_ckpt_path, map_location="cpu")

    model_cfg = ckpt_eval["config"]["model"]
    ctx = int(model_cfg["context_length"])
    if allowed_max_length is not None and allowed_max_length > ctx:
        allowed_max_length = ctx
        collate = make_instruction_collate_fn(
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
            allowed_max_length=allowed_max_length,
            device=device,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate,
            num_workers=0,
        )

    model = GPTModel(model_cfg)
    model.load_state_dict(ckpt_eval["model_state_dict"])
    model.to(device)

    train_batches, val_batches = _resolve_eval_batch_limits(i_cfg)
    sampled = calc_loss_loader_instruction(
        val_loader, model, device, val_batches, ignore_index
    )
    full = calc_loss_loader_instruction(val_loader, model, device, None, ignore_index)
    print(f"val_loss_sampled={sampled:.6f}")
    print(f"val_loss_full={full:.6f}")
    return 0


def main() -> int:
    # --- 参数与配置 ---
    parser = argparse.ArgumentParser(description="Instruction finetuning (SFT) for GPTModel")
    parser.add_argument("--config", type=Path, default=_ROOT / "configs" / "config_instruction_small.json")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override pretrained_checkpoint in config")
    parser.add_argument(
        "--eval-val-only",
        action="store_true",
        help="只评估验证集 CE（抽样 + 全量），不写权重；需配合 --eval-checkpoint",
    )
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help="与 --eval-val-only 连用：要打分的 .pt（通常为 SFT checkpoint_best.pt）",
    )
    args = parser.parse_args()

    if args.eval_val_only:
        if args.eval_checkpoint is None or not args.eval_checkpoint.is_file():
            print(
                "--eval-val-only 需要现有文件路径：--eval-checkpoint runs/.../checkpoint_best.pt",
                file=sys.stderr,
            )
            return 1
        return run_eval_val_only(args.config, args.eval_checkpoint)

    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 123))
    _set_seed(seed)

    from mini_llm.m01_tokenizer import encode_text
    from mini_llm.m04_model import GPTModel
    from mini_llm.m07_instruction_finetune import (
        IGNORE_INDEX_DEFAULT,
        PAD_TOKEN_ID_DEFAULT,
        InstructionDataset,
        download_instruction_json,
        make_instruction_collate_fn,
        split_instruction_entries,
    )

    data_cfg = cfg["data"]
    i_cfg = cfg["instruction_finetune"]

    # --- 数据：JSON → 划分 →（可选）截断；详见 m07 format_input / InstructionDataset ---
    cache_path = _ROOT / Path(data_cfg["cache_path"])
    url = str(data_cfg["url"])
    entries = download_instruction_json(cache_path, url)

    train_ratio = float(data_cfg.get("train_ratio", 0.85))
    test_ratio = float(data_cfg.get("test_ratio", 0.1))
    # 时间顺序是 [train | 书中 10% test 中段 | val 尾段]，但返回为 (train, val, test)：第二个是尾段、第三个是中段。
    train_data, val_data, test_data = split_instruction_entries(
        entries, train_ratio=train_ratio, test_ratio=test_ratio
    )

    # 各划分只保留前 k 条，便于冒烟；正式训设 null。
    smoke_trim = i_cfg.get("smoke_trim")
    if smoke_trim is not None:
        k = int(smoke_trim)
        train_data = train_data[:k]
        val_data = val_data[:k]
        test_data = test_data[:k]

    print(f"Dataset sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # --- 设备、collate（pad + ignore_index）、Dataset、DataLoader（train shuffle / val 否）---
    device = _pick_device(str(cfg.get("device", "auto")))
    print(f"Device: {device}")

    allowed_max_length = i_cfg.get("allowed_max_length")
    if allowed_max_length is not None:
        allowed_max_length = int(allowed_max_length)

    pad_token_id = int(i_cfg.get("pad_token_id", PAD_TOKEN_ID_DEFAULT))
    ignore_index = int(i_cfg.get("ignore_index", IGNORE_INDEX_DEFAULT))

    # Dataset 产出变长 token 列表；collate 负责 pad + 构造错位一位的 targets（见 m07）。
    collate = make_instruction_collate_fn(
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        allowed_max_length=allowed_max_length,
        device=device,
    )

    torch.manual_seed(seed)
    train_ds = InstructionDataset(train_data, encode_text)
    val_ds = InstructionDataset(val_data, encode_text)

    batch_size = int(i_cfg["batch_size"])
    drop_last_train = bool(i_cfg.get("drop_last_train", True))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
        num_workers=0,
    )

    if len(train_loader) == 0:
        print("Train DataLoader is empty (increase smoke_trim or batch_size?).", file=sys.stderr)
        return 1

    # --- 预训练权重：结构来自 ckpt["config"]，权重进 GPTModel；allowed_max_length 不得超过 context ---
    ckpt_path = args.checkpoint or (_ROOT / cfg["pretrained_checkpoint"])
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1
    print(f"Loading pretrained checkpoint: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    model_cfg = ckpt["config"]["model"]
    ctx = int(model_cfg["context_length"])
    # 序列长度不能超过模型 context；收紧后要重做 collate 与 Loader，否则仍按旧截断。
    if allowed_max_length is not None and allowed_max_length > ctx:
        print(
            f"allowed_max_length ({allowed_max_length}) > model context_length ({ctx}); clamping.",
            file=sys.stderr,
        )
        allowed_max_length = ctx
        collate = make_instruction_collate_fn(
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
            allowed_max_length=allowed_max_length,
            device=device,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last_train,
            collate_fn=collate,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate,
            num_workers=0,
        )

    # --- 按 ckpt 构建 GPTModel 并载入预训练权重（全词表 LM 头，非分类头）---
    model = GPTModel(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded pretrained weights (best_val_loss={ckpt.get('best_val_loss', 'n/a')})")

    # 全量微调（非第六章那种只换分类头）；与书 main 里对部分参数解冻的策略不同，见 REQ-P3-01SUB。
    for p in model.parameters():
        p.requires_grad = True

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters (full finetune): {n_params:,} ({n_params / 1e6:.1f}M)")

    # --- 优化器、训练轮数、eval 频率；runs/<run_name>/checkpoint_best.pt ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(i_cfg["learning_rate"]),
        weight_decay=float(i_cfg["weight_decay"]),
    )

    num_epochs = int(i_cfg["num_epochs"])
    eval_freq = int(i_cfg["eval_freq"])
    eval_train_batches, eval_val_batches = _resolve_eval_batch_limits(i_cfg)
    epoch_val_full = bool(i_cfg.get("epoch_val_full", True))
    grad_clip = float(i_cfg.get("grad_clip", 1.0))

    run_name = cfg.get("run_name", "instruction_sft")
    out_root = _ROOT / cfg.get("output_dir", "runs") / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    best_path = out_root / "checkpoint_best.pt"

    # 写入 checkpoint，推理/复现时知道 pad 与模板版本，避免和训练假设不一致。
    instruction_meta = {
        "allowed_max_length": allowed_max_length,
        "pad_token_id": pad_token_id,
        "ignore_index": ignore_index,
        "template": "ch07_book_format_input",
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
    }

    best_val_loss = float("inf")
    global_step = -1  # 先 -1，进入第一个 batch 后为 0，便于按「第几步」做 eval_freq
    t_start = time.time()

    def _fmt_batches(cap: int | None) -> str:
        return "full" if cap is None else str(cap)

    print(f"\n{'='*60}")
    print(
        f"Instruction SFT: {num_epochs} epochs, batch={batch_size}, "
        f"eval every {eval_freq} steps "
        f"(train_batches={_fmt_batches(eval_train_batches)} "
        f"val_batches={_fmt_batches(eval_val_batches)}); "
        f"epoch_val_full={epoch_val_full}"
    )
    print(f"{'='*60}\n")

    # --- 训练：每步 CE loss；每 eval_freq 步抽样算 train/val loss，val 改善则写入 best ---
    model.train()
    for epoch in range(num_epochs):
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = calc_loss_batch_instruction(
                input_batch, target_batch, model, device, ignore_index
            )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            global_step += 1

            # 每隔 eval_freq 步在 train/val 上算平均 loss（batch 上限见 eval_train_batches / eval_val_batches）。
            if global_step % eval_freq == 0:
                tr = calc_loss_loader_instruction(
                    train_loader, model, device, eval_train_batches, ignore_index
                )
                va = calc_loss_loader_instruction(
                    val_loader, model, device, eval_val_batches, ignore_index
                )
                elapsed = _fmt_duration(time.time() - t_start)
                print(
                    f"Ep {epoch + 1} Step {global_step:04d} [{elapsed}] "
                    f"train_loss={tr:.4f} val_loss_sampled={va:.4f}"
                )

                # va != va 检测 NaN；无效 loss 不写 best。
                if va < best_val_loss and not (va != va):  # not nan
                    best_val_loss = va
                    _save_instruction_best(
                        best_path,
                        model,
                        optimizer,
                        ckpt,
                        cfg,
                        global_step,
                        epoch,
                        best_val_loss,
                        instruction_meta,
                    )
                    print(f"  -> New best val_loss_sampled={best_val_loss:.4f}, saved -> {best_path}")

        ep_tr = calc_loss_loader_instruction(
            train_loader, model, device, eval_train_batches, ignore_index
        )
        ep_va_s = calc_loss_loader_instruction(
            val_loader, model, device, eval_val_batches, ignore_index
        )
        ep_va_full = calc_loss_loader_instruction(
            val_loader, model, device, None, ignore_index
        )
        print(
            f"  End epoch {epoch + 1}: train_loss={ep_tr:.4f} "
            f"val_loss_sampled={ep_va_s:.4f} val_loss_full={ep_va_full:.4f}"
        )
        if epoch_val_full and not (ep_va_full != ep_va_full) and ep_va_full < best_val_loss:
            best_val_loss = ep_va_full
            _save_instruction_best(
                best_path,
                model,
                optimizer,
                ckpt,
                cfg,
                global_step,
                epoch,
                best_val_loss,
                instruction_meta,
            )
            print(
                f"  -> New best val_loss_full={best_val_loss:.4f} (epoch end), saved -> {best_path}"
            )

    # --- 收尾：若从未因 val 存过盘，则把最后一轮权重写入 checkpoint_best.pt ---
    total_time = time.time() - t_start
    print(f"\nTraining finished in {_fmt_duration(total_time)}.")
    # 若 eval 从未触发存盘（例如 eval_freq 很大且步数少），至少落盘最后一版权重。
    if best_path.is_file():
        print(f"Best checkpoint -> {best_path} (best_val_loss={best_val_loss:.4f})")
    else:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": ckpt["config"],
                "instruction_finetune_config": cfg,
                "global_step": global_step,
                "epoch": num_epochs - 1,
                "best_val_loss": None,
                "instruction_meta": instruction_meta,
            },
            best_path,
        )
        print(f"No eval-triggered best; saved last weights -> {best_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
