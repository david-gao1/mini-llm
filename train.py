#!/usr/bin/env python3
"""预训练入口：损失、优化器、评估、checkpoint。

P1-06 优化：MPS 设备 / 梯度裁剪 / cosine scheduler / early stopping / temperature 采样。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import gc

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fmt_duration(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS 或 MM:SS。"""
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _pick_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup 线性升温 + cosine 衰减。"""
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(
    data_loader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    total = 0.0
    n = len(data_loader)
    if n == 0:
        return float("nan")
    limit = n if num_batches is None else min(num_batches, n)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= limit:
            break
        total += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total / limit


def evaluate_model(model, train_loader, val_loader, device, eval_iter: int):
    model.eval()
    with torch.no_grad():
        tr = calc_loss_loader(train_loader, model, device, eval_iter)
        va = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return tr, va


def main() -> int:
    parser = argparse.ArgumentParser(description="Team mini-LLM training")
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs" / "config.json",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 123))
    _set_seed(seed)

    from mini_llm import m01_tokenizer as tok_mod
    from mini_llm.m01_tokenizer import decode_token_ids, encode_text
    from mini_llm.m02_data_loader import load_text, train_val_dataloaders
    from mini_llm.m04_model import GPTModel
    from mini_llm.m05_generate import generate

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]

    if not tok_mod.vocab_matches_config(int(model_cfg["vocab_size"])):
        print(
            f"Warning: tokenizer vocab {tok_mod.vocab_size()} != config model.vocab_size {model_cfg['vocab_size']}",
            file=sys.stderr,
        )

    run_name = cfg.get("run_name", "run")
    out_root = _ROOT / cfg.get("output_dir", "runs") / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = out_root / "data_cache"

    token_cache_ready = (
        (cache_dir / "train_tokens.pt").is_file()
        and (cache_dir / "val_tokens.pt").is_file()
    )
    if token_cache_ready:
        text = None
    else:
        text = load_text(data_cfg, cache_dir=cache_dir)
        if data_cfg.get("max_chars") is not None:
            text = text[: int(data_cfg["max_chars"])]

    train_loader, val_loader = train_val_dataloaders(
        text,
        float(data_cfg["train_ratio"]),
        model_cfg,
        train_cfg,
        cache_dir=cache_dir,
    )
    del text
    gc.collect()
    if len(train_loader) == 0:
        print("Train DataLoader is empty.", file=sys.stderr)
        return 1
    if len(val_loader) == 0:
        print(
            "Validation DataLoader is empty (text too short after split?). "
            "Lower train_ratio or use more text.",
            file=sys.stderr,
        )
        return 1

    device = _pick_device(str(cfg.get("device", "auto")))
    print(f"Device: {device}")

    model = GPTModel(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    num_epochs = int(train_cfg["num_epochs"])
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
    patience = int(train_cfg.get("patience", 0))

    if grad_accum_steps > 1:
        print(
            f"Gradient accumulation: {grad_accum_steps} micro-steps, "
            f"effective batch_size={int(train_cfg['batch_size']) * grad_accum_steps}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = num_epochs * steps_per_epoch

    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))
    scheduler = _build_scheduler(optimizer, total_steps, warmup_ratio, min_lr_ratio)

    global_step = 0
    eval_freq = int(train_cfg["eval_freq"])
    eval_iter = int(train_cfg["eval_iter"])
    ckpt_every = int(train_cfg["checkpoint_every_steps"])
    start_context = str(train_cfg.get("start_context", "Every effort moves you"))

    best_val_loss = float("inf")
    patience_counter = 0
    t_start = time.time()

    def text_to_ids(s: str) -> torch.Tensor:
        ids = encode_text(s)
        return torch.tensor(ids, device=device).unsqueeze(0)

    def print_sample() -> None:
        model.eval()
        ctx_len = model_cfg["context_length"]
        encoded = text_to_ids(start_context)
        with torch.no_grad():
            out = generate(
                model,
                encoded,
                max_new_tokens=50,
                context_size=ctx_len,
                temperature=0.8,
                top_k=25,
            )
        flat = out.squeeze(0).tolist()
        print(decode_token_ids(flat).replace("\n", " "))
        model.train()

    ckpt_path = out_root / "checkpoint_latest.pt"
    best_path = out_root / "checkpoint_best.pt"

    def _save_checkpoint(path: Path, epoch: int) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": cfg,
            },
            path,
        )

    early_stopped = False

    micro_step = 0

    for epoch in range(num_epochs):
        if early_stopped:
            break
        model.train()
        for input_batch, target_batch in train_loader:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss = loss / grad_accum_steps
            loss.backward()
            micro_step += 1

            if micro_step % grad_accum_steps != 0:
                continue

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % eval_freq == 0:
                tr, va = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                lr_now = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_start
                steps_per_sec = global_step / elapsed if elapsed > 0 else 0
                remaining_steps = total_steps - global_step
                eta_sec = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                elapsed_str = _fmt_duration(elapsed)
                eta_str = _fmt_duration(eta_sec)
                now_str = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{now_str}] "
                    f"Epoch {epoch + 1}  Step {global_step:06d}  "
                    f"train_loss={tr:.4f}  val_loss={va:.4f}  "
                    f"lr={lr_now:.6f}  "
                    f"[{elapsed_str}<{eta_str}, {steps_per_sec:.1f} step/s]"
                )

                if va < best_val_loss:
                    best_val_loss = va
                    patience_counter = 0
                    _save_checkpoint(best_path, epoch)
                    print(f"  -> New best val_loss={va:.4f}, saved -> {best_path}")
                else:
                    patience_counter += 1
                    if patience > 0 and patience_counter >= patience:
                        print(
                            f"  -> Early stopping: val_loss not improved for "
                            f"{patience} evaluations (best={best_val_loss:.4f})"
                        )
                        early_stopped = True
                        break

            if global_step % ckpt_every == 0:
                _save_checkpoint(ckpt_path, epoch)
                print(f"Saved checkpoint -> {ckpt_path}")

        print_sample()

    _save_checkpoint(ckpt_path, num_epochs if not early_stopped else epoch)
    total_time = time.time() - t_start
    print(
        f"Done. {global_step} steps in {_fmt_duration(total_time)} "
        f"({global_step / total_time:.1f} step/s)"
    )
    print(f"Final checkpoint -> {ckpt_path}")
    if best_path.exists():
        print(f"Best checkpoint (val_loss={best_val_loss:.4f}) -> {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
