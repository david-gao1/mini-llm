#!/usr/bin/env python3
"""预训练入口：损失、优化器、评估、checkpoint。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

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


def _pick_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


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
    from mini_llm.m02_data_loader import load_text, train_val_dataloaders
    from mini_llm.m05_generate import generate_text_simple
    from mini_llm.m04_model import GPTModel

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]

    if tok_mod.vocab_size() != model_cfg["vocab_size"]:
        print(
            f"Warning: tokenizer vocab {tok_mod.vocab_size()} != config model.vocab_size {model_cfg['vocab_size']}",
            file=sys.stderr,
        )

    run_name = cfg.get("run_name", "run")
    out_root = _ROOT / cfg.get("output_dir", "runs") / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = out_root / "data_cache"

    text = load_text(data_cfg, cache_dir=cache_dir)
    if data_cfg.get("max_chars") is not None:
        text = text[: int(data_cfg["max_chars"])]

    train_loader, val_loader = train_val_dataloaders(
        text,
        float(data_cfg["train_ratio"]),
        model_cfg,
        train_cfg,
    )
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
    model = GPTModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    global_step = 0

    enc = tok_mod.get_encoding()
    eval_freq = int(train_cfg["eval_freq"])
    eval_iter = int(train_cfg["eval_iter"])
    ckpt_every = int(train_cfg["checkpoint_every_steps"])
    num_epochs = int(train_cfg["num_epochs"])
    start_context = str(train_cfg.get("start_context", "Every effort moves you"))

    def text_to_ids(s: str) -> torch.Tensor:
        ids = enc.encode(s)
        return torch.tensor(ids, device=device).unsqueeze(0)

    def print_sample() -> None:
        model.eval()
        ctx_len = model_cfg["context_length"]
        encoded = text_to_ids(start_context)
        with torch.no_grad():
            out = generate_text_simple(
                model,
                encoded,
                max_new_tokens=50,
                context_size=ctx_len,
            )
        flat = out.squeeze(0).tolist()
        print(enc.decode(flat).replace("\n", " "))
        model.train()

    ckpt_path = out_root / "checkpoint_latest.pt"

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % eval_freq == 0:
                tr, va = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                print(
                    f"Epoch {epoch + 1}  Step {global_step:06d}  "
                    f"train_loss={tr:.4f}  val_loss={va:.4f}"
                )

            if global_step % ckpt_every == 0:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "config": cfg,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint -> {ckpt_path}")

        print_sample()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "epoch": num_epochs,
            "config": cfg,
        },
        ckpt_path,
    )
    print(f"Done. Final checkpoint -> {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
