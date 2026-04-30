#!/usr/bin/env python3
"""分类微调入口：加载预训练 checkpoint → 改造模型 → SMS Spam 二分类训练。

用法
----
uv run python finetune_classify.py --config configs/config_classify_spam.json

可选覆盖预训练 checkpoint 路径：
uv run python finetune_classify.py \\
    --config configs/config_classify_spam.json \\
    --checkpoint runs/gpt2_small_wikitext103/checkpoint_best.pt
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Finetune GPT for SMS Spam classification")
    parser.add_argument("--config", type=Path, default=_ROOT / "configs" / "config_classify_spam.json")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override pretrained_checkpoint in config")
    args = parser.parse_args()

    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 123))
    _set_seed(seed)

    from mini_llm.m04_model import GPTModel
    from mini_llm.m06_classify_finetune import (
        SpamDataset,
        calc_accuracy_loader,
        collect_predictions_loader,
        confusion_counts_binary_spam,
        download_and_prepare_spam,
        evaluate_model,
        export_false_negative_spam_csv,
        format_classification_eval_lines,
        prf1_from_counts,
    )

    ft_cfg = cfg["finetune"]
    data_cfg = cfg["data"]

    # ---- 1. Data ----
    data_dir = _ROOT / data_cfg["data_dir"]
    train_csv, val_csv, test_csv = download_and_prepare_spam(data_dir)

    train_dataset = SpamDataset(train_csv, max_length=ft_cfg.get("max_length"))
    val_dataset = SpamDataset(val_csv, max_length=train_dataset.max_length)
    test_dataset = SpamDataset(test_csv, max_length=train_dataset.max_length)
    print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    print(f"Sequence max_length: {train_dataset.max_length}")

    batch_size = int(ft_cfg["batch_size"])

    torch.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

    # ---- 2. Load pretrained model ----
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

    assert train_dataset.max_length <= model_cfg["context_length"], (
        f"Dataset max_length ({train_dataset.max_length}) exceeds model context_length "
        f"({model_cfg['context_length']}). Set finetune.max_length <= {model_cfg['context_length']}."
    )

    device = _pick_device(str(cfg.get("device", "auto")))
    print(f"Device: {device}")

    model = GPTModel(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded pretrained weights (val_loss={ckpt.get('best_val_loss', '?')})")

    # ---- 3. Modify model for classification ----
    for param in model.parameters():
        param.requires_grad = False

    num_classes = int(ft_cfg["num_classes"])
    torch.manual_seed(seed)
    model.out_head = nn.Linear(model_cfg["emb_dim"], num_classes)

    unfreeze_n = int(ft_cfg.get("unfreeze_last_n_blocks", 1))
    for block in model.trf_blocks[-unfreeze_n:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} total, {trainable:,} trainable ({trainable / total * 100:.1f}%)")

    # ---- 4. Train ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(ft_cfg["learning_rate"]),
        weight_decay=float(ft_cfg["weight_decay"]),
    )

    num_epochs = int(ft_cfg["num_epochs"])
    eval_freq = int(ft_cfg["eval_freq"])
    eval_iter = int(ft_cfg["eval_iter"])

    run_name = cfg.get("run_name", "finetune_run")
    out_root = _ROOT / cfg.get("output_dir", "runs") / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    best_path = out_root / "checkpoint_best.pt"

    best_val_acc = 0.0
    global_step = -1
    t_start = time.time()

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    print(f"\n{'='*60}")
    print(f"Starting classification finetuning: {num_epochs} epochs, "
          f"{len(train_loader)} batches/epoch, eval every {eval_freq} steps")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(input_batch)[:, -1, :]
            loss = nn.functional.cross_entropy(logits, target_batch)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % eval_freq == 0:
                tr_loss, va_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(tr_loss)
                val_losses.append(va_loss)
                elapsed = _fmt_duration(time.time() - t_start)
                print(
                    f"Ep {epoch + 1} (Step {global_step:04d}) [{elapsed}]: "
                    f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}"
                )

        tr_acc = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        va_acc = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        print(
            f"  Epoch {epoch + 1} accuracy: "
            f"train={tr_acc * 100:.2f}%  val={va_acc * 100:.2f}%"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": ckpt["config"],
                    "finetune_config": cfg,
                    "num_classes": num_classes,
                    "spam_max_length": train_dataset.max_length,
                    "pad_token_id": 50256,
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_accuracy": best_val_acc,
                },
                best_path,
            )
            print(f"  -> New best val_accuracy={va_acc * 100:.2f}%, saved -> {best_path}")

    # ---- 5. Test evaluation (best val checkpoint weights, not last epoch) ----
    print(f"\n{'='*60}")
    print("Final evaluation on test set (best val checkpoint)")
    print(f"{'='*60}")

    if best_path.is_file():
        try:
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        except TypeError:
            best_ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(best_ckpt["model_state_dict"])
        print(f"Loaded weights from best checkpoint for reporting -> {best_path}")
    else:
        print("Warning: no best checkpoint on disk; reporting metrics on last epoch weights.")

    test_preds, test_targets = collect_predictions_loader(test_loader, model, device)
    tn, fp, fn, tp = confusion_counts_binary_spam(test_targets, test_preds)
    metrics = prf1_from_counts(tn, fp, fn, tp)
    for line in format_classification_eval_lines(tn, fp, fn, tp, metrics, split_name="test"):
        print(line)

    fn_csv = out_root / "test_false_negative_spam.csv"
    n_fn = export_false_negative_spam_csv(
        test_dataset, test_targets.cpu(), test_preds.cpu(), fn_csv, max_rows=None
    )
    print(f"Exported {n_fn} spam→ham (FN) rows -> {fn_csv}")

    test_acc = calc_accuracy_loader(test_loader, model, device)
    total_time = time.time() - t_start
    print(f"Test accuracy (full-loader): {test_acc * 100:.2f}%")
    print(f"Training completed in {_fmt_duration(total_time)}.")
    if best_path.exists():
        print(f"Best checkpoint (val_accuracy={best_val_acc * 100:.2f}%) -> {best_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
