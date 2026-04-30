#!/usr/bin/env python3
"""在已有分类 checkpoint 上评估 test 集：混淆矩阵、spam/ham PRF1、导出 FN spam CSV。

默认读取 checkpoint 内 ``finetune_config`` 的 ``data.data_dir``，使用 ``test.csv``。

用法
----
uv run python eval_classify.py --checkpoint runs/spam_classify/checkpoint_best.pt

指定测试集与 FN 导出路径::

  uv run python eval_classify.py \\
    --checkpoint runs/spam_classify/checkpoint_best.pt \\
    --test-csv data_cache/sms_spam/test.csv \\
    --fn-out runs/spam_classify/eval_fn.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _pick_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SMS spam classifier checkpoint on test CSV")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_ROOT / "runs" / "spam_classify" / "checkpoint_best.pt",
        help="Classification finetune checkpoint",
    )
    parser.add_argument("--test-csv", type=Path, default=None, help="Default: <data_dir>/test.csv from checkpoint meta")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size (training batch_size need not match)",
    )
    parser.add_argument(
        "--fn-out",
        type=Path,
        default=None,
        help="Write spam→ham FN rows to this CSV (default: <checkpoint_parent>/eval_false_negative_spam.csv)",
    )
    parser.add_argument(
        "--max-fn-rows",
        type=int,
        default=None,
        help="Cap rows written to FN CSV (default: no cap)",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    from mini_llm.m06_classify_finetune import (
        SpamDataset,
        collect_predictions_loader,
        confusion_counts_binary_spam,
        export_false_negative_spam_csv,
        format_classification_eval_lines,
        load_spam_classifier_checkpoint,
        prf1_from_counts,
    )

    device = _pick_device(args.device)
    model, meta = load_spam_classifier_checkpoint(args.checkpoint, device)

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    ft_cfg = ckpt.get("finetune_config") or {}
    data_cfg = ft_cfg.get("data") or {}
    data_dir = _ROOT / Path(data_cfg.get("data_dir", "data_cache/sms_spam"))

    test_csv = args.test_csv or (data_dir / "test.csv")
    if not test_csv.is_file():
        print(f"Test CSV not found: {test_csv}", file=sys.stderr)
        return 1

    sml = meta.get("spam_max_length")
    model_cfg = model.cfg  # type: ignore[attr-defined]
    if sml is None:
        sml = int(model_cfg["context_length"])
        print(
            "Warning: checkpoint has no spam_max_length; using context_length.",
            file=sys.stderr,
        )
    max_len = int(sml)
    if max_len > int(model_cfg["context_length"]):
        max_len = int(model_cfg["context_length"])

    test_dataset = SpamDataset(test_csv, max_length=max_len)
    if test_dataset.max_length != max_len:
        print(
            f"Note: dataset max_length={test_dataset.max_length} (from file); "
            f"checkpoint indicated {max_len}.",
            file=sys.stderr,
        )

    loader = DataLoader(test_dataset, batch_size=int(args.batch_size), shuffle=False, drop_last=False)
    preds, targets = collect_predictions_loader(loader, model, device)
    tn, fp, fn, tp = confusion_counts_binary_spam(targets, preds)
    m = prf1_from_counts(tn, fp, fn, tp)
    for line in format_classification_eval_lines(tn, fp, fn, tp, m, split_name="test"):
        print(line)

    fn_out = args.fn_out
    if fn_out is None:
        fn_out = args.checkpoint.resolve().parent / "eval_false_negative_spam.csv"
    n_fn = export_false_negative_spam_csv(
        test_dataset,
        targets.cpu(),
        preds.cpu(),
        fn_out,
        max_rows=args.max_fn_rows,
    )
    print(f"Exported {n_fn} spam→ham (FN) rows -> {fn_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
