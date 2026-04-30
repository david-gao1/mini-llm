#!/usr/bin/env python3
"""加载 SMS 分类微调 checkpoint，对一条英文短信输出 ham / spam。

依赖 ``finetune_classify.py`` 写出的 ``checkpoint_best.pt``（内含 ``spam_max_length`` 等字段）。

用法
----
uv run python classify_sms.py \\
  --checkpoint runs/spam_classify/checkpoint_best.pt \\
  --text "WINNER!! As a valued network customer you have been selected..."

从标准输入读一行（无 ``--text`` 时）::

  echo "See you at 8pm" | uv run python classify_sms.py --checkpoint runs/spam_classify/checkpoint_best.pt

可选 ``--probs`` 打印 softmax 概率；``--max-length`` 覆盖 checkpoint 中的序列长度（旧 checkpoint 未含长度时需指定）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

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


def _resolve_max_length(meta: dict, model_cfg: dict, override: int | None) -> tuple[int, bool]:
    """Returns (max_length, used_fallback_to_context)."""
    if override is not None:
        return int(override), False
    sml = meta.get("spam_max_length")
    if sml is not None:
        return int(sml), False
    ft = meta.get("finetune_config") or {}
    if isinstance(ft, dict):
        inner = ft.get("finetune") or {}
        if inner.get("max_length") is not None:
            return int(inner["max_length"]), False
    return int(model_cfg["context_length"]), True


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify one SMS as ham or spam")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_ROOT / "runs" / "spam_classify" / "checkpoint_best.pt",
        help="Path from finetune_classify.py (default: runs/spam_classify/checkpoint_best.pt)",
    )
    parser.add_argument("--text", type=str, default=None, help="SMS text (English); omit to read one line from stdin")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--max-length", type=int, default=None, dest="max_length", help="Override spam_max_length")
    parser.add_argument("--probs", action="store_true", help="Print softmax probabilities for each class")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    text = args.text
    if text is None:
        text = sys.stdin.readline()
        if not text:
            print("No input text (use --text or pipe one line to stdin).", file=sys.stderr)
            return 1
        text = text.rstrip("\n\r")

    from mini_llm.m06_classify_finetune import encode_spam_text_for_model, load_spam_classifier_checkpoint

    device = _pick_device(args.device)
    model, meta = load_spam_classifier_checkpoint(args.checkpoint, device)
    model_cfg = model.cfg  # type: ignore[attr-defined]

    max_len, fallback_ctx = _resolve_max_length(meta, model_cfg, args.max_length)
    if max_len > int(model_cfg["context_length"]):
        print(
            f"max_length={max_len} > context_length={model_cfg['context_length']}; clamping.",
            file=sys.stderr,
        )
        max_len = int(model_cfg["context_length"])
    if fallback_ctx:
        print(
            "Warning: checkpoint has no spam_max_length; using model context_length — "
            "re-run finetune_classify.py to embed length, or pass --max-length.",
            file=sys.stderr,
        )

    pad_id = int(meta["pad_token_id"])
    batch = encode_spam_text_for_model(text, max_len, pad_token_id=pad_id).to(device)

    with torch.no_grad():
        logits = model(batch)[:, -1, :]
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred = int(torch.argmax(probs, dim=-1).item())

    names = {0: "ham", 1: "spam"}
    label = names.get(pred, str(pred))
    print(label)
    if args.probs:
        nc = int(meta["num_classes"])
        parts = [f"P({names.get(i, i)})={probs[i].item():.4f}" for i in range(nc)]
        print(" ".join(parts), file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
