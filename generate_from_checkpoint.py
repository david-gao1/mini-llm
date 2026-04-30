#!/usr/bin/env python3
"""加载训练保存的 checkpoint，做文本生成（检验效果）。

Checkpoint 由 train.py 写入，内含完整 config；本脚本据此重建 GPTModel 并加载权重。

示例::

    uv run python generate_from_checkpoint.py \\
      --checkpoint runs/gpt2_small_wikitext103/checkpoint_best.pt \\
      --prompt "The history of"

    # 贪心（temperature=0），结果固定
    uv run python generate_from_checkpoint.py \\
      --checkpoint runs/gpt2_small_wikitext103/checkpoint_best.pt \\
      --prompt "Hello" --temperature 0
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


def _prompt_is_out_of_training_domain(s: str) -> bool:
    """WikiText-103 为英文维基；含中日韩等字符时与训练分布不一致。"""
    for ch in s:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF:  # CJK Unified Ideographs
            return True
        if 0x3040 <= o <= 0x30FF:  # Hiragana / Katakana
            return True
        if 0xAC00 <= o <= 0xD7AF:  # Hangul syllables
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text from a train.py checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="e.g. runs/gpt2_small_wikitext103/checkpoint_best.pt",
    )
    parser.add_argument("--prompt", type=str, default="The history of", help="起始英文文本")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    from mini_llm.m01_tokenizer import decode_token_ids, encode_text
    from mini_llm.m04_model import GPTModel
    from mini_llm.m05_generate import generate

    if _prompt_is_out_of_training_domain(args.prompt):
        print(
            "提示：本 checkpoint 在英文 WikiText-103 上训练；中文/日文/韩文开头不在训练分布内，"
            "续写会像「乱接英文」且易出现解码异常字符。请改用英文 prompt 检验效果，例如 "
            '"The history of" 或 "In the early 20th century,"。',
            file=sys.stderr,
        )

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model_cfg = cfg["model"]

    device = _pick_device(args.device)
    model = GPTModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ids = encode_text(args.prompt)
    idx = torch.tensor(ids, device=device).unsqueeze(0)
    ctx_len = int(model_cfg["context_length"])

    with torch.no_grad():
        out = generate(
            model,
            idx,
            max_new_tokens=args.max_new_tokens,
            context_size=ctx_len,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
        )

    text = decode_token_ids(out.squeeze(0).tolist()).replace("\n", " ")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
