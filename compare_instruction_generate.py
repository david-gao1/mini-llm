#!/usr/bin/env python3
"""对同一批英文 instruction prompt，分别用两个 checkpoint 调用 generate_from_checkpoint.py， stdout 并列输出（Markdown）。

用法（在 team-mini-llm 根目录）::

    uv run python compare_instruction_generate.py \\
      --pretrained runs/gpt2_small_wikitext103/checkpoint_best.pt \\
      --sft runs/instruction_train_small/checkpoint_best.pt \\
      --prompt-file docs/prompts/instruction_compare_sample.txt

prompt 文件：多个 prompt 用单独一行 ``---`` 分隔；以 ``#`` 开头的行视为注释跳过。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _split_prompts(text: str) -> list[str]:
    blocks: list[str] = []
    for raw in text.split("\n---\n"):
        lines = [
            ln
            for ln in raw.splitlines()
            if not (ln.lstrip().startswith("#") and not ln.lstrip().startswith("###"))
        ]
        block = "\n".join(lines).strip()
        if block:
            blocks.append(block)
    return blocks


def _run_generate(
    checkpoint: Path,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> str:
    cmd = [
        sys.executable,
        str(_ROOT / "generate_from_checkpoint.py"),
        "--checkpoint",
        str(checkpoint),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top-k",
        str(top_k),
        "--device",
        device,
    ]
    r = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout or "generate failed")
    return r.stdout.rstrip()


def main() -> int:
    p = argparse.ArgumentParser(description="Compare instruction-style generation: pretrained vs SFT")
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--sft", type=Path, required=True)
    p.add_argument("--prompt-file", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if not args.prompt_file.is_file():
        print(f"Prompt file not found: {args.prompt_file}", file=sys.stderr)
        return 1
    for label, ck in ("pretrained", args.pretrained), ("sft", args.sft):
        if not ck.is_file():
            print(f"{label} checkpoint not found: {ck}", file=sys.stderr)
            return 1

    text = args.prompt_file.read_text(encoding="utf-8")
    prompts = _split_prompts(text)
    if not prompts:
        print("No prompts found (use non-empty blocks separated by ---)", file=sys.stderr)
        return 1

    gen_kw = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )

    for i, prompt in enumerate(prompts, start=1):
        print(f"## Prompt {i}\n")
        print("### Input\n")
        print("```")
        print(prompt)
        print("```\n")
        try:
            out_pre = _run_generate(args.pretrained, prompt, **gen_kw)
            out_sft = _run_generate(args.sft, prompt, **gen_kw)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print("### Pretrained\n")
        print(out_pre)
        print("\n### After instruction SFT\n")
        print(out_sft)
        print("\n---\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
