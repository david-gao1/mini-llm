#!/usr/bin/env python3
"""只计算指令数据 val 上的 CE（抽样口径 + 全量 val），不写 checkpoint。

与 ``finetune_instruction.py`` 使用相同 collate / 配置字段；参见 REQ-P3-02 · C2。

    uv run python eval_instruction_loss.py \\
      --config configs/config_instruction_small.json \\
      --checkpoint runs/instruction_sft_small/checkpoint_best.pt

等价于::

    uv run python finetune_instruction.py --eval-val-only \\
      --eval-checkpoint runs/instruction_sft_small/checkpoint_best.pt \\
      --config configs/config_instruction_small.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from finetune_instruction import run_eval_val_only


def main() -> int:
    p = argparse.ArgumentParser(description="Instruction val loss only (sampled + full)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    args = p.parse_args()
    return run_eval_val_only(args.config, args.checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())
