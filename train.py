#!/usr/bin/env python3
"""训练入口：在 src/mini_llm/ 实现模型与数据后在此串联。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


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
    try:
        import mini_llm.model  # noqa: F401
    except ImportError as e:
        print(e, file=sys.stderr)
        return 1
    print("请在 mini_llm 中完成实现后，在此编写训练循环（损失、优化器、checkpoint）。")
    print(f"配置: {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
