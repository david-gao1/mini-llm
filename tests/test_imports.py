from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_package_importable():
    import mini_llm
    import mini_llm.data_loader  # noqa: F401
    import mini_llm.tokenizer  # noqa: F401

    assert hasattr(mini_llm, "__version__")
