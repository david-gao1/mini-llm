from __future__ import annotations

import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_gpt_forward_shape():
    from mini_llm.m04_model import GPTModel

    cfg = {
        "vocab_size": 100,
        "context_length": 8,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 1,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    m = GPTModel(cfg)
    x = torch.randint(0, cfg["vocab_size"], (2, 8))
    logits = m(x)
    assert logits.shape == (2, 8, cfg["vocab_size"])


def test_generate_step():
    from mini_llm.m05_generate import generate
    from mini_llm.m04_model import GPTModel

    cfg = {
        "vocab_size": 50,
        "context_length": 4,
        "emb_dim": 16,
        "n_heads": 2,
        "n_layers": 1,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    m = GPTModel(cfg)
    idx = torch.randint(0, cfg["vocab_size"], (1, 4))
    out = generate(m, idx, max_new_tokens=3, context_size=4, temperature=0.8, top_k=5)
    assert out.shape == (1, 7)
