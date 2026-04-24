"""m03_attention：P1-03 Harness（因果 mask、多头输出形状、梯度有限）。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


D_IN = 32
D_OUT = 32
CTX_LEN = 8
NUM_HEADS = 4
BATCH = 2


def _make_mha(**overrides):
    from mini_llm.m03_attention import MultiHeadAttention

    kw = dict(
        d_in=D_IN,
        d_out=D_OUT,
        context_length=CTX_LEN,
        dropout=0.0,
        num_heads=NUM_HEADS,
        qkv_bias=False,
    )
    kw.update(overrides)
    return MultiHeadAttention(**kw)


def test_output_shape():
    """输出形状应为 [B, T, d_out]。"""
    mha = _make_mha()
    x = torch.randn(BATCH, CTX_LEN, D_IN)
    out = mha(x)
    assert out.shape == (BATCH, CTX_LEN, D_OUT)


def test_output_shape_shorter_sequence():
    """序列长度 < context_length 时也能正确运行。"""
    mha = _make_mha()
    seq_len = 3
    x = torch.randn(BATCH, seq_len, D_IN)
    out = mha(x)
    assert out.shape == (BATCH, seq_len, D_OUT)


def test_causal_mask():
    """因果 mask：位置 i 不应受位置 j > i 的输入影响。

    将 future 位置的输入替换为极大值，如果 mask 正确，
    early 位置的输出应完全不变。
    """
    torch.manual_seed(42)
    mha = _make_mha(dropout=0.0)
    mha.eval()

    x1 = torch.randn(1, CTX_LEN, D_IN)
    x2 = x1.clone()
    x2[0, CTX_LEN // 2 :, :] = torch.randn(CTX_LEN - CTX_LEN // 2, D_IN) * 100

    out1 = mha(x1)
    out2 = mha(x2)

    early = CTX_LEN // 2
    assert torch.allclose(out1[0, :early], out2[0, :early], atol=1e-5), (
        "Causal mask violated: early positions changed when future inputs differ"
    )


def test_gradient_is_finite():
    """反向传播后梯度应为有限值（无 NaN / Inf）。"""
    mha = _make_mha()
    x = torch.randn(BATCH, CTX_LEN, D_IN, requires_grad=True)
    out = mha(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), "Input gradient contains NaN or Inf"

    for name, p in mha.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Gradient of {name} contains NaN or Inf"


def test_d_out_not_divisible_by_heads_raises():
    """d_out 不能被 num_heads 整除时应报错。"""
    import pytest

    with pytest.raises(AssertionError, match="divisible"):
        _make_mha(d_out=30, num_heads=4)


def test_qkv_bias():
    """qkv_bias=True 时线性层应带 bias。"""
    mha = _make_mha(qkv_bias=True)
    assert mha.W_query.bias is not None
    assert mha.W_key.bias is not None
    assert mha.W_value.bias is not None

    mha_no = _make_mha(qkv_bias=False)
    assert mha_no.W_query.bias is None
