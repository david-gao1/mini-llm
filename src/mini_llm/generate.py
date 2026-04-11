"""自回归生成：贪心 / temperature / top-k。对照 ch04 generate_text_simple。"""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def generate_text_simple(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """逐步 argmax 采样（与书本一致）。"""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def _sample_next(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
    """logits: [batch, vocab] -> 下一 token 索引 [batch, 1]。"""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        thresh = v[:, [-1]]
        logits = logits.masked_fill(logits < thresh, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """自回归生成；支持 temperature 与 top-k。"""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)[:, -1, :]
        idx_next = _sample_next(logits, temperature, top_k)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
