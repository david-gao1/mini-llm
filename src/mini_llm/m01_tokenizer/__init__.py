"""GPT-2 BPE（tiktoken）。对照 ch02；契约见 docs/m01_tokenizer.md。"""

from __future__ import annotations

from typing import AbstractSet, Literal

import tiktoken

ENCODING_NAME = "gpt2"

# 语料中若出现文档边界符，须在 encode 时显式允许（与 m02 滑动窗口管线一致）
ALLOWED_SPECIAL_DEFAULT: frozenset[str] = frozenset({"<|endoftext|>"})

__all__ = [
    "ALLOWED_SPECIAL_DEFAULT",
    "ENCODING_NAME",
    "assert_vocab_size",
    "decode_token_ids",
    "encode_text",
    "get_encoding",
    "vocab_size",
]


def get_encoding() -> tiktoken.Encoding:
    """与 OpenAI GPT-2 一致的 BPE 编码；词表大小 50257。"""
    return tiktoken.get_encoding(ENCODING_NAME)


def vocab_size() -> int:
    return get_encoding().n_vocab


def encode_text(
    text: str,
    *,
    allowed_special: AbstractSet[str] | Literal["all"] | None = None,
) -> list[int]:
    """
    文本 → token id 列表。

    ``allowed_special is None`` 时使用 :data:`ALLOWED_SPECIAL_DEFAULT`
   （允许语料中的 ``<|endoftext|>``）。若确定正文不含任何 special，可传
    ``allowed_special=set()`` 以禁止 special。
    """
    enc = get_encoding()
    if allowed_special is None:
        allowed_special = ALLOWED_SPECIAL_DEFAULT
    return enc.encode(text, allowed_special=allowed_special)


def decode_token_ids(token_ids: list[int]) -> str:
    """token id 列表 → 文本（与 :func:`encode_text` 互逆）。"""
    return get_encoding().decode(token_ids)


def assert_vocab_size(expected: int) -> None:
    """
    与 ``configs/config.json`` 中 ``model.vocab_size`` 对齐时调用。
    不一致则抛错，避免静默错配嵌入维。
    """
    actual = vocab_size()
    if actual != expected:
        msg = (
            f"tokenizer vocab_size ({actual}) != expected ({expected}); "
            "keep ENCODING_NAME / model.vocab_size in sync."
        )
        raise ValueError(msg)


def vocab_matches_config(model_vocab_size: int) -> bool:
    """是否与给定配置中的词表大小一致（训练脚本可据此打日志而不断训）。"""
    return vocab_size() == model_vocab_size
