"""m01_tokenizer 包入口。

实现类：:class:`GPT2Tokenizer`（``tokenizer.py``）。
本文件提供模块级便捷函数，保持下游 ``from mini_llm.m01_tokenizer import encode_text`` 等用法不变。
"""

from __future__ import annotations

from typing import AbstractSet, Literal

import tiktoken

from mini_llm.m01_tokenizer.tokenizer import GPT2Tokenizer

__all__ = [
    "ALLOWED_SPECIAL_DEFAULT",
    "ENCODING_NAME",
    "GPT2Tokenizer",
    "assert_vocab_size",
    "decode_token_ids",
    "encode_text",
    "get_encoding",
    "vocab_size",
]

_default = GPT2Tokenizer()

ENCODING_NAME: str = GPT2Tokenizer.ENCODING_NAME
ALLOWED_SPECIAL_DEFAULT: frozenset[str] = GPT2Tokenizer.ALLOWED_SPECIAL_DEFAULT


def get_encoding() -> tiktoken.Encoding:
    """与 OpenAI GPT-2 一致的 BPE 编码；词表大小 50257。"""
    return _default.encoding


def vocab_size() -> int:
    return _default.vocab_size


def encode_text(
    text: str,
    *,
    allowed_special: AbstractSet[str] | Literal["all"] | None = None,
) -> list[int]:
    """文本 → token id 列表（委托给默认 :class:`GPT2Tokenizer` 实例）。"""
    return _default.encode(text, allowed_special=allowed_special)


def decode_token_ids(token_ids: list[int]) -> str:
    """token id 列表 → 文本。"""
    return _default.decode(token_ids)


def assert_vocab_size(expected: int) -> None:
    """不一致则 raise ValueError。"""
    _default.assert_vocab_size(expected)


def vocab_matches_config(model_vocab_size: int) -> bool:
    """是否与给定配置中的词表大小一致。"""
    return _default.vocab_matches_config(model_vocab_size)
