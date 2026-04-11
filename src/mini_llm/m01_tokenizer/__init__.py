"""GPT-2 BPE（tiktoken）。对照 ch02。"""

from __future__ import annotations

import tiktoken

ENCODING_NAME = "gpt2"


def get_encoding() -> tiktoken.Encoding:
    """与 OpenAI GPT-2 一致的 BPE 编码；词表大小 50257。"""
    return tiktoken.get_encoding(ENCODING_NAME)


def vocab_size() -> int:
    return get_encoding().n_vocab
