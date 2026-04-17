"""m01_tokenizer：P1-01 Harness（往返、词表、与 config 契约）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_vocab_size_matches_gpt2():
    from mini_llm.m01_tokenizer import ENCODING_NAME, vocab_size

    assert ENCODING_NAME == "gpt2"
    assert vocab_size() == 50257


def test_encode_decode_roundtrip():
    from mini_llm.m01_tokenizer import decode_token_ids, encode_text

    text = "Hello, 世界. Every effort moves you."
    ids = encode_text(text)
    assert all(0 <= i < 50257 for i in ids)
    assert decode_token_ids(ids) == text


def test_endoftext_allowed_in_corpus_string():
    from mini_llm.m01_tokenizer import encode_text

    s = "a<|endoftext|>b"
    ids = encode_text(s)
    assert len(ids) >= 3
    from mini_llm.m01_tokenizer import decode_token_ids

    assert decode_token_ids(ids) == s


def test_vocab_matches_config_ok():
    from mini_llm.m01_tokenizer import assert_vocab_size, vocab_matches_config

    assert vocab_matches_config(50257) is True
    assert_vocab_size(50257)


def test_vocab_mismatch_raises():
    from mini_llm.m01_tokenizer import assert_vocab_size

    with pytest.raises(ValueError, match="vocab_size"):
        assert_vocab_size(1000)
