"""GPT-2 BPE 分词器（tiktoken 封装）。契约见 docs/m01_tokenizer.md。"""

from __future__ import annotations

from typing import AbstractSet, Literal

import tiktoken


class GPT2Tokenizer:
    """基于 tiktoken 的 GPT-2 BPE 分词器。

    职责：text ↔ list[int] 的可逆转换 + 词表校验。
    """

    ENCODING_NAME: str = "gpt2"
    ALLOWED_SPECIAL_DEFAULT: frozenset[str] = frozenset({"<|endoftext|>"})

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding(self.ENCODING_NAME)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def encoding(self) -> tiktoken.Encoding:
        """底层 tiktoken Encoding 对象。"""
        return self._encoding

    @property
    def vocab_size(self) -> int:
        """词表大小（GPT-2 = 50257）。"""
        return self._encoding.n_vocab

    # ------------------------------------------------------------------
    # 核心能力：编码 / 解码
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        allowed_special: AbstractSet[str] | Literal["all"] | None = None,
    ) -> list[int]:
        """文本 → token id 列表。

        ``allowed_special is None`` 时使用 :data:`ALLOWED_SPECIAL_DEFAULT`
        （允许语料中的 ``<|endoftext|>``）。若确定正文不含任何 special，可传
        ``allowed_special=set()`` 以禁止 special。
        """
        if allowed_special is None:
            allowed_special = self.ALLOWED_SPECIAL_DEFAULT
        return self._encoding.encode(text, allowed_special=allowed_special)

    def decode(self, token_ids: list[int]) -> str:
        """token id 列表 → 文本（与 :meth:`encode` 互逆）。"""
        return self._encoding.decode(token_ids)

    # ------------------------------------------------------------------
    # 词表契约校验
    # ------------------------------------------------------------------

    def assert_vocab_size(self, expected: int) -> None:
        """与 ``configs/config.json`` 中 ``model.vocab_size`` 对齐时调用。

        不一致则抛 ValueError，避免静默错配嵌入维。
        """
        actual = self.vocab_size
        if actual != expected:
            raise ValueError(
                f"tokenizer vocab_size ({actual}) != expected ({expected}); "
                "keep ENCODING_NAME / model.vocab_size in sync."
            )

    def vocab_matches_config(self, model_vocab_size: int) -> bool:
        """是否与给定配置中的词表大小一致（软检查，不抛错）。"""
        return self.vocab_size == model_vocab_size
