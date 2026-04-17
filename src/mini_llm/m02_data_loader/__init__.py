"""滑动窗口 Dataset / DataLoader 与语料加载。对照 ch02。"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from mini_llm.m01_tokenizer import encode_text


class GPTDataset(Dataset):
    """整段文本 tokenize 后按滑动窗口切分为 (input, target) 对。"""

    def __init__(
        self,
        txt: str,
        max_length: int,
        stride: int,
    ) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []
        token_ids = encode_text(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            chunk_in = token_ids[i : i + max_length]
            chunk_tgt = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(chunk_in, dtype=torch.long))
            self.target_ids.append(torch.tensor(chunk_tgt, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def _sibling_book_corpus(filename: str) -> Path | None:
    """若工作区含官方书本仓库，可直接读其 ch02 示例语料（离线开发）。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "LLMs-from-scratch" / "ch02" / "01_main-chapter-code" / filename
        if candidate.is_file():
            return candidate
    return None


def load_text(data_cfg: dict[str, Any], cache_dir: Path | None = None) -> str:
    """
    读取语料：优先 ``TEAM_LLM_DATA_DIR/<filename>``；
    其次同级 ``LLMs-from-scratch/ch02/...`` 中的同名文件；
    否则将 ``url`` 下载到 cache_dir。
    """
    filename = data_cfg["filename"]
    env_dir = os.environ.get("TEAM_LLM_DATA_DIR")
    if env_dir:
        p = Path(env_dir) / filename
        if p.is_file():
            return p.read_text(encoding="utf-8")

    sibling = _sibling_book_corpus(filename)
    if sibling is not None:
        return sibling.read_text(encoding="utf-8")

    if cache_dir is not None:
        local = cache_dir / filename
        if local.is_file():
            return local.read_text(encoding="utf-8")
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = data_cfg["url"]
        with urllib.request.urlopen(url) as resp:
            text = resp.read().decode("utf-8")
        local.write_text(text, encoding="utf-8")
        return text

    raise ValueError("Provide cache_dir or set TEAM_LLM_DATA_DIR with the file present")


def create_dataloader(
    text: str,
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int = 0,
) -> DataLoader:
    ds = GPTDataset(text, max_length, stride)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def train_val_dataloaders(
    full_text: str,
    train_ratio: float,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    split_idx = int(train_ratio * len(full_text))
    train_text = full_text[:split_idx]
    val_text = full_text[split_idx:]
    ctx = model_cfg["context_length"]
    bs = train_cfg["batch_size"]

    train_loader = create_dataloader(
        train_text,
        batch_size=bs,
        max_length=ctx,
        stride=ctx,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_text,
        batch_size=bs,
        max_length=ctx,
        stride=ctx,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader
