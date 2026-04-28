"""滑动窗口 Dataset / DataLoader 与语料加载。对照 ch02。"""

from __future__ import annotations

import gc
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mini_llm.m01_tokenizer import encode_text


class GPTDataset(Dataset):
    """整段文本 tokenize 后按滑动窗口切分为 (input, target) 对。

    使用单个连续 Tensor 存储所有 token，__getitem__ 通过索引切片返回
    窗口视图，避免为每个样本分配独立 Tensor（大语料下快 10-100 倍）。
    """

    def __init__(
        self,
        txt: str,
        max_length: int,
        stride: int,
    ) -> None:
        t0 = time.time()
        n_chars = len(txt)
        print(f"  Tokenizing {n_chars / 1e6:.1f}M chars ...", end="", flush=True)
        token_ids = encode_text(txt)
        n_tokens = len(token_ids)
        print(f" {n_tokens:,} tokens ({time.time() - t0:.1f}s)")

        t1 = time.time()
        print(f"  Building tensor ...", end="", flush=True)
        np_arr = np.array(token_ids, dtype=np.int64)
        del token_ids
        gc.collect()
        self._tokens = torch.from_numpy(np_arr)
        print(f" done ({time.time() - t1:.1f}s)")

        self._max_length = max_length
        n_samples = max(0, (n_tokens - max_length) // stride)
        self._offsets = torch.arange(0, n_samples * stride, stride, dtype=torch.long)
        print(f"  Dataset ready: {n_samples:,} samples, "
              f"total {time.time() - t0:.1f}s")

    @classmethod
    def from_tokens(
        cls, tokens: torch.Tensor, max_length: int, stride: int
    ) -> "GPTDataset":
        """从已有 token tensor 构造，跳过 tokenization。"""
        obj = object.__new__(cls)
        obj._tokens = tokens
        obj._max_length = max_length
        n_tokens = len(tokens)
        n_samples = max(0, (n_tokens - max_length) // stride)
        obj._offsets = torch.arange(0, n_samples * stride, stride, dtype=torch.long)
        print(f"  Dataset ready: {n_samples:,} samples (from cached tokens)")
        return obj

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self._offsets[idx].item()
        end = start + self._max_length
        return self._tokens[start:end], self._tokens[start + 1 : end + 1]


def _sibling_book_corpus(filename: str) -> Path | None:
    """若工作区含官方书本仓库，可直接读其 ch02 示例语料（离线开发）。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "LLMs-from-scratch" / "ch02" / "01_main-chapter-code" / filename
        if candidate.is_file():
            return candidate
    return None


def _load_from_huggingface(data_cfg: dict[str, Any], cache_dir: Path | None) -> str:
    """通过 HuggingFace datasets 库下载语料并缓存为本地 txt。"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required: pip install datasets"
        )

    filename = data_cfg["filename"]
    if cache_dir is not None:
        local = cache_dir / filename
        if local.is_file():
            print(f"  [HF cache hit] {local}")
            return local.read_text(encoding="utf-8")

    hf_path = data_cfg["hf_path"]
    hf_name = data_cfg["hf_name"]
    hf_split = data_cfg.get("hf_split", "train")

    print(f"  Downloading {hf_path}/{hf_name} split={hf_split} via HuggingFace ...")
    ds = load_dataset(hf_path, hf_name, split=hf_split)
    print(f"  Joining {len(ds):,} rows ...", end="", flush=True)
    t0 = time.time()
    text = "\n".join(row["text"] for row in ds)
    print(f" {len(text) / 1e6:.1f}M chars ({time.time() - t0:.1f}s)")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        local = cache_dir / filename
        print(f"  Writing cache -> {local} ...", end="", flush=True)
        t1 = time.time()
        local.write_text(text, encoding="utf-8")
        print(f" done ({time.time() - t1:.1f}s)")

    return text


def load_text(data_cfg: dict[str, Any], cache_dir: Path | None = None) -> str:
    """
    读取语料。支持两种 source：

    - ``"url"``（默认）：优先本地 / 同级书本仓库，否则 URL 下载
    - ``"huggingface"``：通过 HuggingFace ``datasets`` 库下载并缓存
    """
    source = data_cfg.get("source", "url")
    filename = data_cfg["filename"]

    env_dir = os.environ.get("TEAM_LLM_DATA_DIR")
    if env_dir:
        p = Path(env_dir) / filename
        if p.is_file():
            return p.read_text(encoding="utf-8")

    if source == "huggingface":
        return _load_from_huggingface(data_cfg, cache_dir)

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
    full_text: str | None,
    train_ratio: float,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    cache_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    ctx = model_cfg["context_length"]
    bs = train_cfg["batch_size"]

    train_pt = cache_dir / "train_tokens.pt" if cache_dir else None
    val_pt = cache_dir / "val_tokens.pt" if cache_dir else None

    if train_pt and train_pt.is_file() and val_pt and val_pt.is_file():
        t0 = time.time()
        print("[Token cache hit] Loading pre-tokenized tensors ...")
        train_tokens = torch.load(train_pt, weights_only=True)
        val_tokens = torch.load(val_pt, weights_only=True)
        print(f"  Loaded {len(train_tokens):,} + {len(val_tokens):,} tokens "
              f"in {time.time() - t0:.1f}s")

        print("[Train split]")
        train_ds = GPTDataset.from_tokens(train_tokens, ctx, ctx)
        print("[Val split]")
        val_ds = GPTDataset.from_tokens(val_tokens, ctx, ctx)

        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=0,
        )
        return train_loader, val_loader

    assert full_text is not None, (
        "full_text required on first run (no token cache found)"
    )
    split_idx = int(train_ratio * len(full_text))
    train_text = full_text[:split_idx]
    val_text = full_text[split_idx:]
    del full_text
    gc.collect()

    print(f"[Train split] {len(train_text)/1e6:.1f}M chars")
    train_loader = create_dataloader(
        train_text,
        batch_size=bs,
        max_length=ctx,
        stride=ctx,
        shuffle=True,
        drop_last=True,
    )
    del train_text
    gc.collect()
    print(f"[Val split] {len(val_text)/1e6:.1f}M chars")
    val_loader = create_dataloader(
        val_text,
        batch_size=bs,
        max_length=ctx,
        stride=ctx,
        shuffle=False,
        drop_last=False,
    )
    del val_text
    gc.collect()

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print("[Saving token cache] ...", end="", flush=True)
        torch.save(train_loader.dataset._tokens, train_pt)
        torch.save(val_loader.dataset._tokens, val_pt)
        print(f" done ({time.time() - t0:.1f}s)")

    return train_loader, val_loader
