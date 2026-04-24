"""m02_data_loader：P1-02 Harness（batch 形状、滑动窗口、边界）。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


SAMPLE_TEXT = "Hello world. " * 200


def test_dataset_sample_shapes():
    """每个 sample 的 input 和 target 形状均为 [max_length]。"""
    from mini_llm.m02_data_loader import GPTDataset

    max_length = 8
    ds = GPTDataset(SAMPLE_TEXT, max_length=max_length, stride=max_length)
    assert len(ds) > 0
    inp, tgt = ds[0]
    assert inp.shape == (max_length,)
    assert tgt.shape == (max_length,)
    assert inp.dtype == torch.long
    assert tgt.dtype == torch.long


def test_target_is_shifted_input():
    """target 应为 input 右移一位（next-token prediction）。"""
    from mini_llm.m02_data_loader import GPTDataset

    ds = GPTDataset(SAMPLE_TEXT, max_length=4, stride=4)
    inp, tgt = ds[0]
    from mini_llm.m01_tokenizer import encode_text

    all_ids = encode_text(SAMPLE_TEXT)
    assert inp.tolist() == all_ids[0:4]
    assert tgt.tolist() == all_ids[1:5]


def test_dataloader_batch_shape():
    """DataLoader 输出 batch 形状为 [B, T]。"""
    from mini_llm.m02_data_loader import create_dataloader

    batch_size = 4
    max_length = 8
    loader = create_dataloader(
        SAMPLE_TEXT,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
        drop_last=True,
    )
    batch_in, batch_tgt = next(iter(loader))
    assert batch_in.shape == (batch_size, max_length)
    assert batch_tgt.shape == (batch_size, max_length)


def test_no_token_id_out_of_range():
    """所有 token id 在 [0, vocab_size) 范围内。"""
    from mini_llm.m02_data_loader import GPTDataset
    from mini_llm.m01_tokenizer import vocab_size

    ds = GPTDataset(SAMPLE_TEXT, max_length=8, stride=8)
    v = vocab_size()
    for i in range(len(ds)):
        inp, tgt = ds[i]
        assert inp.min() >= 0 and inp.max() < v
        assert tgt.min() >= 0 and tgt.max() < v


def test_stride_controls_overlap():
    """stride < max_length 时窗口有重叠，样本数应更多。"""
    from mini_llm.m02_data_loader import GPTDataset

    max_length = 8
    ds_no_overlap = GPTDataset(SAMPLE_TEXT, max_length=max_length, stride=max_length)
    ds_overlap = GPTDataset(SAMPLE_TEXT, max_length=max_length, stride=max_length // 2)
    assert len(ds_overlap) > len(ds_no_overlap)


def test_short_text_produces_empty_dataset():
    """文本太短（不足 max_length + 1 个 token）时 dataset 为空。"""
    from mini_llm.m02_data_loader import GPTDataset

    ds = GPTDataset("hi", max_length=1024, stride=1024)
    assert len(ds) == 0


def test_train_val_split():
    """train_val_dataloaders 返回两个非空 loader。"""
    from mini_llm.m02_data_loader import train_val_dataloaders

    model_cfg = {"context_length": 8}
    train_cfg = {"batch_size": 2}
    train_loader, val_loader = train_val_dataloaders(
        SAMPLE_TEXT, train_ratio=0.8, model_cfg=model_cfg, train_cfg=train_cfg
    )
    assert len(train_loader) > 0
    assert len(val_loader) > 0
