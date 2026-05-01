"""指令微调数据管线单测（对齐书本 collate / mask 行为）。"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from mini_llm.m07_instruction_finetune import (
    PAD_TOKEN_ID_DEFAULT,
    InstructionDataset,
    download_instruction_json,
    format_input,
    instruction_collate_fn,
    split_instruction_entries,
)


def test_format_input_optional_input():
    e1 = {"instruction": "Say hi", "input": "", "output": "Hello"}
    assert "### Instruction:\nSay hi" in format_input(e1)
    assert "### Input:" not in format_input(e1)

    e2 = {"instruction": "Translate", "input": "bonjour", "output": "hello"}
    s = format_input(e2)
    assert "### Input:\nbonjour" in s
    assert "### Instruction:\nTranslate" in s


def test_split_instruction_entries_counts():
    data = [{"instruction": str(i), "input": "", "output": "x"} for i in range(100)]
    train, val, test = split_instruction_entries(data, train_ratio=0.85, test_ratio=0.1)
    assert len(train) == 85
    assert len(test) == 10
    assert len(val) == 5


def test_instruction_collate_batches_two_lengths():
    pad = PAD_TOKEN_ID_DEFAULT
    batch = [
        [1, 2, 3],
        [4, 5],
    ]
    inputs, targets = instruction_collate_fn(batch, pad_token_id=pad, ignore_index=-100, device=None)
    assert inputs.shape == targets.shape
    assert inputs.shape[0] == 2
    B, T = inputs.shape
    assert B == 2 and T == 3


def test_instruction_collate_allowed_max_length_truncates():
    long_seq = list(range(80))
    batch = [long_seq, [1, 2]]
    inputs, targets = instruction_collate_fn(
        batch,
        allowed_max_length=16,
        pad_token_id=PAD_TOKEN_ID_DEFAULT,
        ignore_index=-100,
        device=None,
    )
    assert inputs.shape[1] == 16
    assert targets.shape[1] == 16


def test_instruction_dataset_roundtrip_encode():
    entries = [
        {"instruction": "Add", "input": "1+1", "output": "2"},
        {"instruction": "Name color", "input": "", "output": "Blue"},
    ]

    def fake_encode(text: str) -> list[int]:
        return [min(100 + len(text) % 17 + i, 50000) for i in range(min(len(text), 32))]

    ds = InstructionDataset(entries, fake_encode)
    assert len(ds) == 2
    assert isinstance(ds[0], list)
    assert len(ds[0]) > 0


def test_download_instruction_json_reads_local(tmp_path: Path):
    path = tmp_path / "instruction-data.json"
    data = [{"instruction": "a", "input": "", "output": "b"}]
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = download_instruction_json(path, url="http://invalid.example/no-fetch")
    assert loaded == data
