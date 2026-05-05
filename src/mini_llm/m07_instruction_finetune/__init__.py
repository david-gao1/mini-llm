"""第 7 章风格指令微调数据管线（与书本 ``gpt_instruction_finetuning.py`` 对齐）。

提供 ``format_input``、``InstructionDataset``、``instruction_collate_fn``（pad +
``ignore_index`` mask）、数据划分与可选下载。
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

__all__ = [
    "IGNORE_INDEX_DEFAULT",
    "PAD_TOKEN_ID_DEFAULT",
    "InstructionDataset",
    "download_instruction_json",
    "format_input",
    "instruction_collate_fn",
    "make_instruction_collate_fn",
    "split_instruction_entries",
]

PAD_TOKEN_ID_DEFAULT = 50256
IGNORE_INDEX_DEFAULT = -100


def format_input(entry: dict[str, Any]) -> str:
    """拼 Instruction / 可选 Input（与书本一致，不含 Response）。"""
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    inp = entry.get("input") or ""
    input_text = f"\n\n### Input:\n{inp}" if inp else ""
    return instruction_text + input_text


def split_instruction_entries(
    data: list[dict[str, Any]],
    *,
    train_ratio: float = 0.85,
    test_ratio: float = 0.1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """书本划分：train 前 85%，再接 10% test，剩余为 val。

    返回顺序为 (train, val, test)，与时间上连续三段不完全同名：中段切片名为 test_data，
    尾段为 val_data；调用方常用 train + val，中段可留作别的评估。
    """
    n = len(data)
    train_end = int(n * train_ratio)
    test_end = train_end + int(n * test_ratio)
    train_data = data[:train_end]
    test_data = data[train_end:test_end]
    val_data = data[test_end:]
    return train_data, val_data, test_data


def download_instruction_json(cache_path: Path, url: str) -> list[dict[str, Any]]:
    """若本地无文件则从 URL 下载 JSON；返回解析后的列表。"""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.is_file():
        with urllib.request.urlopen(url) as response:  # noqa: S310
            text_data = response.read().decode("utf-8")
        cache_path.write_text(text_data, encoding="utf-8")
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("instruction JSON must be a list of objects")
    return raw


class InstructionDataset(Dataset):
    """每条样本预编码为整段 token（含 ``### Response:\\n`` + output）。"""

    def __init__(
        self,
        data: list[dict[str, Any]],
        encode_fn: Callable[[str], list[int]],
    ) -> None:
        self._encoded: list[list[int]] = []
        for entry in data:
            prefix = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            # 模型学习整段续写；监督涵盖 instruction 到 answer（pad 再交给 collate mask）。
            full_text = prefix + response_text
            self._encoded.append(encode_fn(full_text))

    def __getitem__(self, index: int) -> list[int]:
        return self._encoded[index]

    def __len__(self) -> int:
        return len(self._encoded)


def instruction_collate_fn(
    batch: list[list[int]],
    *,
    pad_token_id: int = PAD_TOKEN_ID_DEFAULT,
    ignore_index: int = IGNORE_INDEX_DEFAULT,
    allowed_max_length: int | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """与书本 ``custom_collate_fn`` 一致：末尾 eos（pad_id）、pad 对齐、targets 除首个 pad 外 mask。"""
    # +1：每条先追加一个 pad_id，当作「预测序列结束」的一步边界（与书一致）。
    batch_max_length = max(len(item) + 1 for item in batch)

    inputs_lst: list[torch.Tensor] = []
    targets_lst: list[torch.Tensor] = []

    for item in batch:
        new_item = list(item)
        new_item.append(pad_token_id)
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        # 错位一位：inputs[t] 预测 targets[t]（即下一 token）。
        inputs = torch.tensor(padded[:-1], dtype=torch.long)
        targets = torch.tensor(padded[1:], dtype=torch.long)

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        # 保留第一个 pad 位的监督（转移出真实内容）；其后 pad 不记分，避免学「填满 pad」。
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst)
    targets_tensor = torch.stack(targets_lst)
    if device is not None:
        inputs_tensor = inputs_tensor.to(device)
        targets_tensor = targets_tensor.to(device)
    return inputs_tensor, targets_tensor


def make_instruction_collate_fn(
    *,
    pad_token_id: int = PAD_TOKEN_ID_DEFAULT,
    ignore_index: int = IGNORE_INDEX_DEFAULT,
    allowed_max_length: int | None = None,
    device: torch.device | None = None,
) -> Callable[[list[list[int]]], tuple[torch.Tensor, torch.Tensor]]:
    """供 ``DataLoader(collate_fn=...)`` 使用的偏函数工厂。"""

    def collate(batch: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        return instruction_collate_fn(
            batch,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
            allowed_max_length=allowed_max_length,
            device=device,
        )

    return collate
