"""m06_classify_finetune：P2-02 Harness（SpamDataset 形状、padding、accuracy 函数）。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

PAD_TOKEN_ID = 50256


def _make_csv(tmp: Path, n: int = 20) -> Path:
    """Create a tiny CSV with fake SMS data for testing."""
    rows = []
    for i in range(n):
        label = i % 2
        text = f"Hello this is message number {i}" if label == 0 else f"FREE prize winner call now {i}"
        rows.append({"Label": label, "Text": text})
    df = pd.DataFrame(rows)
    csv_path = tmp / "test_spam.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_spam_dataset_shapes():
    """Each sample returns (token_ids[max_length], label scalar)."""
    from mini_llm.m06_classify_finetune import SpamDataset

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _make_csv(Path(tmp))
        ds = SpamDataset(csv_path)
        assert len(ds) == 20
        token_ids, label = ds[0]
        assert token_ids.shape == (ds.max_length,)
        assert token_ids.dtype == torch.long
        assert label.dtype == torch.long
        assert label.item() in (0, 1)


def test_spam_dataset_padding():
    """Shorter sequences are padded with pad_token_id on the right."""
    from mini_llm.m06_classify_finetune import SpamDataset

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _make_csv(Path(tmp))
        ds = SpamDataset(csv_path)
        token_ids, _ = ds[0]
        ids_list = token_ids.tolist()
        if PAD_TOKEN_ID in ids_list:
            first_pad = ids_list.index(PAD_TOKEN_ID)
            assert all(t == PAD_TOKEN_ID for t in ids_list[first_pad:])


def test_spam_dataset_max_length_override():
    """Explicit max_length truncates longer sequences."""
    from mini_llm.m06_classify_finetune import SpamDataset

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _make_csv(Path(tmp))
        ds = SpamDataset(csv_path, max_length=5)
        assert ds.max_length == 5
        token_ids, _ = ds[0]
        assert token_ids.shape == (5,)


def test_spam_dataset_labels_binary():
    """All labels are 0 or 1."""
    from mini_llm.m06_classify_finetune import SpamDataset

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _make_csv(Path(tmp))
        ds = SpamDataset(csv_path)
        for i in range(len(ds)):
            _, label = ds[i]
            assert label.item() in (0, 1)


def test_calc_accuracy_loader_returns_valid_float():
    """calc_accuracy_loader returns a float in [0, 1]."""
    from torch.utils.data import DataLoader

    from mini_llm.m04_model import GPTModel
    from mini_llm.m06_classify_finetune import SpamDataset, calc_accuracy_loader

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _make_csv(Path(tmp))
        ds = SpamDataset(csv_path, max_length=16)
        loader = DataLoader(ds, batch_size=4, drop_last=False)

        tiny_cfg = {
            "vocab_size": 50257,
            "context_length": 16,
            "emb_dim": 12,
            "n_heads": 2,
            "n_layers": 1,
            "drop_rate": 0.0,
            "qkv_bias": False,
        }
        model = GPTModel(tiny_cfg)
        model.out_head = torch.nn.Linear(12, 2)
        model.eval()

        acc = calc_accuracy_loader(loader, model, torch.device("cpu"), num_batches=2)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0


def test_calc_loss_batch_finite():
    """calc_loss_batch returns a finite scalar."""
    from mini_llm.m04_model import GPTModel
    from mini_llm.m06_classify_finetune import SpamDataset, calc_loss_batch

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _make_csv(Path(tmp))
        ds = SpamDataset(csv_path, max_length=16)
        token_ids, label = ds[0]

        tiny_cfg = {
            "vocab_size": 50257,
            "context_length": 16,
            "emb_dim": 12,
            "n_heads": 2,
            "n_layers": 1,
            "drop_rate": 0.0,
            "qkv_bias": False,
        }
        model = GPTModel(tiny_cfg)
        model.out_head = torch.nn.Linear(12, 2)

        loss = calc_loss_batch(
            token_ids.unsqueeze(0),
            label.unsqueeze(0),
            model,
            torch.device("cpu"),
        )
        assert torch.isfinite(loss)


def test_encode_spam_text_matches_dataset_row():
    """encode_spam_text_for_model matches SpamDataset padding for same raw text."""
    from mini_llm.m06_classify_finetune import SpamDataset, encode_spam_text_for_model

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _make_csv(Path(tmp))
        ds = SpamDataset(csv_path)
        idx = 3
        text = str(ds.data.iloc[idx]["Text"])
        row_tensor, _ = ds[idx]
        encoded = encode_spam_text_for_model(text, ds.max_length).squeeze(0)
        assert encoded.shape == row_tensor.shape
        assert (encoded == row_tensor).all()


def test_load_spam_classifier_checkpoint_roundtrip():
    """Tiny checkpoint save/load used by classify_sms."""
    from mini_llm.m04_model import GPTModel
    from mini_llm.m06_classify_finetune import (
        encode_spam_text_for_model,
        load_spam_classifier_checkpoint,
    )

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "spam_cls.pt"
        tiny_cfg = {
            "vocab_size": 50257,
            "context_length": 64,
            "emb_dim": 32,
            "n_heads": 4,
            "n_layers": 2,
            "drop_rate": 0.0,
            "qkv_bias": False,
        }
        model = GPTModel(tiny_cfg)
        model.out_head = torch.nn.Linear(32, 2)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {"model": tiny_cfg, "run_name": "test"},
                "num_classes": 2,
                "spam_max_length": 12,
                "pad_token_id": PAD_TOKEN_ID,
                "best_val_accuracy": 0.99,
            },
            ckpt_path,
        )

        loaded, meta = load_spam_classifier_checkpoint(ckpt_path, torch.device("cpu"))
        assert meta["spam_max_length"] == 12
        assert meta["pad_token_id"] == PAD_TOKEN_ID
        assert meta["num_classes"] == 2

        x = encode_spam_text_for_model("hello test sms", 12)
        with torch.no_grad():
            logits = loaded(x)[:, -1, :]
        assert logits.shape == (1, 2)
        assert torch.isfinite(logits).all()
