"""混淆矩阵、PRF、FN 导出与探针清单（BL-P2-02-02 / 阶段 A）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_REPO = Path(__file__).resolve().parents[1]
PROBES_JSON = _REPO / "docs" / "probes" / "classify_spam_probes.json"


def test_confusion_counts_and_prf_known():
    from mini_llm.m06_classify_finetune import (
        accuracy_from_counts,
        confusion_counts_binary_spam,
        prf1_from_counts,
    )

    t = torch.tensor([0, 0, 1, 1, 1])
    p = torch.tensor([0, 1, 1, 0, 1])
    tn, fp, fn, tp = confusion_counts_binary_spam(t, p)
    assert (tn, fp, fn, tp) == (1, 1, 1, 2)
    assert accuracy_from_counts(tn, fp, fn, tp) == 3 / 5
    m = prf1_from_counts(tn, fp, fn, tp)
    assert abs(m["recall_spam"] - 2 / 3) < 1e-6
    assert abs(m["precision_spam"] - 2 / 3) < 1e-6


def test_collect_predictions_loader_matches_manual(tmp_path):
    from mini_llm.m04_model import GPTModel
    from mini_llm.m06_classify_finetune import (
        SpamDataset,
        collect_predictions_loader,
        confusion_counts_binary_spam,
    )

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
    model.eval()
    device = torch.device("cpu")

    rows = [
        {"Label": 0, "Text": "hello world normal sms"},
        {"Label": 1, "Text": "winner free prize call now"},
        {"Label": 0, "Text": "coffee at five"},
    ]
    csv_path = tmp_path / "ds.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    ds = SpamDataset(csv_path, max_length=16)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    preds, tgts = collect_predictions_loader(loader, model, device)
    assert preds.shape == tgts.shape == (3,)
    tn, fp, fn, tp = confusion_counts_binary_spam(tgts, preds)
    assert tn + fp + fn + tp == 3


def test_export_fn_csv(tmp_path):
    from mini_llm.m06_classify_finetune import SpamDataset, export_false_negative_spam_csv

    rows = [
        {"Label": 1, "Text": "spam one"},
        {"Label": 1, "Text": "spam two missed"},
        {"Label": 0, "Text": "ham ok"},
    ]
    csv_in = tmp_path / "in.csv"
    pd.DataFrame(rows).to_csv(csv_in, index=False)
    ds = SpamDataset(csv_in, max_length=12)
    targets = torch.tensor([1, 1, 0])
    preds = torch.tensor([0, 1, 0])
    out = tmp_path / "fn.csv"
    n = export_false_negative_spam_csv(ds, targets, preds, out)
    assert n == 1
    body = out.read_text(encoding="utf-8")
    assert "spam one" in body


def test_classify_spam_probes_json_schema():
    assert PROBES_JSON.is_file(), f"missing {PROBES_JSON}"
    data = json.loads(PROBES_JSON.read_text(encoding="utf-8"))
    assert data.get("version") == 1
    probes = data["probes"]
    assert len(probes) >= 1
    for p in probes:
        assert "id" in p and "text" in p and "expected" in p
        assert p["expected"] in ("ham", "spam")


def test_format_classification_eval_lines_non_empty():
    from mini_llm.m06_classify_finetune import format_classification_eval_lines, prf1_from_counts

    tn, fp, fn, tp = 10, 2, 3, 15
    m = prf1_from_counts(tn, fp, fn, tp)
    lines = format_classification_eval_lines(tn, fp, fn, tp, m)
    assert any("pred_ham" in ln for ln in lines)
    assert any("F1=" in ln for ln in lines)
