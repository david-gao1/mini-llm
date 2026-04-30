"""分类微调数据管道与评估工具。

提供 SMS Spam 数据集下载 / 平衡 / 切分、SpamDataset、单条编码与 checkpoint 加载、
分类损失 / 准确率评估。
"""

from __future__ import annotations

import csv
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import tiktoken
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "download_and_prepare_spam",
    "SpamDataset",
    "encode_spam_text_for_model",
    "load_spam_classifier_checkpoint",
    "calc_loss_batch",
    "calc_loss_loader",
    "evaluate_model",
    "calc_accuracy_loader",
    "collect_predictions_loader",
    "confusion_counts_binary_spam",
    "prf1_from_counts",
    "accuracy_from_counts",
    "export_false_negative_spam_csv",
    "format_classification_eval_lines",
]

_PRIMARY_URL = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
)
_BACKUP_URL = (
    "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
)


def _download_and_unzip(url: str, zip_path: Path, extract_dir: Path) -> None:
    with urllib.request.urlopen(url, timeout=30) as resp:
        zip_path.write_bytes(resp.read())
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def download_and_prepare_spam(
    data_dir: Path,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> tuple[Path, Path, Path]:
    """Download UCI SMS Spam → balance → split → return CSV paths.

    Returns (train.csv, validation.csv, test.csv) inside *data_dir*.
    """
    data_dir = Path(data_dir)
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "validation.csv"
    test_csv = data_dir / "test.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        print(f"SMS Spam CSVs already exist in {data_dir}, skipping download.")
        return train_csv, val_csv, test_csv

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "sms_spam_collection.zip"
    extract_dir = data_dir / "raw"
    tsv_path = extract_dir / "SMSSpamCollection.tsv"

    if not tsv_path.exists():
        try:
            print(f"Downloading SMS Spam dataset from primary URL …")
            _download_and_unzip(_PRIMARY_URL, zip_path, extract_dir)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            print(f"Primary URL failed ({exc}), trying backup …")
            _download_and_unzip(_BACKUP_URL, zip_path, extract_dir)

        original = extract_dir / "SMSSpamCollection"
        if original.exists() and not tsv_path.exists():
            os.rename(original, tsv_path)
        print(f"Downloaded → {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["Label", "Text"])

    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    balanced["Label"] = balanced["Label"].map({"ham": 0, "spam": 1})

    balanced = balanced.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(balanced) * train_frac)
    val_end = train_end + int(len(balanced) * val_frac)

    balanced[:train_end].to_csv(train_csv, index=False)
    balanced[train_end:val_end].to_csv(val_csv, index=False)
    balanced[val_end:].to_csv(test_csv, index=False)
    print(
        f"SMS Spam split: train={train_end}, val={val_end - train_end}, "
        f"test={len(balanced) - val_end}"
    )
    return train_csv, val_csv, test_csv


def encode_spam_text_for_model(
    text: str,
    max_length: int,
    *,
    pad_token_id: int = 50256,
) -> torch.Tensor:
    """Encode one SMS string like ``SpamDataset``: GPT-2 BPE → truncate → right-pad.

    Returns ``token_ids`` of shape ``[1, max_length]`` (batch size 1).
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    ids = tokenizer.encode(text)
    if len(ids) > max_length:
        ids = ids[:max_length]
    ids = ids + [pad_token_id] * (max_length - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def load_spam_classifier_checkpoint(
    path: Path | str,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Load ``finetune_classify.py`` best checkpoint: rebuild GPT + classification head, eval mode.

    Returns ``(model, meta)``. ``meta`` includes ``spam_max_length``, ``pad_token_id``,
    ``num_classes``, ``best_val_accuracy`` (if present).
    """
    from mini_llm.m04_model import GPTModel

    path = Path(path)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    if "model_state_dict" not in ckpt or "config" not in ckpt:
        raise ValueError(f"Not a classification finetune checkpoint: {path}")

    model_cfg = ckpt["config"]["model"]
    num_classes = int(ckpt.get("num_classes", 2))

    model = GPTModel(model_cfg)
    model.out_head = nn.Linear(model_cfg["emb_dim"], num_classes, bias=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    meta = {
        "spam_max_length": ckpt.get("spam_max_length"),
        "pad_token_id": int(ckpt.get("pad_token_id", 50256)),
        "num_classes": num_classes,
        "best_val_accuracy": ckpt.get("best_val_accuracy"),
        "finetune_config": ckpt.get("finetune_config"),
    }
    return model, meta


class SpamDataset(Dataset):
    """Fixed-length tokenised SMS dataset for classification.

    Each sample: ``(token_ids: LongTensor[max_length], label: LongTensor scalar)``.
    """

    def __init__(
        self,
        csv_path: Path | str,
        max_length: int | None = None,
        pad_token_id: int = 50256,
    ) -> None:
        self.data = pd.read_csv(csv_path)
        tokenizer = tiktoken.get_encoding("gpt2")

        self.encoded_texts: list[list[int]] = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = max(len(t) for t in self.encoded_texts)
        else:
            self.max_length = max_length
            self.encoded_texts = [t[: self.max_length] for t in self.encoded_texts]

        self.encoded_texts = [
            t + [pad_token_id] * (self.max_length - len(t))
            for t in self.encoded_texts
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            torch.tensor(self.data.iloc[idx]["Label"], dtype=torch.long),
        )


# --------------- loss / accuracy helpers ---------------


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Cross-entropy on last-token logits for classification."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return nn.functional.cross_entropy(logits, target_batch)


def calc_loss_loader(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    total = 0.0
    n = len(loader)
    if n == 0:
        return float("nan")
    limit = n if num_batches is None else min(num_batches, n)
    for i, (inp, tgt) in enumerate(loader):
        if i >= limit:
            break
        total += calc_loss_batch(inp, tgt, model, device).item()
    return total / limit


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        tr = calc_loss_loader(train_loader, model, device, eval_iter)
        va = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return tr, va


def calc_accuracy_loader(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """Return accuracy (0.0–1.0) on last-token classification logits."""
    model.eval()
    correct, total = 0, 0
    n = len(loader)
    limit = n if num_batches is None else min(num_batches, n)
    for i, (inp, tgt) in enumerate(loader):
        if i >= limit:
            break
        inp, tgt = inp.to(device), tgt.to(device)
        with torch.no_grad():
            logits = model(inp)[:, -1, :]
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == tgt).sum().item()
        total += tgt.shape[0]
    model.train()
    return correct / total if total > 0 else 0.0


def collect_predictions_loader(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``model`` on *loader* (use ``shuffle=False`` for stable row order).

    Returns ``(preds, targets)`` as 1-D ``long`` tensors of equal length.
    """
    model.eval()
    preds_chunks: list[torch.Tensor] = []
    tgt_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for inp, tgt in loader:
            inp = inp.to(device)
            logits = model(inp)[:, -1, :]
            pr = torch.argmax(logits, dim=-1).cpu()
            preds_chunks.append(pr)
            tgt_chunks.append(tgt.long().cpu())
    model.train()
    return torch.cat(preds_chunks, dim=0), torch.cat(tgt_chunks, dim=0)


def confusion_counts_binary_spam(
    targets: torch.Tensor,
    preds: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Binary labels ``ham=0``, ``spam=1``. Returns ``(TN, FP, FN, TP)``.

    - TN: true ham, pred ham
    - FP: true ham, pred spam
    - FN: true spam, pred ham (missed spam)
    - TP: true spam, pred spam
    """
    t = targets.long().reshape(-1)
    p = preds.long().reshape(-1)
    tn = int(((t == 0) & (p == 0)).sum().item())
    fp = int(((t == 0) & (p == 1)).sum().item())
    fn = int(((t == 1) & (p == 0)).sum().item())
    tp = int(((t == 1) & (p == 1)).sum().item())
    return tn, fp, fn, tp


def accuracy_from_counts(tn: int, fp: int, fn: int, tp: int) -> float:
    den = tn + fp + fn + tp
    return float(tn + tp) / float(den) if den > 0 else 0.0


def prf1_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    """Precision / recall / F1 for ham (0) and spam (1)."""

    def safe_div(num: float, den: float) -> float:
        return float(num / den) if den > 0 else 0.0

    prec_spam = safe_div(tp, tp + fp)
    rec_spam = safe_div(tp, tp + fn)
    prec_ham = safe_div(tn, tn + fn)
    rec_ham = safe_div(tn, tn + fp)

    def f1(prec: float, rec: float) -> float:
        return 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "precision_spam": prec_spam,
        "recall_spam": rec_spam,
        "f1_spam": f1(prec_spam, rec_spam),
        "precision_ham": prec_ham,
        "recall_ham": rec_ham,
        "f1_ham": f1(prec_ham, rec_ham),
    }


def export_false_negative_spam_csv(
    dataset: SpamDataset,
    targets: torch.Tensor,
    preds: torch.Tensor,
    out_path: Path,
    *,
    max_rows: int | None = None,
) -> int:
    """Write rows where true label is spam (1) but prediction is ham (0).

    Columns: ``Index``, ``Label``, ``Pred``, ``Text``.
    Returns number of rows written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = targets.long().reshape(-1)
    p = preds.long().reshape(-1)
    n = min(len(t), len(dataset))
    rows_out = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Index", "Label", "Pred", "Text"])
        for i in range(n):
            if int(t[i].item()) != 1 or int(p[i].item()) != 0:
                continue
            text = str(dataset.data.iloc[i]["Text"])
            w.writerow([i, int(t[i].item()), int(p[i].item()), text])
            rows_out += 1
            if max_rows is not None and rows_out >= max_rows:
                break
    return rows_out


def format_classification_eval_lines(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    metrics: dict[str, float],
    *,
    split_name: str = "test",
) -> list[str]:
    """Human-readable lines for logging (confusion + PRF)."""
    acc = accuracy_from_counts(tn, fp, fn, tp)
    lines = [
        f"[{split_name}] Confusion matrix (true rows, pred columns; ham=0 spam=1)",
        "           pred_ham  pred_spam",
        f"  true_ham    {tn:5d}      {fp:5d}",
        f"  true_spam   {fn:5d}      {tp:5d}",
        f"[{split_name}] Accuracy (from counts): {acc * 100:.2f}%",
        f"[{split_name}] spam: P={metrics['precision_spam']:.4f} "
        f"R={metrics['recall_spam']:.4f} F1={metrics['f1_spam']:.4f}",
        f"[{split_name}] ham:  P={metrics['precision_ham']:.4f} "
        f"R={metrics['recall_ham']:.4f} F1={metrics['f1_ham']:.4f}",
    ]
    return lines
