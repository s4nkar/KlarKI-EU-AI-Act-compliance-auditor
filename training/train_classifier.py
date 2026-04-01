#!/usr/bin/env python3
"""Fine-tune deepset/gbert-base for 8-class EU AI Act clause classification.

Trains a sequence classification model on clause_labels.jsonl.
Exports the trained model to ./bert_classifier/ for ONNX conversion.

Usage:
    pip install transformers datasets scikit-learn torch
    python training/train_classifier.py \
        --data training/data/clause_labels.jsonl \
        --output training/bert_classifier \
        --epochs 5 \
        --batch-size 16

Hardware:
    GPU recommended (RTX 3050 Ti sufficient). CPU-only training works but
    takes ~3x longer. The model is ~440 MB; fine-tuning peaks at ~3 GB VRAM.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    EarlyStoppingCallback,
)

# ── Colour helpers ─────────────────────────────────────────────────────────────
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_GREEN = "\033[32m"
_AMBER = "\033[33m"
_RED   = "\033[31m"
_CYAN  = "\033[36m"
_DIM   = "\033[2m"


def _c(col: str, txt: str) -> str:
    return f"{col}{txt}{_RESET}"


def _bar(current: int, total: int, width: int = 20) -> str:
    filled = int(width * current / max(total, 1))
    return f"[{'#' * filled}{'.' * (width - filled)}]"


class EpochProgressCallback(TrainerCallback):
    """Print a clear per-epoch summary line with F1 bar."""

    def __init__(self, total_epochs: int) -> None:
        self._total = total_epochs
        self._epoch_start: float = 0.0
        self._train_start: float = time.time()

    def on_epoch_begin(
        self, _args: TrainingArguments, _state: TrainerState, _control: TrainerControl, **kwargs
    ) -> None:
        self._epoch_start = time.time()

    def on_evaluate(
        self, _args: TrainingArguments, state: TrainerState, _control: TrainerControl,
        metrics: dict, **kwargs
    ) -> None:
        epoch = int(state.epoch or 0)
        f1 = metrics.get("eval_macro_f1", 0.0)
        loss = metrics.get("eval_loss", 0.0)
        epoch_time = time.time() - self._epoch_start
        total_elapsed = time.time() - self._train_start

        bar = _bar(epoch, self._total)
        f1_colour = _GREEN if f1 >= 0.85 else _AMBER if f1 >= 0.70 else _RED
        f1_str = _c(f1_colour, f"{f1:.4f}")

        print(
            f"\n  {_c(_BOLD, f'Epoch {epoch}/{self._total}')}  {bar}"
            f"  loss={loss:.4f}  macro_F1={f1_str}"
            + _c(_DIM, f"  ({epoch_time:.0f}s this epoch  /  {total_elapsed:.0f}s total)")
        )

        if f1 >= 0.85:
            print(_c(_GREEN, "  [OK] Above 0.85 F1 target"))
        elif epoch == self._total:
            print(_c(_AMBER, "  [!!] Below 0.85 F1 target -- consider more data or epochs"))

# ── Label definitions (must match ArticleDomain enum) ────────────────────────

LABELS = [
    "risk_management",
    "data_governance",
    "technical_documentation",
    "record_keeping",
    "transparency",
    "human_oversight",
    "security",
    "unrelated",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

BASE_MODEL = "deepset/gbert-base"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    """Load JSONL training data."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_dataset(records: list[dict], val_ratio: float = 0.15) -> tuple[list, list]:
    """Stratified train/val split."""
    from collections import defaultdict
    import random

    by_label: dict[str, list] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    train, val = [], []
    for items in by_label.values():
        random.shuffle(items)
        cut = max(1, int(len(items) * (1 - val_ratio)))
        train.extend(items[:cut])
        val.extend(items[cut:])

    return train, val


# ── Tokenisation ──────────────────────────────────────────────────────────────

def tokenize(batch: dict, tokenizer, max_length: int = 256) -> dict:
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred) -> dict:
    import numpy as np
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="macro")
    return {"macro_f1": f1}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune gbert-base for EU AI Act clause classification")
    parser.add_argument("--data", default="training/data/clause_labels.jsonl")
    parser.add_argument("--output", default="training/bert_classifier")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Max epochs — early stopping triggers before this (default: 10)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Per-device batch size (effective batch = 2x via grad accumulation)")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Peak learning rate with cosine decay (default: 3e-5)")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max token length for tokeniser (default: 256)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading data from {args.data}")
    records = load_jsonl(args.data)
    print(f"  Loaded {len(records)} examples across {len(LABELS)} classes")

    train_data, val_data = split_dataset(records)
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # Convert labels to IDs
    for r in train_data + val_data:
        r["label"] = LABEL2ID[r["label"]]

    train_ds = Dataset.from_list(train_data)
    val_ds   = Dataset.from_list(val_data)

    print(f"\nLoading tokenizer from {args.base_model}")
    tokenizer = BertTokenizer.from_pretrained(args.base_model)

    train_ds = train_ds.map(lambda b: tokenize(b, tokenizer, args.max_length), batched=True)
    val_ds   = val_ds.map(  lambda b: tokenize(b, tokenizer, args.max_length), batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds   = val_ds.rename_column(  "label", "labels")
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(  "torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"\nLoading model {args.base_model}")
    model = BertForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    model = model.to(device)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # Gradient accumulation: effective batch = batch_size * 2 (e.g. 32 with bs=16)
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        weight_decay=0.01,
        # Cosine schedule with 6% warmup — smoother decay than linear, better for ~10 epochs
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        # Label smoothing handles noise in LLM-generated training data (0.1 = 10%)
        label_smoothing_factor=0.1,
        # Windows-safe: no multiprocessing in DataLoader
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            EpochProgressCallback(total_epochs=args.epochs),
        ],
    )

    print("\nStarting training…")
    trainer.train()

    # Final evaluation
    results = trainer.evaluate()
    print(f"\nFinal validation macro F1: {results['eval_macro_f1']:.4f}")

    if results["eval_macro_f1"] < 0.85:
        print("WARNING: macro F1 below 0.85 target — consider more training data or epochs.")

    # Full classification report
    import numpy as np
    from sklearn.metrics import confusion_matrix
    preds_output = trainer.predict(val_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=LABELS))

    # Save model + tokenizer
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"\nModel saved to {output_path}")

    # Persist metrics as JSON for the dashboard endpoint
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(LABELS))), zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(LABELS)))).tolist()

    per_class = [
        {
            "label": LABELS[i],
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(LABELS))
    ]
    macro_f1 = round(float(results["eval_macro_f1"]), 4)

    metrics_payload = {
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": LABELS,
        "val_size": len(val_data),
        "train_size": len(train_data),
        "base_model": args.base_model,
    }
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print("Next step: python scripts/export_onnx.py --model-path training/bert_classifier \\")
    print("           --output-path model_repository/bert_clause_classifier/1/model.onnx")


if __name__ == "__main__":
    main()
