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
import os
from pathlib import Path

import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

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

def tokenize(batch: dict, tokenizer) -> dict:
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
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
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    train_ds = train_ds.map(lambda b: tokenize(b, tokenizer), batched=True)
    val_ds   = val_ds.map(  lambda b: tokenize(b, tokenizer), batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds   = val_ds.rename_column(  "label", "labels")
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(  "torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"\nLoading model {args.base_model}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    model = model.to(device)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
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
    print("Next step: python scripts/export_onnx.py --model-path training/bert_classifier \\")
    print("           --output-path model_repository/bert_clause_classifier/1/model.onnx")


if __name__ == "__main__":
    main()
