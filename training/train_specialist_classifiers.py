#!/usr/bin/env python3
"""Fine-tune deepset/gbert-base for Phase 3 specialist EU AI Act classifiers.

Three classifier types, each a separate run:

  actor      — 4-class: provider / deployer / importer / distributor
               Input:  training/data/actor_labels.jsonl
               Output: training/actor_classifier/

  risk       — binary:  high_risk / not_high_risk (Article 6 + Annex III gate)
               Input:  training/data/risk_labels.jsonl
               Output: training/risk_classifier/

  prohibited — binary:  prohibited / not_prohibited (Article 5 gate)
               Input:  training/data/prohibited_labels.jsonl
               Output: training/prohibited_classifier/

Usage:
    python training/train_specialist_classifiers.py --type actor
    python training/train_specialist_classifiers.py --type risk --epochs 10
    python training/train_specialist_classifiers.py --type prohibited --batch-size 32

Hardware: GPU recommended; CPU fallback ~3× slower.
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainingArguments,
    TrainerState,
)

_RESET = "\033[0m"
_BOLD  = "\033[1m"
_GREEN = "\033[32m"
_AMBER = "\033[33m"
_RED   = "\033[31m"
_DIM   = "\033[2m"


def _c(col: str, txt: str) -> str:
    return f"{col}{txt}{_RESET}"


def _bar(current: int, total: int, width: int = 20) -> str:
    filled = int(width * current / max(total, 1))
    return f"[{'#' * filled}{'.' * (width - filled)}]"


# ── Classifier configurations ─────────────────────────────────────────────────

CLASSIFIER_CONFIGS: dict[str, dict] = {
    "actor": {
        "data_file": "training/data/actor_labels.jsonl",
        "output_dir": "training/actor_classifier",
        "labels": ["provider", "deployer", "importer", "distributor"],
        "description": "Article 3 actor role classifier (4-class)",
        "f1_target": 0.90,  # slightly lower target — importer/distributor are rare in practice
    },
    "risk": {
        "data_file": "training/data/risk_labels.jsonl",
        "output_dir": "training/risk_classifier",
        "labels": ["high_risk", "not_high_risk"],
        "description": "Article 6 + Annex III high-risk binary classifier",
        "f1_target": 0.93,
    },
    "prohibited": {
        "data_file": "training/data/prohibited_labels.jsonl",
        "output_dir": "training/prohibited_classifier",
        "labels": ["prohibited", "not_prohibited"],
        "description": "Article 5 prohibited-practice binary classifier",
        "f1_target": 0.95,  # higher bar — false negatives here are legally dangerous
    },
}

BASE_MODEL = "deepset/gbert-base"


# ── Progress callback (same as train_classifier.py) ───────────────────────────

class EpochProgressCallback(TrainerCallback):
    def __init__(self, total_epochs: int, f1_target: float) -> None:
        self._total = total_epochs
        self._f1_target = f1_target
        self._epoch_start: float = 0.0
        self._train_start: float = time.time()

    def on_epoch_begin(self, _args, _state, _control, **kwargs) -> None:
        self._epoch_start = time.time()

    def on_evaluate(
        self, _args: TrainingArguments, state: TrainerState, _control: TrainerControl,
        metrics: dict, **kwargs,
    ) -> None:
        epoch = int(state.epoch or 0)
        val_f1   = metrics.get("eval_macro_f1", 0.0)
        val_loss = metrics.get("eval_loss", 0.0)
        epoch_time    = time.time() - self._epoch_start
        total_elapsed = time.time() - self._train_start

        train_loss_entries = [
            e["loss"] for e in state.log_history
            if "loss" in e and "eval_loss" not in e
        ]
        train_loss = train_loss_entries[-1] if train_loss_entries else None

        bar = _bar(epoch, self._total)
        f1_colour = _GREEN if val_f1 >= self._f1_target else _AMBER if val_f1 >= 0.80 else _RED

        if train_loss is not None:
            gap = val_loss - train_loss
            loss_colour = _RED if gap > 0.3 else _AMBER if val_loss > 1.0 else _GREEN
            loss_str = _c(loss_colour, f"train={train_loss:.4f}  val={val_loss:.4f}  gap={gap:+.4f}")
        else:
            loss_str = f"val_loss={val_loss:.4f}"

        print(
            f"\n  {_c(_BOLD, f'Epoch {epoch}/{self._total}')}  {bar}"
            f"  {loss_str}  val_F1={_c(f1_colour, f'{val_f1:.4f}')}"
            + _c(_DIM, f"  ({epoch_time:.0f}s / {total_elapsed:.0f}s total)")
        )

        if val_f1 >= self._f1_target:
            print(_c(_GREEN, f"  [OK] Above {self._f1_target} production target"))
        elif val_f1 >= 0.80:
            print(_c(_AMBER, f"  [~] Above 0.80 but below {self._f1_target} production target"))
        elif epoch == self._total:
            print(_c(_RED, f"  [!!] Below 0.80 — consider more data or epochs"))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_dataset(records: list[dict], val_ratio: float = 0.15, seed: int = 42) -> tuple[list, list]:
    rng = random.Random(seed)
    by_label: dict[str, list] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)
    train, val = [], []
    for items in by_label.values():
        rng.shuffle(items)
        cut = max(1, int(len(items) * (1 - val_ratio)))
        train.extend(items[:cut])
        val.extend(items[cut:])
    return train, val


def tokenize(batch: dict, tokenizer, max_length: int) -> dict:
    return tokenizer(batch["text"], truncation=True, max_length=max_length)


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"macro_f1": f1}


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    classifier_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    seed: int,
) -> None:
    cfg = CLASSIFIER_CONFIGS[classifier_type]
    labels: list[str] = cfg["labels"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label  = {i: l for i, l in enumerate(labels)}
    f1_target: float = cfg["f1_target"]

    print(f"\n{'='*60}")
    print(f"  {cfg['description']}")
    print(f"  Labels: {labels}")
    print(f"  Data:   {cfg['data_file']}")
    print(f"  Output: {cfg['output_dir']}")
    print(f"{'='*60}\n")

    records = load_jsonl(cfg["data_file"])
    print(f"Loaded {len(records)} examples")

    # Filter out records with unknown labels
    records = [r for r in records if r["label"] in label2id]
    if not records:
        raise ValueError(f"No valid records found for labels {labels} in {cfg['data_file']}")

    train_data, val_data = split_dataset(records, seed=seed)
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    label_dist = Counter(r["label"] for r in records)
    print(f"  Label distribution: {dict(label_dist)}")

    for r in train_data + val_data:
        r["label"] = label2id[r["label"]]

    train_ds = Dataset.from_list(train_data)
    val_ds   = Dataset.from_list(val_data)

    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    train_ds = train_ds.map(lambda b: tokenize(b, tokenizer, max_length), batched=True)
    val_ds   = val_ds.map(  lambda b: tokenize(b, tokenizer, max_length), batched=True)
    train_ds = train_ds.rename_column("label", "labels")
    val_ds   = val_ds.rename_column(  "label", "labels")
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    model = BertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    model = model.to(device)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        seed=seed,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        save_total_limit=3,
        report_to="none",
    )

    # Inverse-frequency class weights, clipped to [0.5, 3.0]
    label_counts = Counter(r["label"] for r in train_data)
    total = sum(label_counts.values())
    raw_weights = [
        total / (len(labels) * max(label_counts.get(i, 1), 1))
        for i in range(len(labels))
    ]
    class_weights = torch.tensor(
        [min(max(w, 0.5), 3.0) for w in raw_weights], dtype=torch.float
    ).to(device)
    print(f"  Class weights: { {labels[i]: round(class_weights[i].item(), 3) for i in range(len(labels))} }")

    _SMOOTH = 0.1

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            lbs = inputs.pop("labels")
            outputs = model(**inputs)
            loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=_SMOOTH)(
                outputs.logits, lbs
            )
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=4),
            EpochProgressCallback(total_epochs=epochs, f1_target=f1_target),
        ],
    )

    print("\nStarting training…")
    trainer.train()

    results = trainer.evaluate()
    final_f1 = results["eval_macro_f1"]
    print(f"\nFinal validation macro F1: {final_f1:.4f}")

    if final_f1 < f1_target:
        print(f"WARNING: macro F1 below {f1_target} target — add more training data or increase epochs.")

    # Classification report + confusion matrix
    preds_out = trainer.predict(val_ds)
    preds  = np.argmax(preds_out.predictions, axis=-1)
    y_true = preds_out.label_ids
    print("\nClassification Report:")
    print(classification_report(y_true, preds, target_names=labels, zero_division=0))

    # Save model + tokenizer
    out_path = Path(cfg["output_dir"])
    out_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print(f"Model saved to {out_path}")

    # Persist metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, preds, labels=list(range(len(labels))), zero_division=0
    )
    cm = confusion_matrix(y_true, preds, labels=list(range(len(labels)))).tolist()
    per_class = [
        {
            "label": labels[i],
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(labels))
    ]
    metrics_payload = {
        "classifier_type": classifier_type,
        "macro_f1": round(float(final_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels,
        "val_size": len(val_data),
        "train_size": len(train_data),
        "base_model": BASE_MODEL,
    }
    metrics_path = out_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(f"\nNext: python scripts/export_onnx.py --model-path {cfg['output_dir']} "
          f"--output-path model_repository/{classifier_type}_classifier/1/model.onnx")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Phase 3 specialist classifier (actor / risk / prohibited)"
    )
    parser.add_argument(
        "--type", required=True, choices=list(CLASSIFIER_CONFIGS.keys()),
        help="Which specialist classifier to train",
    )
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Allow overriding base model globally
    global BASE_MODEL
    BASE_MODEL = args.base_model

    train(
        classifier_type=args.type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
