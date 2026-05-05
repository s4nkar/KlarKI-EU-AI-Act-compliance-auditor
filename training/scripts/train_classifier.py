#!/usr/bin/env python3
"""Fine-tune deepset/gbert-base for 8-class EU AI Act clause classification.

Trains a sequence classification model on clause_labels.jsonl.
Exports the trained model to ./bert_classifier/ for ONNX conversion.

Usage:
    pip install transformers datasets scikit-learn torch
    python training/train_classifier.py \
        --data training/data/clause_labels.jsonl \
        --output training/artifacts/bert_classifier \
        --epochs 5 \
        --batch-size 16

Hardware:
    GPU recommended (RTX 3050 Ti sufficient). CPU-only training works but
    takes ~3x longer. The model is ~440 MB; fine-tuning peaks at ~3 GB VRAM.
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    EarlyStoppingCallback,
)

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
        val_f1   = metrics.get("eval_macro_f1", 0.0)
        val_loss = metrics.get("eval_loss", 0.0)
        epoch_time    = time.time() - self._epoch_start
        total_elapsed = time.time() - self._train_start

        # Pull the most recent training loss from log history (logged every N steps)
        train_loss_entries = [
            e["loss"] for e in state.log_history
            if "loss" in e and "eval_loss" not in e
        ]
        train_loss = train_loss_entries[-1] if train_loss_entries else None

        bar = _bar(epoch, self._total)
        f1_colour = _GREEN if val_f1 >= 0.85 else _AMBER if val_f1 >= 0.70 else _RED

        # Build side-by-side loss display to reveal overfitting/underfitting
        if train_loss is not None:
            gap = val_loss - train_loss
            if gap > 0.3:
                loss_colour = _RED    # val >> train → overfitting
            elif val_loss > 1.0:
                loss_colour = _AMBER  # both high → underfitting
            else:
                loss_colour = _GREEN
            loss_str = _c(loss_colour, f"train={train_loss:.4f}  val={val_loss:.4f}  gap={gap:+.4f}")
        else:
            loss_str = f"val_loss={val_loss:.4f}"

        print(
            f"\n  {_c(_BOLD, f'Epoch {epoch}/{self._total}')}  {bar}"
            f"  {loss_str}  val_F1={_c(f1_colour, f'{val_f1:.4f}')}"
            + _c(_DIM, f"  ({epoch_time:.0f}s / {total_elapsed:.0f}s total)")
        )

        # Bias/variance diagnosis
        if train_loss is not None:
            gap = val_loss - train_loss
            if gap > 0.3:
                print(_c(_RED,   "  [!!] Large train/val loss gap — model may be OVERFITTING"))
            elif val_loss > 1.0 and val_f1 < 0.60:
                print(_c(_AMBER, "  [!]  High loss on both sets — model may be UNDERFITTING"))

        if val_f1 >= 0.93:
            print(_c(_GREEN, "  [OK] Above 0.93 production target"))
        elif val_f1 >= 0.85:
            print(_c(_AMBER, "  [~] Above 0.85 but below 0.93 production target"))
        elif epoch == self._total:
            print(_c(_RED, "  [!!] Below 0.85 — consider more data or epochs"))

# Label order must match ArticleDomain enum in api/models/schemas.py.
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


def split_dataset(records: list[dict], val_ratio: float = 0.15, seed: int = 42) -> tuple[list, list]:
    """Stratified train/val split by (label, source) — reproducible with seed.

    Records are grouped by (label, source) before splitting so val gets a
    proportional mix of regulatory and synthetic examples. With label-only
    stratification, regulatory records would scatter randomly into train and
    val, letting the model memorise synthetic-distribution phrasing while
    val numbers stay artificially high. Pairing on source forces every
    (label, source) bucket to contribute to val in the same ratio, which
    means regulatory examples reliably appear on the val side and the
    `macro_f1_regulatory_only` number computed at training-end actually
    reflects out-of-distribution performance on real text.
    """
    from collections import defaultdict
    rng = random.Random(seed)

    by_label_source: dict[tuple[str, str], list] = defaultdict(list)
    for r in records:
        key = (r["label"], r.get("source", "generated"))
        by_label_source[key].append(r)

    train, val = [], []
    for items in by_label_source.values():
        rng.shuffle(items)
        cut = max(1, int(len(items) * (1 - val_ratio)))
        train.extend(items[:cut])
        val.extend(items[cut:])

    return train, val


# ── Tokenisation ──────────────────────────────────────────────────────────────

def tokenize(batch: dict, tokenizer, max_length: int = 256) -> dict:
    # No padding here — DataCollatorWithPadding pads each batch to its own max
    # length, saving ~30-40% memory vs padding every sequence to 256.
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="macro")
    return {"macro_f1": f1}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune gbert-base for EU AI Act clause classification")
    parser.add_argument("--data", default="training/data/clause_labels.jsonl")
    parser.add_argument("--output", default="training/artifacts/bert_classifier")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=12,
                        help="Max epochs — early stopping triggers before this (default: 12)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Per-device batch size (effective batch = 2x via grad accumulation)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Peak learning rate with cosine decay (default: 2e-5)")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max token length for tokeniser (default: 256)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-version", default=None,
                        help="Data version tag to record in model registry (e.g. 'v2')")
    args = parser.parse_args()

    print(f"Loading data from {args.data}")
    records = load_jsonl(args.data)
    print(f"  Loaded {len(records)} examples across {len(LABELS)} classes")

    train_data, val_data = split_dataset(records, seed=args.seed)
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
    train_ds.set_format("torch")
    val_ds.set_format(  "torch")

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
        # Windows-safe: no multiprocessing in DataLoader
        dataloader_num_workers=0,
        # Keep only the 3 best checkpoints to avoid disk bloat
        save_total_limit=3,
        report_to="none",
    )

    # Class weights: inverse frequency from raw training list (before torch conversion).
    # Clipped to [0.5, 2.5] to avoid extreme values on balanced datasets.
    label_counts = Counter(r["label"] for r in train_data)  # int labels post-conversion
    total_samples = sum(label_counts.values())
    raw_weights = [
        total_samples / (len(LABELS) * max(label_counts.get(i, 1), 1))
        for i in range(len(LABELS))
    ]
    class_weights = torch.tensor(
        [min(max(w, 0.5), 2.5) for w in raw_weights], dtype=torch.float
    ).to(device)
    print(f"  Class weights: { {LABELS[i]: round(class_weights[i].item(), 3) for i in range(len(LABELS))} }")

    # Label smoothing constant — applied manually since we override compute_loss.
    # Smoothing reduces overconfidence on LLM-generated (noisy) training data.
    _SMOOTH = 0.1

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            # Weighted cross-entropy + manual label smoothing
            ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=_SMOOTH)(logits, labels)
            return (ce_loss, outputs) if return_outputs else ce_loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=4),
            EpochProgressCallback(total_epochs=args.epochs),
        ],
    )

    print("\nStarting training…")
    trainer.train()

    # Final evaluation
    results = trainer.evaluate()
    print(f"\nFinal validation macro F1: {results['eval_macro_f1']:.4f}")

    if results["eval_macro_f1"] < 0.93:
        print("WARNING: macro F1 below 0.93 production target — consider more training data or epochs.")

    # Full classification report
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

    # Regulatory-only macro F1 — honest signal on real (non-synthetic) text.
    # A wide gap to the mixed-set macro_f1 means the model is overfitting to
    # generator-specific phrasing rather than learning the legal distinctions.
    reg_mask = [i for i, r in enumerate(val_data) if r.get("source") == "regulatory"]
    if reg_mask:
        reg_labels = labels[reg_mask]
        reg_preds = preds[reg_mask]
        reg_macro_f1 = round(float(f1_score(reg_labels, reg_preds, average="macro", zero_division=0)), 4)
        reg_n = len(reg_mask)
    else:
        reg_macro_f1 = None
        reg_n = 0

    metrics_payload = {
        "macro_f1": macro_f1,
        "macro_f1_regulatory_only": reg_macro_f1,
        "regulatory_val_size": reg_n,
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": LABELS,
        "val_size": len(val_data),
        "train_size": len(train_data),
        "base_model": args.base_model,
    }

    if reg_macro_f1 is not None:
        gap = macro_f1 - reg_macro_f1
        gap_colour = _RED if gap > 0.15 else _AMBER if gap > 0.08 else _GREEN
        print(f"\nMixed val macro F1   : {macro_f1:.4f}")
        print(f"Regulatory-only F1   : {_c(gap_colour, f'{reg_macro_f1:.4f}')}  "
              f"(gap={_c(gap_colour, f'{gap:+.4f}')}, n={reg_n})")
        if gap > 0.15:
            print(_c(_RED, "  [!!] Large gap — model is overfitting to synthetic distribution"))
    else:
        print("\nNo regulatory examples in val set — cannot compute honest F1")
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Version management — snapshot this trained model and promote if best
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from version_manager import VersionManager
        vm = VersionManager()
        data_ver = args.data_version if hasattr(args, "data_version") else None
        new_ver = vm.save_and_promote("bert", output_path, metrics_payload, data_version=data_ver)
        print(f"Version recorded: bert@{new_ver}")
    except Exception as ve:
        print(f"[version] Skipped (version_manager error): {ve}")

    print("Next step: python scripts/export_onnx.py --model-path training/artifacts/bert_classifier \\")
    print("           --output-path model_repository/bert_clause_classifier/1/model.onnx")


if __name__ == "__main__":
    main()
