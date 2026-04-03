"""
Evaluation 1 — Gold Dataset Classification.

Loads the 80-example hand-labeled gold dataset and runs the trained BERT
model directly (no API required).  Produces accuracy, macro F1, and
per-class precision/recall/F1.

Usage:
    python tests/evaluation/eval_classifier.py
    python tests/evaluation/eval_classifier.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
GOLD_PATH   = Path(__file__).parent / "datasets" / "gold_classifier.jsonl"
MODEL_PATH  = next(
    (p for p in [REPO_ROOT / "training" / "bert_classifier", Path("/training/bert_classifier")] if p.exists()),
    Path("/training/bert_classifier"),
)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LABELS = [
    "risk_management", "data_governance", "technical_documentation",
    "record_keeping", "transparency", "human_oversight", "security", "unrelated",
]


# ── helpers ────────────────────────────────────────────────────────────────

def load_gold_dataset() -> list[dict]:
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Gold dataset not found: {GOLD_PATH}")
    examples = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def run(verbose: bool = False) -> dict:
    """Run BERT classifier on the gold dataset. Returns a results dict."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return _skip("transformers/torch not installed — run: pip install transformers torch")

    try:
        from sklearn.metrics import (
            accuracy_score, classification_report, f1_score,
        )
    except ImportError:
        return _skip("scikit-learn not installed — run: pip install scikit-learn")

    if not MODEL_PATH.exists():
        return _skip(f"BERT model not found at {MODEL_PATH}. Run ./run.sh setup first.")

    # ── load model ─────────────────────────────────────────────────────────
    if verbose:
        print(f"  Loading BERT model from {MODEL_PATH} …")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.eval()
    id2label: dict[int, str] = model.config.id2label  # type: ignore[assignment]

    # ── load data ──────────────────────────────────────────────────────────
    dataset     = load_gold_dataset()
    texts       = [ex["text"]  for ex in dataset]
    true_labels = [ex["label"] for ex in dataset]

    if verbose:
        print(f"  Running inference on {len(texts)} examples (batch_size=16) …")

    # ── batch inference ────────────────────────────────────────────────────
    pred_labels: list[str] = []
    batch_size  = 16
    t0          = time.perf_counter()

    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).tolist()
        pred_labels.extend(id2label[p] for p in preds)

    elapsed = time.perf_counter() - t0

    # ── metrics ────────────────────────────────────────────────────────────
    accuracy  = float(accuracy_score(true_labels, pred_labels))
    macro_f1  = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
    report    = classification_report(
        true_labels, pred_labels,
        labels=LABELS, output_dict=True, zero_division=0,
    )

    per_class = {
        lbl: {
            "precision": round(report[lbl]["precision"], 4),
            "recall":    round(report[lbl]["recall"],    4),
            "f1":        round(report[lbl]["f1-score"],  4),
            "support":   int(report[lbl]["support"]),
        }
        for lbl in LABELS
        if lbl in report
    }

    # ── wrong predictions for diagnostics ─────────────────────────────────
    errors = [
        {"text": texts[i][:80] + "…", "true": true_labels[i], "pred": pred_labels[i]}
        for i in range(len(texts))
        if true_labels[i] != pred_labels[i]
    ]

    results = {
        "eval":              "classifier",
        "status":            "pass" if macro_f1 >= 0.85 else "warn",
        "accuracy":          round(accuracy, 4),
        "macro_f1":          round(macro_f1, 4),
        "per_class":         per_class,
        "n_samples":         len(dataset),
        "n_errors":          len(errors),
        "errors":            errors,
        "inference_seconds": round(elapsed, 3),
        "threshold_macro_f1": 0.85,
    }

    # ── persist ────────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "classifier.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def _skip(reason: str) -> dict:
    return {"eval": "classifier", "status": "skip", "reason": reason}


def print_report(r: dict) -> None:
    if r["status"] == "skip":
        print(f"  [SKIP] {r['reason']}")
        return

    status_icon = "✓" if r["status"] == "pass" else "✗"
    print(f"\n  {'─'*52}")
    print(f"  {status_icon} Classifier — Gold Dataset Evaluation")
    print(f"  {'─'*52}")
    print(f"  Accuracy  : {r['accuracy']*100:.1f}%")
    print(f"  Macro F1  : {r['macro_f1']*100:.1f}%   (threshold ≥ 85%)")
    print(f"  Samples   : {r['n_samples']}  |  Errors: {r['n_errors']}")
    print(f"  Time      : {r['inference_seconds']}s")
    print()
    print(f"  {'Class':<28} {'P':>6} {'R':>6} {'F1':>6} {'N':>5}")
    print(f"  {'─'*54}")
    for lbl, m in r["per_class"].items():
        flag = " ←" if m["f1"] < 0.80 else ""
        print(f"  {lbl.replace('_', ' '):<28} {m['precision']*100:>5.1f}% {m['recall']*100:>5.1f}% {m['f1']*100:>5.1f}%{flag} {m['support']:>4}")


# ── pytest integration ─────────────────────────────────────────────────────

def test_classifier_accuracy() -> None:
    """pytest: BERT classifier must reach ≥ 85 % macro F1 on gold dataset."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r["reason"])
    assert r["macro_f1"] >= 0.85, (
        f"Macro F1 {r['macro_f1']:.4f} is below the 0.85 threshold. "
        f"Accuracy: {r['accuracy']:.4f}. Re-train with python training/train_classifier.py"
    )


def test_no_class_below_75_f1() -> None:
    """pytest: No individual class should fall below 75 % F1."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r["reason"])
    failures = {lbl: m["f1"] for lbl, m in r["per_class"].items() if m["f1"] < 0.75}
    assert not failures, f"Classes below 75% F1: {failures}"


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold dataset classifier evaluation")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("Running classifier evaluation …")
    results = run(verbose=args.verbose)
    print_report(results)

    sys.exit(0 if results.get("status") != "fail" else 1)
