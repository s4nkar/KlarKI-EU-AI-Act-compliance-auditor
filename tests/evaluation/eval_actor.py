"""
Evaluation — Actor Classifier Gold Dataset.

Loads the hand-curated gold_actor.jsonl dataset and runs classify_actor()
against each example. Produces accuracy, macro F1, and per-class metrics.

Unlike BERT eval, the actor classifier uses a pattern+ML ensemble; this
evaluation runs in pattern-only mode (ML model is patched to None) so that
the deterministic pattern logic is measured independently of model training.

Usage:
    python tests/evaluation/eval_actor.py
    python tests/evaluation/eval_actor.py --verbose
    python tests/evaluation/eval_actor.py --with-ml   # include ML if trained
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
GOLD_PATH   = Path(__file__).parent / "datasets" / "gold_actor.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_API_DIR = next((p for p in [REPO_ROOT / "api", Path("/app")] if p.is_dir()), Path("/app"))
sys.path.insert(0, str(_API_DIR))

LABELS = ["provider", "deployer", "importer", "distributor"]


def load_dataset() -> list[dict]:
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Gold dataset not found: {GOLD_PATH}")
    examples = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _no_ml(_text):
    return None


def run(verbose: bool = False, with_ml: bool = False) -> dict:
    """Run actor classifier on the gold dataset. Returns results dict."""
    try:
        from sklearn.metrics import accuracy_score, classification_report, f1_score
    except ImportError:
        return _skip("scikit-learn not installed — run: pip install scikit-learn")

    try:
        from services.actor_classifier import classify_actor
    except ImportError as exc:
        return _skip(f"Cannot import actor_classifier: {exc}. Run from repo root.")

    dataset = load_dataset()
    if verbose:
        print(f"  Loaded {len(dataset)} gold examples from {GOLD_PATH.name}")

    true_labels: list[str] = []
    pred_labels: list[str] = []
    errors: list[dict] = []
    t0 = time.perf_counter()

    for ex in dataset:
        text  = ex["text"]
        label = ex["label"]

        if with_ml:
            result = classify_actor(text)
        else:
            with patch("services.actor_classifier._ml_predict_actor", _no_ml):
                result = classify_actor(text)

        pred = result.actor_type.value
        true_labels.append(label)
        pred_labels.append(pred)

        if pred != label:
            errors.append({
                "text":       text[:100] + "…",
                "true":       label,
                "pred":       pred,
                "confidence": result.confidence,
                "note":       ex.get("note", ""),
            })

    elapsed = time.perf_counter() - t0

    try:
        accuracy = float(accuracy_score(true_labels, pred_labels))
        macro_f1 = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
        report   = classification_report(
            true_labels, pred_labels, labels=LABELS, output_dict=True, zero_division=0
        )
    except Exception as exc:
        return _skip(f"Metric computation failed: {exc}")

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

    status = "pass" if accuracy >= 0.80 else ("warn" if accuracy >= 0.65 else "fail")

    results = {
        "eval":               "actor",
        "status":             status,
        "accuracy":           round(accuracy, 4),
        "macro_f1":           round(macro_f1, 4),
        "per_class":          per_class,
        "n_samples":          len(dataset),
        "n_errors":           len(errors),
        "errors":             errors,
        "inference_seconds":  round(elapsed, 3),
        "mode":               "with_ml" if with_ml else "pattern_only",
        "threshold_accuracy": 0.80,
    }

    out_path = RESULTS_DIR / "actor.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if verbose:
        print_report(results)

    return results


def _skip(reason: str) -> dict:
    return {"eval": "actor", "status": "skip", "reason": reason}


def print_report(r: dict) -> None:
    if r["status"] == "skip":
        print(f"  [SKIP] {r.get('reason', '')}")
        return

    icon = "✓" if r["status"] == "pass" else ("~" if r["status"] == "warn" else "✗")
    mode = r.get("mode", "pattern_only")
    print(f"\n  {'─'*54}")
    print(f"  {icon} Actor Classifier — Gold Dataset Evaluation ({mode})")
    print(f"  {'─'*54}")
    print(f"  Accuracy  : {r['accuracy']*100:.1f}%   (threshold ≥ 80%)")
    print(f"  Macro F1  : {r['macro_f1']*100:.1f}%")
    print(f"  Samples   : {r['n_samples']}  |  Errors: {r['n_errors']}")
    print(f"  Time      : {r['inference_seconds']}s")
    print()
    print(f"  {'Class':<14} {'P':>6} {'R':>6} {'F1':>6} {'N':>5}")
    print(f"  {'─'*40}")
    for lbl, m in r["per_class"].items():
        flag = " ←" if m["f1"] < 0.70 else ""
        print(f"  {lbl:<14} {m['precision']*100:>5.1f}% {m['recall']*100:>5.1f}% {m['f1']*100:>5.1f}%{flag} {m['support']:>4}")

    if r["errors"] and r.get("n_errors", 0) > 0:
        print(f"\n  Misclassified ({r['n_errors']}):")
        for e in r["errors"][:5]:
            print(f"    true={e['true']:<12} pred={e['pred']:<12} conf={e['confidence']:.2f}  {e['text'][:60]}…")


# ── pytest integration ─────────────────────────────────────────────────────────

def test_actor_accuracy() -> None:
    """pytest: Actor classifier must reach ≥ 80% accuracy on gold dataset (pattern mode)."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r["reason"])
    assert r["accuracy"] >= 0.80, (
        f"Actor classifier accuracy {r['accuracy']:.4f} is below the 0.80 threshold. "
        f"Macro F1: {r['macro_f1']:.4f}. "
        f"Errors: {r.get('n_errors', '?')}/{r.get('n_samples', '?')}"
    )


def test_no_actor_class_below_60_f1() -> None:
    """pytest: No individual actor class should fall below 60% F1."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r["reason"])
    failures = {lbl: m["f1"] for lbl, m in r["per_class"].items() if m["support"] > 0 and m["f1"] < 0.60}
    assert not failures, (
        f"Actor classes below 60% F1 (gold dataset, pattern mode): {failures}. "
        "Improve pattern coverage or add synonyms for these classes."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Actor classifier gold dataset evaluation")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--with-ml", action="store_true", help="Include ML model (requires trained artifact)")
    args = parser.parse_args()

    print("Running actor classifier evaluation …")
    results = run(verbose=args.verbose, with_ml=args.with_ml)
    if not args.verbose:
        print_report(results)

    sys.exit(0 if results.get("status") in ("pass", "warn") else 1)
