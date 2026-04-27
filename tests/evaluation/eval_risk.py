"""
Evaluation — Risk Classifier Gold Dataset.

Loads gold_risk.jsonl and runs predict_high_risk() directly against each
example to measure how well the trained ML classifier generalises to
realistic policy-document language (NOT the regulatory-excerpt style used
in training data).

Metrics:
  Accuracy           : fraction of examples correctly classified
  High-risk recall   : fraction of high_risk examples correctly identified
  High-risk precision: fraction of predicted high_risk that are truly high_risk
  Specificity (TNR)  : fraction of not_high_risk correctly identified
  F1 (high_risk)     : harmonic mean of precision and recall for the positive class

Thresholds:
  Accuracy     >= 0.80  (warn below, fail below 0.60)
  Recall       >= 0.75  (missing a high-risk system is the critical error)

Skipped automatically if the trained risk classifier model is not found.

Usage:
    python tests/evaluation/eval_risk.py
    python tests/evaluation/eval_risk.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
GOLD_PATH   = Path(__file__).parent / "datasets" / "gold_risk.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_API_DIR = next((p for p in [REPO_ROOT / "api", Path("/app")] if p.is_dir()), Path("/app"))
sys.path.insert(0, str(_API_DIR))


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


def run(verbose: bool = False) -> dict:
    """Run risk classifier on the gold dataset. Returns results dict."""
    try:
        from services.ml_classifiers import predict_high_risk
    except ImportError as exc:
        return _skip(f"Cannot import ml_classifiers: {exc}. Run from repo root.")

    dataset = load_dataset()
    if verbose:
        print(f"  Loaded {len(dataset)} gold examples from {GOLD_PATH.name}")

    # Probe: if model is not trained, predict_high_risk returns None
    probe = predict_high_risk("test")
    if probe is None:
        return _skip(
            "Risk classifier model not trained — run ./run.sh setup or "
            "train-specialist stage to generate the model."
        )

    true_labels: list[str] = []
    pred_labels: list[str] = []
    errors: list[dict] = []
    confidences: list[float] = []
    t0 = time.perf_counter()

    for ex in dataset:
        text  = ex["text"]
        label = ex["label"]  # "high_risk" | "not_high_risk"

        result = predict_high_risk(text[:2000])

        if result is None:
            # Model returned None mid-run — treat as not_high_risk
            pred = "not_high_risk"
            conf = 0.0
        else:
            pred = result.label
            conf = result.confidence

        true_labels.append(label)
        pred_labels.append(pred)
        confidences.append(conf)

        if pred != label:
            errors.append({
                "text":       text[:120] + "…",
                "true":       label,
                "pred":       pred,
                "confidence": conf,
                "note":       ex.get("note", ""),
            })

    elapsed = time.perf_counter() - t0

    n = len(dataset)
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == "high_risk"     and p == "high_risk")
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == "not_high_risk" and p == "high_risk")
    tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == "not_high_risk" and p == "not_high_risk")
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == "high_risk"     and p == "not_high_risk")

    def _sdiv(a, b):
        return round(a / b, 4) if b > 0 else 0.0

    accuracy  = _sdiv(tp + tn, n)
    recall    = _sdiv(tp, tp + fn)   # high_risk recall (sensitivity)
    precision = _sdiv(tp, tp + fp)
    tnr       = _sdiv(tn, tn + fp)   # specificity
    f1        = _sdiv(2 * precision * recall, precision + recall)
    avg_conf  = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    n_high_risk     = sum(1 for ex in dataset if ex["label"] == "high_risk")
    n_not_high_risk = sum(1 for ex in dataset if ex["label"] == "not_high_risk")

    if accuracy >= 0.80 and recall >= 0.75:
        status = "pass"
    elif accuracy >= 0.65 or recall >= 0.60:
        status = "warn"
    else:
        status = "fail"

    results = {
        "eval":               "risk",
        "status":             status,
        "accuracy":           accuracy,
        "recall":             recall,
        "precision":          precision,
        "tnr":                tnr,
        "f1":                 f1,
        "avg_confidence":     avg_conf,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_samples":          n,
        "n_high_risk":        n_high_risk,
        "n_not_high_risk":    n_not_high_risk,
        "n_errors":           len(errors),
        "errors":             errors,
        "inference_seconds":  round(elapsed, 3),
        "threshold_accuracy": 0.80,
        "threshold_recall":   0.75,
    }

    out_path = RESULTS_DIR / "risk.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if verbose:
        print_report(results)

    return results


def _skip(reason: str) -> dict:
    return {"eval": "risk", "status": "skip", "reason": reason}


def print_report(r: dict) -> None:
    if r["status"] == "skip":
        print(f"  [SKIP] {r.get('reason', '')}")
        return

    icon = "✓" if r["status"] == "pass" else ("~" if r["status"] == "warn" else "✗")
    print(f"\n  {'─'*56}")
    print(f"  {icon} Risk Classifier — Gold Dataset Evaluation")
    print(f"  {'─'*56}")
    print(f"  Accuracy         : {r['accuracy']*100:.1f}%   (threshold ≥ 80%)")
    print(f"  Recall (high_risk): {r['recall']*100:.1f}%   (threshold ≥ 75%)")
    print(f"  Precision        : {r['precision']*100:.1f}%")
    print(f"  Specificity (TNR): {r['tnr']*100:.1f}%")
    print(f"  F1 (high_risk)   : {r['f1']*100:.1f}%")
    print(f"  Avg confidence   : {r['avg_confidence']:.3f}")
    print(f"  Samples          : {r['n_samples']}  "
          f"(high_risk={r['n_high_risk']}, not_high_risk={r['n_not_high_risk']})")
    print(f"  TP={r['tp']}  FP={r['fp']}  TN={r['tn']}  FN={r['fn']}")
    print(f"  Time             : {r['inference_seconds']}s")

    if r["errors"]:
        fn_errors = [e for e in r["errors"] if e["true"] == "high_risk"]
        fp_errors = [e for e in r["errors"] if e["true"] == "not_high_risk"]
        if fn_errors:
            print(f"\n  False Negatives — missed high-risk ({len(fn_errors)}):")
            for e in fn_errors[:4]:
                print(f"    conf={e['confidence']:.2f}  {e['text'][:80]}…")
        if fp_errors:
            print(f"\n  False Positives — incorrectly flagged ({len(fp_errors)}):")
            for e in fp_errors[:4]:
                print(f"    conf={e['confidence']:.2f}  {e['text'][:80]}…")


# ── pytest integration ─────────────────────────────────────────────────────────

def test_risk_accuracy() -> None:
    """pytest: Risk classifier must reach ≥ 80% accuracy on gold dataset."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r["reason"])
    assert r["accuracy"] >= 0.80, (
        f"Risk classifier accuracy {r['accuracy']:.4f} is below 0.80. "
        f"Errors: {r.get('n_errors', '?')}/{r.get('n_samples', '?')}. "
        "The model may be overfitting to regulatory-style training text."
    )


def test_risk_high_risk_recall() -> None:
    """pytest: High-risk recall must be ≥ 75% — missing a high-risk system is a critical gap."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r["reason"])
    assert r["recall"] >= 0.75, (
        f"High-risk recall {r['recall']:.4f} is below 0.75. "
        f"FN={r['fn']} — the classifier is missing high-risk systems in policy document language. "
        "The model may have learned regulatory-text surface patterns rather than semantic risk signals. "
        "Add more policy-document-style examples to training data."
    )


def test_risk_specificity() -> None:
    """pytest: Specificity (TNR) must be ≥ 0.70 — too many false positives create audit noise."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r["reason"])
    assert r["tnr"] >= 0.70, (
        f"Risk classifier specificity {r['tnr']:.4f} is below 0.70. "
        f"FP={r['fp']} — too many non-high-risk systems are being flagged. "
        "Review the not_high_risk training examples for quality and diversity."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk classifier gold dataset evaluation")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("Running risk classifier evaluation …")
    results = run(verbose=args.verbose)
    if not args.verbose:
        print_report(results)

    sys.exit(0 if results.get("status") in ("pass", "warn") else 1)
