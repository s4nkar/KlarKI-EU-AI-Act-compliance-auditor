"""
Evaluation — Prohibited Classifier Gold Dataset.

Loads gold_prohibited.jsonl and runs predict_prohibited() directly against
each example to measure how well the trained ML classifier generalises to
realistic policy-document language — particularly the hard boundary between
genuinely prohibited Article 5 practices and AI use cases that sound
concerning but are NOT prohibited (high-risk, GDPR-adjacent, or simply
privacy-sensitive but lawful).

Metrics:
  Accuracy              : fraction of examples correctly classified
  Prohibited recall     : fraction of prohibited examples correctly identified
  Prohibited precision  : fraction of predicted prohibited that are truly prohibited
  Specificity (TNR)     : fraction of not_prohibited correctly identified
  F1 (prohibited)       : harmonic mean of precision and recall for the positive class

Thresholds:
  Accuracy     >= 0.80  (warn below, fail below 0.60)
  Recall       >= 0.85  (missing a prohibited use is the most critical error)
  Specificity  >= 0.75  (false positives on lawful AI undermine trust)

Skipped automatically if the trained prohibited classifier model is not found.

Usage:
    python tests/evaluation/eval_prohibited.py
    python tests/evaluation/eval_prohibited.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
GOLD_PATH   = Path(__file__).parent / "datasets" / "gold_prohibited.jsonl"
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
    """Run prohibited classifier on the gold dataset. Returns results dict."""
    try:
        from services.ml_classifiers import predict_prohibited
    except ImportError as exc:
        return _skip(f"Cannot import ml_classifiers: {exc}. Run from repo root.")

    dataset = load_dataset()
    if verbose:
        print(f"  Loaded {len(dataset)} gold examples from {GOLD_PATH.name}")

    probe = predict_prohibited("test")
    if probe is None:
        return _skip(
            "Prohibited classifier model not trained — run ./run.sh setup or "
            "train-specialist stage to generate the model."
        )

    true_labels: list[str] = []
    pred_labels: list[str] = []
    errors: list[dict] = []
    confidences: list[float] = []
    t0 = time.perf_counter()

    # Per-article-5-type breakdown
    article5_types = {
        "subliminal":    {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "vulnerability": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "social_scoring":{"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "biometric_rt":  {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "emotion":       {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }

    for ex in dataset:
        text  = ex["text"]
        label = ex["label"]  # "prohibited" | "not_prohibited"
        note  = ex.get("note", "")

        result = predict_prohibited(text[:2000])

        if result is None:
            pred = "not_prohibited"
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
                "note":       note,
            })

        # Tag by Article 5 sub-type for diagnostics
        note_lc = note.lower()
        for tag, key in [
            ("subliminal",         "subliminal"),
            ("vulnerability",      "vulnerability"),
            ("social scoring",     "social_scoring"),
            ("biometric",          "biometric_rt"),
            ("emotion recognition","emotion"),
        ]:
            if tag in note_lc:
                bucket = article5_types[key]
                if label == "prohibited" and pred == "prohibited":
                    bucket["tp"] += 1
                elif label == "prohibited" and pred != "prohibited":
                    bucket["fn"] += 1
                elif label != "prohibited" and pred == "prohibited":
                    bucket["fp"] += 1
                else:
                    bucket["tn"] += 1

    elapsed = time.perf_counter() - t0

    n = len(dataset)
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == "prohibited"     and p == "prohibited")
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == "not_prohibited" and p == "prohibited")
    tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == "not_prohibited" and p == "not_prohibited")
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == "prohibited"     and p == "not_prohibited")

    def _sdiv(a, b):
        return round(a / b, 4) if b > 0 else 0.0

    accuracy  = _sdiv(tp + tn, n)
    recall    = _sdiv(tp, tp + fn)
    precision = _sdiv(tp, tp + fp)
    tnr       = _sdiv(tn, tn + fp)
    f1        = _sdiv(2 * precision * recall, precision + recall)
    avg_conf  = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    n_prohibited     = sum(1 for ex in dataset if ex["label"] == "prohibited")
    n_not_prohibited = sum(1 for ex in dataset if ex["label"] == "not_prohibited")

    if accuracy >= 0.80 and recall >= 0.85 and tnr >= 0.75:
        status = "pass"
    elif accuracy >= 0.65 or recall >= 0.70:
        status = "warn"
    else:
        status = "fail"

    # Compute per-type recall
    type_summary = {}
    for key, counts in article5_types.items():
        t_tp, t_fn = counts["tp"], counts["fn"]
        type_summary[key] = {
            "recall": _sdiv(t_tp, t_tp + t_fn),
            "tp": t_tp, "fp": counts["fp"], "tn": counts["tn"], "fn": t_fn,
        }

    results = {
        "eval":               "prohibited",
        "status":             status,
        "accuracy":           accuracy,
        "recall":             recall,
        "precision":          precision,
        "tnr":                tnr,
        "f1":                 f1,
        "avg_confidence":     avg_conf,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_samples":          n,
        "n_prohibited":       n_prohibited,
        "n_not_prohibited":   n_not_prohibited,
        "n_errors":           len(errors),
        "errors":             errors,
        "by_type":            type_summary,
        "inference_seconds":  round(elapsed, 3),
        "threshold_accuracy": 0.80,
        "threshold_recall":   0.85,
        "threshold_tnr":      0.75,
    }

    out_path = RESULTS_DIR / "prohibited.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if verbose:
        print_report(results)

    return results


def _skip(reason: str) -> dict:
    return {"eval": "prohibited", "status": "skip", "reason": reason}


def print_report(r: dict) -> None:
    if r["status"] == "skip":
        print(f"  [SKIP] {r.get('reason', '')}")
        return

    icon = "✓" if r["status"] == "pass" else ("~" if r["status"] == "warn" else "✗")
    print(f"\n  {'─'*56}")
    print(f"  {icon} Prohibited Classifier — Gold Dataset Evaluation")
    print(f"  {'─'*56}")
    print(f"  Accuracy           : {r['accuracy']*100:.1f}%   (threshold ≥ 80%)")
    print(f"  Recall (prohibited): {r['recall']*100:.1f}%   (threshold ≥ 85%)")
    print(f"  Precision          : {r['precision']*100:.1f}%")
    print(f"  Specificity (TNR)  : {r['tnr']*100:.1f}%   (threshold ≥ 75%)")
    print(f"  F1 (prohibited)    : {r['f1']*100:.1f}%")
    print(f"  Avg confidence     : {r['avg_confidence']:.3f}")
    print(f"  Samples            : {r['n_samples']}  "
          f"(prohibited={r['n_prohibited']}, not_prohibited={r['n_not_prohibited']})")
    print(f"  TP={r['tp']}  FP={r['fp']}  TN={r['tn']}  FN={r['fn']}")
    print(f"  Time               : {r['inference_seconds']}s")

    if r.get("by_type"):
        print(f"\n  {'Article 5 Type':<20} {'Recall':>7}  TP  FP  TN  FN")
        print(f"  {'─'*50}")
        for key, m in r["by_type"].items():
            if m["tp"] + m["fn"] + m["fp"] + m["tn"] == 0:
                continue
            flag = " ←" if m["recall"] < 0.70 else ""
            print(f"  {key:<20} {m['recall']*100:>6.0f}%{flag}  "
                  f"{m['tp']:>2}  {m['fp']:>2}  {m['tn']:>2}  {m['fn']:>2}")

    if r["errors"]:
        fn_errors = [e for e in r["errors"] if e["true"] == "prohibited"]
        fp_errors = [e for e in r["errors"] if e["true"] == "not_prohibited"]
        if fn_errors:
            print(f"\n  False Negatives — missed prohibited uses ({len(fn_errors)}):")
            for e in fn_errors[:4]:
                print(f"    conf={e['confidence']:.2f}  {e['text'][:80]}…")
                if e["note"]:
                    print(f"    note: {e['note'][:80]}")
        if fp_errors:
            print(f"\n  False Positives — lawful AI flagged as prohibited ({len(fp_errors)}):")
            for e in fp_errors[:4]:
                print(f"    conf={e['confidence']:.2f}  {e['text'][:80]}…")
                if e["note"]:
                    print(f"    note: {e['note'][:80]}")


# ── pytest integration ─────────────────────────────────────────────────────────

def test_prohibited_accuracy() -> None:
    """pytest: Prohibited classifier must reach ≥ 80% accuracy on gold dataset."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r["reason"])
    assert r["accuracy"] >= 0.80, (
        f"Prohibited classifier accuracy {r['accuracy']:.4f} is below 0.80. "
        f"Errors: {r.get('n_errors', '?')}/{r.get('n_samples', '?')}. "
        "The model may be overfitting to generated training text style."
    )


def test_prohibited_recall() -> None:
    """pytest: Prohibited recall must be ≥ 85% — missing an Article 5 violation is the critical failure."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r["reason"])
    assert r["recall"] >= 0.85, (
        f"Prohibited classifier recall {r['recall']:.4f} is below the 0.85 threshold. "
        f"FN={r['fn']} — the classifier is missing Article 5 prohibited practices. "
        "This is a critical gap: a missed prohibited use allows unlawful AI to pass compliance checks. "
        "Add more policy-document-style prohibited examples to training data or improve patterns."
    )


def test_prohibited_specificity() -> None:
    """pytest: Specificity must be ≥ 75% — false positives on lawful AI erode trust."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r["reason"])
    assert r["tnr"] >= 0.75, (
        f"Prohibited classifier specificity {r['tnr']:.4f} is below 0.75. "
        f"FP={r['fp']} — lawful AI systems are being incorrectly flagged as prohibited. "
        "Review hard-negative training examples for quality and add more nuanced not_prohibited cases."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prohibited classifier gold dataset evaluation")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("Running prohibited classifier evaluation …")
    results = run(verbose=args.verbose)
    if not args.verbose:
        print_report(results)

    sys.exit(0 if results.get("status") in ("pass", "warn") else 1)
