"""
Evaluation — Applicability Engine Gold Dataset.

Loads gold_applicability.jsonl and runs check_applicability() on each
example to measure how accurately the deterministic decision tree classifies:
  - Prohibited practices (Article 5)
  - High-risk systems (Article 6 + Annex III)
  - Minimal-risk systems

ML model paths are patched to None so only the pattern-based logic is
evaluated — the point is to test the deterministic layer in isolation.

Usage:
    python tests/evaluation/eval_applicability.py
    python tests/evaluation/eval_applicability.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
GOLD_PATH   = Path(__file__).parent / "datasets" / "gold_applicability.jsonl"
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


def _no_ml(_text):
    return None


def _make_chunk(text: str, idx: int = 0):
    from models.schemas import DocumentChunk
    return DocumentChunk(
        chunk_id=f"eval_{idx}", text=text, source_file="gold_eval.txt", chunk_index=idx
    )


def run(verbose: bool = False) -> dict:
    """Run applicability engine on the gold dataset. Returns results dict."""
    try:
        from services.applicability_engine import check_applicability
    except ImportError as exc:
        return _skip(f"Cannot import applicability_engine: {exc}. Run from repo root.")

    dataset = load_dataset()
    if verbose:
        print(f"  Loaded {len(dataset)} gold examples from {GOLD_PATH.name}")

    t0 = time.perf_counter()

    # Outcome tracking
    total = 0
    correct_prohibited = 0
    correct_high_risk  = 0
    correct_minimal    = 0
    tp_prohibited = fp_prohibited = fn_prohibited = 0
    tp_high_risk  = fp_high_risk  = fn_high_risk  = 0
    errors: list[dict] = []

    for idx, ex in enumerate(dataset):
        text            = ex["text"]
        exp_prohibited  = ex["expected_prohibited"]
        exp_high_risk   = ex["expected_high_risk"]

        chunks = [_make_chunk(text, idx)]

        with patch("services.applicability_engine._ml_prohibited", _no_ml), \
             patch("services.applicability_engine._ml_high_risk", _no_ml):
            result = check_applicability(chunks)

        pred_prohibited = result.is_prohibited
        pred_high_risk  = result.is_high_risk

        # Prohibited accuracy
        if pred_prohibited == exp_prohibited:
            correct_prohibited += 1
            if exp_prohibited:
                tp_prohibited += 1
        else:
            if exp_prohibited:
                fn_prohibited += 1
            else:
                fp_prohibited += 1

        # High-risk accuracy (only counted for non-prohibited examples)
        if not exp_prohibited and not pred_prohibited:
            if pred_high_risk == exp_high_risk:
                correct_high_risk += 1
                if exp_high_risk:
                    tp_high_risk += 1
            else:
                if exp_high_risk:
                    fn_high_risk += 1
                else:
                    fp_high_risk += 1

        # Minimal: both prohibited and high_risk are False
        exp_minimal  = not exp_prohibited and not exp_high_risk
        pred_minimal = not pred_prohibited and not pred_high_risk
        if exp_minimal and pred_minimal:
            correct_minimal += 1

        # Track any mismatch
        if pred_prohibited != exp_prohibited or pred_high_risk != exp_high_risk:
            errors.append({
                "text":            text[:100] + "…",
                "expected_prohibited": exp_prohibited,
                "expected_high_risk":  exp_high_risk,
                "pred_prohibited":     pred_prohibited,
                "pred_high_risk":      pred_high_risk,
                "note":                ex.get("note", ""),
            })

        total += 1

    elapsed = time.perf_counter() - t0

    # Per-outcome counts
    n_prohibited_ex = sum(1 for ex in dataset if ex["expected_prohibited"])
    n_high_risk_ex  = sum(1 for ex in dataset if not ex["expected_prohibited"] and ex["expected_high_risk"])
    n_minimal_ex    = sum(1 for ex in dataset if not ex["expected_prohibited"] and not ex["expected_high_risk"])

    def _safe_div(a, b):
        return round(a / b, 4) if b > 0 else 0.0

    # Precision / Recall for prohibited and high_risk
    prec_prohibited = _safe_div(tp_prohibited, tp_prohibited + fp_prohibited)
    rec_prohibited  = _safe_div(tp_prohibited, tp_prohibited + fn_prohibited)
    f1_prohibited   = _safe_div(
        2 * prec_prohibited * rec_prohibited,
        prec_prohibited + rec_prohibited,
    )

    prec_high_risk  = _safe_div(tp_high_risk, tp_high_risk + fp_high_risk)
    rec_high_risk   = _safe_div(tp_high_risk, tp_high_risk + fn_high_risk)
    f1_high_risk    = _safe_div(
        2 * prec_high_risk * rec_high_risk,
        prec_high_risk + rec_high_risk,
    )

    # Overall accuracy: fraction of examples where both flags match
    n_correct = total - len(errors)
    accuracy  = _safe_div(n_correct, total)

    status = "pass" if accuracy >= 0.80 else ("warn" if accuracy >= 0.65 else "fail")

    results = {
        "eval":             "applicability",
        "status":           status,
        "accuracy":         accuracy,
        "n_samples":        total,
        "n_errors":         len(errors),
        "errors":           errors,
        "inference_seconds": round(elapsed, 3),
        "threshold_accuracy": 0.80,
        "by_outcome": {
            "prohibited": {
                "n_examples": n_prohibited_ex,
                "n_correct":  correct_prohibited,
                "precision":  prec_prohibited,
                "recall":     rec_prohibited,
                "f1":         f1_prohibited,
            },
            "high_risk": {
                "n_examples": n_high_risk_ex,
                "n_correct":  correct_high_risk,
                "precision":  prec_high_risk,
                "recall":     rec_high_risk,
                "f1":         f1_high_risk,
            },
            "minimal": {
                "n_examples": n_minimal_ex,
                "n_correct":  correct_minimal,
            },
        },
    }

    out_path = RESULTS_DIR / "applicability.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if verbose:
        print_report(results)

    return results


def _skip(reason: str) -> dict:
    return {"eval": "applicability", "status": "skip", "reason": reason}


def print_report(r: dict) -> None:
    if r["status"] == "skip":
        print(f"  [SKIP] {r.get('reason', '')}")
        return

    icon = "✓" if r["status"] == "pass" else ("~" if r["status"] == "warn" else "✗")
    print(f"\n  {'─'*54}")
    print(f"  {icon} Applicability Engine — Gold Dataset Evaluation")
    print(f"  {'─'*54}")
    print(f"  Overall accuracy : {r['accuracy']*100:.1f}%   (threshold ≥ 80%)")
    print(f"  Samples          : {r['n_samples']}  |  Errors: {r['n_errors']}")
    print(f"  Time             : {r['inference_seconds']}s")
    print()

    by = r.get("by_outcome", {})
    print(f"  {'Outcome':<14} {'N':>5} {'Correct':>8} {'Prec':>6} {'Recall':>7} {'F1':>6}")
    print(f"  {'─'*48}")
    for key in ("prohibited", "high_risk", "minimal"):
        m = by.get(key, {})
        n = m.get("n_examples", 0)
        c = m.get("n_correct", 0)
        p = m.get("precision", None)
        rc = m.get("recall", None)
        f1 = m.get("f1", None)
        flag = " ←" if f1 is not None and f1 < 0.70 else ""
        p_str  = f"{p*100:>5.1f}%" if p is not None else "     —"
        r_str  = f"{rc*100:>6.1f}%" if rc is not None else "      —"
        f1_str = f"{f1*100:>5.1f}%{flag}" if f1 is not None else "     —"
        print(f"  {key:<14} {n:>5} {c:>8} {p_str} {r_str} {f1_str}")

    if r["errors"]:
        print(f"\n  Misclassified ({r['n_errors']}):")
        for e in r["errors"][:5]:
            print(
                f"    prohibited exp={e['expected_prohibited']} got={e['pred_prohibited']}  "
                f"high_risk exp={e['expected_high_risk']} got={e['pred_high_risk']}"
            )
            print(f"    {e['text'][:70]}…")


# ── pytest integration ─────────────────────────────────────────────────────────

def test_applicability_accuracy() -> None:
    """pytest: Applicability engine must reach ≥ 80% accuracy on gold dataset."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r["reason"])
    assert r["accuracy"] >= 0.80, (
        f"Applicability accuracy {r['accuracy']:.4f} is below the 0.80 threshold. "
        f"Errors: {r.get('n_errors', '?')}/{r.get('n_samples', '?')}"
    )


def test_prohibited_recall() -> None:
    """pytest: Prohibited detection recall must be ≥ 0.85 — missing a prohibited use is critical."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r["reason"])
    prohibited = r.get("by_outcome", {}).get("prohibited", {})
    recall = prohibited.get("recall", 0.0)
    n_ex   = prohibited.get("n_examples", 0)
    if n_ex == 0:
        import pytest
        pytest.skip("No prohibited examples in gold dataset.")
    assert recall >= 0.85, (
        f"Prohibited recall {recall:.4f} is below 0.85. "
        "Missing a prohibited AI practice is a critical false negative — "
        "extend _PROHIBITED_PATTERNS in applicability_engine.py."
    )


def test_high_risk_recall() -> None:
    """pytest: High-risk detection recall must be ≥ 0.75 — missing Annex III categories is a gap."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r["reason"])
    high_risk = r.get("by_outcome", {}).get("high_risk", {})
    recall = high_risk.get("recall", 0.0)
    n_ex   = high_risk.get("n_examples", 0)
    if n_ex == 0:
        import pytest
        pytest.skip("No high-risk examples in gold dataset.")
    assert recall >= 0.75, (
        f"High-risk recall {recall:.4f} is below 0.75. "
        "Extend _ANNEX_III_PATTERNS to cover more use-case variants."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Applicability engine gold dataset evaluation")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("Running applicability engine evaluation …")
    results = run(verbose=args.verbose)
    if not args.verbose:
        print_report(results)

    sys.exit(0 if results.get("status") in ("pass", "warn") else 1)
