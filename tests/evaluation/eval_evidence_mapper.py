"""
Evaluation — Evidence Mapper Gold Dataset.

Loads gold_evidence.jsonl and tests whether _evidence_present() correctly
identifies the presence or absence of each evidence artefact in a document
chunk, using regex synonyms only (NLI model is patched out for offline eval).

Metrics:
  True Positive Rate (TPR / Recall): fraction of present evidence found
  True Negative Rate (TNR / Specificity): fraction of absent evidence correctly not found
  Balanced Accuracy: (TPR + TNR) / 2

Threshold: TPR >= 0.80 (missing evidence that's present is the critical error).

Usage:
    python tests/evaluation/eval_evidence_mapper.py
    python tests/evaluation/eval_evidence_mapper.py --verbose
    python tests/evaluation/eval_evidence_mapper.py --with-nli  # enable NLI model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
GOLD_PATH   = Path(__file__).parent / "datasets" / "gold_evidence.jsonl"
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


def run(verbose: bool = False, with_nli: bool = False) -> dict:
    """Run evidence present/absent classification on gold dataset."""
    try:
        from services.evidence_mapper import _evidence_present
        from models.schemas import DocumentChunk
    except ImportError as exc:
        return _skip(f"Cannot import evidence_mapper: {exc}. Run from repo root.")

    dataset = load_dataset()
    if verbose:
        print(f"  Loaded {len(dataset)} gold examples from {GOLD_PATH.name}")

    tp = fp = tn = fn = 0
    errors: list[dict] = []
    t0 = time.perf_counter()

    # Group errors by evidence term for diagnostics
    per_term: dict[str, dict] = {}

    for idx, ex in enumerate(dataset):
        chunk_text    = ex["chunk_text"]
        term          = ex["evidence_term"]
        expected      = ex["expected_present"]

        chunk = DocumentChunk(
            chunk_id=f"ev_{idx}", text=chunk_text, source_file="gold_eval.txt", chunk_index=idx
        )

        if with_nli:
            matched_ids = _evidence_present(term, [chunk])
        else:
            with patch("services.evidence_mapper._get_nli_model", return_value=None):
                matched_ids = _evidence_present(term, [chunk])

        predicted = len(matched_ids) > 0

        # Update counts
        if term not in per_term:
            per_term[term] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "support": 0}
        per_term[term]["support"] += 1

        if expected and predicted:
            tp += 1
            per_term[term]["tp"] += 1
        elif expected and not predicted:
            fn += 1
            per_term[term]["fn"] += 1
            errors.append({
                "term":     term,
                "outcome":  "false_negative",
                "text":     chunk_text[:100] + "…",
                "note":     ex.get("note", ""),
            })
        elif not expected and predicted:
            fp += 1
            per_term[term]["fp"] += 1
            errors.append({
                "term":     term,
                "outcome":  "false_positive",
                "text":     chunk_text[:100] + "…",
                "note":     ex.get("note", ""),
            })
        else:
            tn += 1
            per_term[term]["tn"] += 1

    elapsed = time.perf_counter() - t0

    def _safe_div(a, b):
        return round(a / b, 4) if b > 0 else 0.0

    tpr = _safe_div(tp, tp + fn)  # True Positive Rate (Recall)
    tnr = _safe_div(tn, tn + fp)  # True Negative Rate (Specificity)
    balanced_accuracy = round((tpr + tnr) / 2, 4)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * precision * tpr, precision + tpr)

    # Per-term summary
    term_summary = {}
    for term, counts in per_term.items():
        t_tp, t_fp, t_tn, t_fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
        t_tpr = _safe_div(t_tp, t_tp + t_fn)
        t_tnr = _safe_div(t_tn, t_tn + t_fp)
        term_summary[term] = {
            "support": counts["support"],
            "tpr": t_tpr,
            "tnr": t_tnr,
            "tp": t_tp, "fp": t_fp, "tn": t_tn, "fn": t_fn,
        }

    status = "pass" if tpr >= 0.80 else ("warn" if tpr >= 0.65 else "fail")

    results = {
        "eval":               "evidence_mapper",
        "status":             status,
        "tpr":                tpr,
        "tnr":                tnr,
        "balanced_accuracy":  balanced_accuracy,
        "precision":          precision,
        "f1":                 f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_samples":          len(dataset),
        "n_errors":           len(errors),
        "errors":             errors,
        "per_term":           term_summary,
        "inference_seconds":  round(elapsed, 3),
        "mode":               "with_nli" if with_nli else "regex_only",
        "threshold_tpr":      0.80,
    }

    out_path = RESULTS_DIR / "evidence_mapper.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if verbose:
        print_report(results)

    return results


def _skip(reason: str) -> dict:
    return {"eval": "evidence_mapper", "status": "skip", "reason": reason}


def print_report(r: dict) -> None:
    if r["status"] == "skip":
        print(f"  [SKIP] {r.get('reason', '')}")
        return

    icon = "✓" if r["status"] == "pass" else ("~" if r["status"] == "warn" else "✗")
    mode = r.get("mode", "regex_only")
    print(f"\n  {'─'*56}")
    print(f"  {icon} Evidence Mapper — Gold Dataset Evaluation ({mode})")
    print(f"  {'─'*56}")
    print(f"  TPR (recall)     : {r['tpr']*100:.1f}%   (threshold ≥ 80%)")
    print(f"  TNR (specificity): {r['tnr']*100:.1f}%")
    print(f"  Balanced acc.    : {r['balanced_accuracy']*100:.1f}%")
    print(f"  Precision        : {r['precision']*100:.1f}%   F1: {r['f1']*100:.1f}%")
    print(f"  Samples          : {r['n_samples']}  TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")
    print(f"  Time             : {r['inference_seconds']}s")

    if r.get("per_term"):
        print(f"\n  {'Evidence Term':<35} {'N':>4} {'TPR':>6} {'TNR':>6}")
        print(f"  {'─'*54}")
        for term, m in sorted(r["per_term"].items()):
            flag = " ←" if m["tpr"] < 0.70 else ""
            print(
                f"  {term:<35} {m['support']:>4} "
                f"{m['tpr']*100:>5.0f}%{flag} "
                f"{m['tnr']*100:>5.0f}%"
            )

    if r["errors"]:
        fn_errors = [e for e in r["errors"] if e["outcome"] == "false_negative"]
        if fn_errors:
            print(f"\n  False Negatives (missed evidence, {len(fn_errors)}):")
            for e in fn_errors[:4]:
                print(f"    term='{e['term']}'  {e['text'][:65]}…")


# ── pytest integration ─────────────────────────────────────────────────────────

def test_evidence_mapper_tpr() -> None:
    """pytest: Evidence mapper TPR must be ≥ 80% on gold dataset (regex-only mode)."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r["reason"])
    assert r["tpr"] >= 0.80, (
        f"Evidence mapper TPR {r['tpr']:.4f} is below the 0.80 threshold. "
        f"FN={r['fn']} — some evidence terms are not matched by regex synonyms. "
        "Extend _EVIDENCE_SYNONYMS in evidence_mapper.py."
    )


def test_evidence_mapper_tnr() -> None:
    """pytest: Evidence mapper TNR must be ≥ 0.75 — too many false positives degrade trust."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r["reason"])
    assert r["tnr"] >= 0.75, (
        f"Evidence mapper TNR {r['tnr']:.4f} is below 0.75. "
        f"FP={r['fp']} — synonym patterns are too broad and are over-matching. "
        "Review overly general synonyms in _EVIDENCE_SYNONYMS."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evidence mapper gold dataset evaluation")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--with-nli", action="store_true", help="Enable NLI model (requires download)")
    args = parser.parse_args()

    print("Running evidence mapper evaluation …")
    results = run(verbose=args.verbose, with_nli=args.with_nli)
    if not args.verbose:
        print_report(results)

    sys.exit(0 if results.get("status") in ("pass", "warn") else 1)
