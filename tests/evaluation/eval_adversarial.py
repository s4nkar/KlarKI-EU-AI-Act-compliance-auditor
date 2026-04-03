"""
Evaluation 5 — Adversarial / Semantic Robustness.

Tests whether the BERT classifier maps unusual paraphrases of compliance
concepts to the same article domain as the canonical wording.

Example: "hazard governance process" should still map to risk_management
         even though it never uses the word "risk".

Metric: Adversarial Accuracy — fraction of paraphrases classified correctly.
Target: ≥ 75% (lower threshold than gold set due to intentional difficulty).

Usage:
    python tests/evaluation/eval_adversarial.py
    python tests/evaluation/eval_adversarial.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
ADV_PATH    = Path(__file__).parent / "datasets" / "adversarial_queries.jsonl"
MODEL_PATH  = next(
    (p for p in [REPO_ROOT / "training" / "bert_classifier", Path("/training/bert_classifier")] if p.exists()),
    Path("/training/bert_classifier"),
)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_adversarial() -> list[dict]:
    examples = []
    with open(ADV_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def run(verbose: bool = False) -> dict:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return _skip("transformers/torch not installed")

    if not MODEL_PATH.exists():
        return _skip(f"BERT model not found at {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.eval()
    id2label: dict[int, str] = model.config.id2label  # type: ignore[assignment]

    examples    = load_adversarial()
    texts       = [ex["text"]           for ex in examples]
    true_labels = [ex["expected_label"] for ex in examples]

    # Batch inference
    pred_labels: list[str] = []
    t0 = time.perf_counter()
    batch_size = 16
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

    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    accuracy = correct / len(examples)

    # Per-concept breakdown
    by_concept: dict[str, dict] = {}
    for ex, pred in zip(examples, pred_labels):
        key = ex.get("paraphrase_of", ex["expected_label"])
        if key not in by_concept:
            by_concept[key] = {"total": 0, "correct": 0, "failures": []}
        by_concept[key]["total"] += 1
        if ex["expected_label"] == pred:
            by_concept[key]["correct"] += 1
        else:
            by_concept[key]["failures"].append({
                "text": ex["text"][:80],
                "expected": ex["expected_label"],
                "predicted": pred,
            })

    if verbose:
        for concept, info in by_concept.items():
            icon = "✓" if info["correct"] == info["total"] else "!"
            print(f"  {icon} {concept}: {info['correct']}/{info['total']}")
            for f in info["failures"]:
                print(f"      ✗ [{f['expected']} → {f['predicted']}] \"{f['text']}\"")

    results = {
        "eval":                 "adversarial",
        "status":               "pass" if accuracy >= 0.75 else "warn",
        "adversarial_accuracy": round(accuracy, 4),
        "correct":              correct,
        "total":                len(examples),
        "by_concept":           by_concept,
        "inference_seconds":    round(elapsed, 3),
        "threshold_accuracy":   0.75,
    }

    out_path = RESULTS_DIR / "adversarial.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _skip(reason: str) -> dict:
    return {"eval": "adversarial", "status": "skip", "reason": reason}


def print_report(r: dict) -> None:
    if r["status"] == "skip":
        print(f"  [SKIP] {r['reason']}")
        return

    status_icon = "✓" if r["status"] == "pass" else "!"
    print(f"\n  {'─'*52}")
    print(f"  {status_icon} Adversarial Robustness Evaluation")
    print(f"  {'─'*52}")
    print(f"  Adversarial Acc : {r['adversarial_accuracy']*100:.1f}%  (threshold ≥ 75%)")
    print(f"  Correct         : {r['correct']}/{r['total']}")

    print("\n  Per concept:")
    for concept, info in r.get("by_concept", {}).items():
        pct  = info["correct"] / info["total"] * 100
        icon = "✓" if info["correct"] == info["total"] else ("!" if pct >= 50 else "✗")
        print(f"    {icon} {concept:<32} {info['correct']}/{info['total']}  ({pct:.0f}%)")


# ── pytest ─────────────────────────────────────────────────────────────────

def test_adversarial_accuracy() -> None:
    """pytest: Adversarial paraphrase accuracy must be ≥ 75%."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r.get("reason", ""))
    assert r["adversarial_accuracy"] >= 0.75, (
        f"Adversarial accuracy {r['adversarial_accuracy']:.2%} is below 75%. "
        "The classifier is not semantically robust to paraphrased inputs."
    )


def test_no_concept_below_50_pct() -> None:
    """pytest: No single concept group should score below 50%."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r.get("reason", ""))
    weak = {
        k: v["correct"] / v["total"]
        for k, v in r.get("by_concept", {}).items()
        if v["correct"] / v["total"] < 0.50
    }
    assert not weak, f"Concepts below 50% adversarial accuracy: {weak}"


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial robustness evaluation")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("Running adversarial robustness evaluation …")
    results = run(verbose=args.verbose)
    print_report(results)
    sys.exit(0 if results.get("status") in ("pass", "warn") else 1)
