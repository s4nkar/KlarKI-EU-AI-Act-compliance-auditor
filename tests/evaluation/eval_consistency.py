"""
Evaluation 6 — Consistency Testing.

Runs the same 10 probe texts through the BERT classifier N times and
measures label stability.  A consistent system should return the same
label on every run for every text.

Metric:
  Consistency Rate — fraction of (text, run) pairs that match the
  majority label for that text.  Target: 100% (BERT is deterministic
  in eval mode — any inconsistency indicates an environment issue).

Also runs Ollama-based consistency (if available): sends the same
prompt 5 times and measures label agreement.  Target: ≥ 90% (LLMs
can be nondeterministic even at temperature 0).

Usage:
    python tests/evaluation/eval_consistency.py
    python tests/evaluation/eval_consistency.py --runs 10 --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
MODEL_PATH  = next(
    (p for p in [REPO_ROOT / "training" / "bert_classifier", Path("/training/bert_classifier")] if p.exists()),
    Path("/training/bert_classifier"),
)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
_API_DIR = next((p for p in [REPO_ROOT / "api", Path("/app")] if p.is_dir()), Path("/app"))
sys.path.insert(0, str(_API_DIR))

# Representative probe texts — one per article domain + one unrelated
PROBE_TEXTS = [
    {"text": "The risk management system must identify all foreseeable hazards.", "expected": "risk_management"},
    {"text": "Training data shall be representative of all demographic groups.", "expected": "data_governance"},
    {"text": "Technical documentation must describe the system architecture.", "expected": "technical_documentation"},
    {"text": "All AI decisions must be automatically logged with timestamps.", "expected": "record_keeping"},
    {"text": "Users must be informed when interacting with an AI system.", "expected": "transparency"},
    {"text": "Human operators can override the AI at any decision point.", "expected": "human_oversight"},
    {"text": "The system must achieve defined accuracy metrics under adversarial conditions.", "expected": "security"},
    {"text": "The quarterly sales report showed strong revenue growth this period.", "expected": "unrelated"},
    {"text": "Risk mitigation measures address all identified safety hazards.", "expected": "risk_management"},
    {"text": "Cybersecurity testing confirmed resilience against model poisoning.", "expected": "security"},
]


# ── BERT consistency (deterministic) ──────────────────────────────────────

def run_bert_consistency(n_runs: int, verbose: bool) -> dict:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return _skip_section("BERT", "transformers/torch not installed")

    if not MODEL_PATH.exists():
        return _skip_section("BERT", f"Model not found at {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.eval()
    id2label: dict[int, str] = model.config.id2label  # type: ignore[assignment]

    texts = [p["text"] for p in PROBE_TEXTS]

    # Collect n_runs × n_texts predictions
    all_runs: list[list[str]] = []
    for _ in range(n_runs):
        inputs = tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).tolist()
        all_runs.append([id2label[p] for p in preds])

    # For each text: check all runs agree
    per_text: list[dict] = []
    total_consistent = 0

    for i, probe in enumerate(PROBE_TEXTS):
        labels_for_text = [run[i] for run in all_runs]
        majority, count = Counter(labels_for_text).most_common(1)[0]
        consistent      = count == n_runs
        total_consistent += int(consistent)

        per_text.append({
            "text":      probe["text"][:60],
            "expected":  probe["expected"],
            "majority":  majority,
            "correct":   majority == probe["expected"],
            "consistent": consistent,
            "all_same":  consistent,
        })

        if verbose:
            icon = "✓" if consistent else "✗"
            print(f"  BERT {icon} [{majority}] {probe['text'][:55]}")

    rate = total_consistent / len(PROBE_TEXTS)
    return {
        "consistency_rate": round(rate, 4),
        "n_runs":           n_runs,
        "per_text":         per_text,
        "status":           "pass" if rate >= 1.0 else "warn",
    }


# ── Ollama consistency (nondeterministic) ─────────────────────────────────

async def _ollama_consistency_async(n_runs: int, verbose: bool) -> dict:
    import os
    try:
        from services.ollama_client  import OllamaClient
        from services.classifier     import classify_chunks
        from models.schemas          import DocumentChunk
    except ImportError as e:
        return _skip_section("ollama", f"Cannot import services: {e}")

    ollama_host = os.getenv("OLLAMA_HOST", "localhost")
    ollama = OllamaClient(host=ollama_host)
    try:
        if not await ollama.health_check():
            raise RuntimeError()
    except Exception:
        return _skip_section("ollama", f"Ollama not reachable at {ollama_host}")

    # Run only first 5 probes to keep runtime reasonable
    probes = PROBE_TEXTS[:5]
    per_text: list[dict] = []
    total_consistent = 0

    for probe in probes:
        labels_across_runs: list[str] = []
        for _ in range(n_runs):
            chunk = DocumentChunk(
                chunk_id="consistency-probe",
                text=probe["text"],
                source_file="probe.txt",
                chunk_index=0,
            )
            result = await classify_chunks([chunk], ollama)
            labels_across_runs.append(result[0].domain.value if result[0].domain else "unrelated")

        majority, count = Counter(labels_across_runs).most_common(1)[0]
        consistent      = count / n_runs >= 0.80  # ≥ 80% agreement for LLM

        total_consistent += int(consistent)
        per_text.append({
            "text":               probe["text"][:60],
            "expected":           probe["expected"],
            "majority":           majority,
            "agreement_rate":     round(count / n_runs, 2),
            "consistent_at_80pct": consistent,
        })

        if verbose:
            icon = "✓" if consistent else "✗"
            print(f"  Ollama {icon} [{majority} {count}/{n_runs}] {probe['text'][:50]}")

    rate = total_consistent / len(probes)
    return {
        "consistency_rate":       round(rate, 4),
        "n_runs":                 n_runs,
        "per_text":               per_text,
        "status":                 "pass" if rate >= 0.80 else "warn",
        "agreement_threshold":    0.80,
    }


def _skip_section(backend: str, reason: str) -> dict:
    return {"status": "skip", "backend": backend, "reason": reason}


def run(n_runs: int = 5, verbose: bool = False) -> dict:
    bert_results   = run_bert_consistency(n_runs=n_runs, verbose=verbose)
    ollama_results = asyncio.run(_ollama_consistency_async(n_runs=n_runs, verbose=verbose))

    overall_status = "pass"
    if bert_results.get("status") == "fail" or ollama_results.get("status") == "fail":
        overall_status = "fail"
    elif bert_results.get("status") == "warn" or ollama_results.get("status") == "warn":
        overall_status = "warn"

    results = {
        "eval":   "consistency",
        "status": overall_status,
        "bert":   bert_results,
        "ollama": ollama_results,
    }

    out_path = RESULTS_DIR / "consistency.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def print_report(r: dict) -> None:
    status_icon = "✓" if r["status"] == "pass" else ("!" if r["status"] == "warn" else "✗")
    print(f"\n  {'─'*52}")
    print(f"  {status_icon} Consistency Testing")
    print(f"  {'─'*52}")

    for backend in ("bert", "ollama"):
        b = r.get(backend, {})
        if b.get("status") == "skip":
            print(f"  [{backend.upper()} SKIP] {b.get('reason')}")
            continue
        rate = b.get("consistency_rate", 0)
        icon = "✓" if b.get("status") == "pass" else "!"
        print(f"  {icon} {backend.upper():<8} consistency: {rate*100:.1f}%  (runs={b.get('n_runs')})")


# ── pytest ─────────────────────────────────────────────────────────────────

def test_bert_fully_consistent() -> None:
    """pytest: BERT (deterministic) must return identical labels on every run."""
    r = run(n_runs=3)
    b = r.get("bert", {})
    if b.get("status") == "skip":
        import pytest
        pytest.skip(b.get("reason", ""))
    assert b["consistency_rate"] >= 1.0, (
        "BERT produced inconsistent labels across runs — check that the model "
        "is in eval() mode and no dropout is active during inference."
    )


def test_ollama_consistent_at_80pct() -> None:
    """pytest: Ollama must agree with itself ≥ 80% of the time per probe."""
    r = run(n_runs=5)
    o = r.get("ollama", {})
    if o.get("status") == "skip":
        import pytest
        pytest.skip(o.get("reason", ""))
    assert o["consistency_rate"] >= 0.80, (
        f"Ollama consistency rate {o['consistency_rate']:.2%} is below 80%. "
        "Consider lowering Ollama temperature or improving the prompt."
    )


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consistency testing")
    parser.add_argument("--runs",    type=int, default=5,  help="Number of inference runs per text")
    parser.add_argument("--verbose", "-v",     action="store_true")
    args = parser.parse_args()

    print(f"Running consistency testing ({args.runs} runs per text) …")
    results = run(n_runs=args.runs, verbose=args.verbose)
    print_report(results)
    sys.exit(0 if results.get("status") in ("pass", "warn") else 1)
