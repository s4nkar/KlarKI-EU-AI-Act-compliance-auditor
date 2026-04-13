"""
Evaluation — NER Model Quality (8-label EU AI Act entity recognition).

Runs the trained spaCy NER model against a hand-crafted gold dataset of
sentences with known entity spans. Measures per-label and overall F1.

Thresholds:
  Overall F1  ≥ 0.80   (warn below, fail below 0.60)
  Per-label   ≥ 0.70   (flagged in report, fail gate at 0.50)

Skipped automatically if the trained model is not found.
No Ollama or ChromaDB required.

Usage:
    python tests/evaluation/eval_ner.py
    python tests/evaluation/eval_ner.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = next(
    (p for p in [
        REPO_ROOT / "training" / "spacy_ner_model" / "model-final",
        Path("/training/spacy_ner_model/model-final"),
    ] if p.exists()),
    None,
)

ENTITY_LABELS = [
    "ARTICLE", "OBLIGATION", "ACTOR", "AI_SYSTEM",
    "RISK_TIER", "PROCEDURE", "REGULATION", "PROHIBITED_USE",
]

# ---------------------------------------------------------------------------
# Hand-crafted gold dataset — (text, [(start, end, label), ...])
# Offsets are verified to match the text exactly.
# ---------------------------------------------------------------------------

def _e(text: str, span: str, label: str) -> dict:
    """Build an entity dict with verified char offsets."""
    idx = text.find(span)
    assert idx != -1, f"Span '{span}' not found in '{text}'"
    return {"start": idx, "end": idx + len(span), "label": label}


_RAW_GOLD: list[tuple[str, list[tuple[str, str]]]] = [
    # ARTICLE
    ("Providers must comply with Article 9 of the EU AI Act.",
     [("Article 9", "ARTICLE"), ("EU AI Act", "REGULATION")]),
    ("Both Article 13 and Article 14 address operator responsibilities.",
     [("Article 13", "ARTICLE"), ("Article 14", "ARTICLE")]),
    ("Gemäß Artikel 9 müssen Anbieter ein Risikomanagementsystem einrichten.",
     [("Artikel 9", "ARTICLE")]),

    # OBLIGATION
    ("Providers must document all training data sources used in development.",
     [("Providers", "ACTOR"), ("must document", "OBLIGATION")]),
    ("The operator shall maintain an audit trail of all system decisions.",
     [("shall maintain", "OBLIGATION")]),
    ("Importers are required to verify that the system conforms to Article 10.",
     [("are required to", "OBLIGATION"), ("Article 10", "ARTICLE")]),
    ("Anbieter müssen alle Trainingsdaten vollständig dokumentieren.",
     [("müssen", "OBLIGATION")]),

    # ACTOR
    ("Under Article 9, providers must establish a risk management system.",
     [("Article 9", "ARTICLE"), ("providers", "ACTOR"), ("must establish", "OBLIGATION")]),
    ("Notified bodies assess conformity of high-risk AI systems.",
     [("Notified bodies", "ACTOR"), ("high-risk", "RISK_TIER")]),
    ("Operators are responsible for the post-market monitoring of the AI system.",
     [("Operators", "ACTOR"), ("post-market monitoring", "PROCEDURE")]),
    ("Betreiber müssen eine Konformitätsbewertung durchführen.",
     [("Betreiber", "ACTOR")]),

    # AI_SYSTEM
    ("A high-risk AI system requires a conformity assessment under Article 43.",
     [("high-risk AI system", "AI_SYSTEM"), ("conformity assessment", "PROCEDURE"), ("Article 43", "ARTICLE")]),
    ("The general-purpose AI model must comply with transparency requirements.",
     [("general-purpose AI model", "AI_SYSTEM")]),
    ("An emotion recognition system deployed in the workplace is prohibited.",
     [("emotion recognition system", "AI_SYSTEM")]),
    ("Ein Hochrisiko-KI-System erfordert eine Konformitätsbewertung.",
     [("Hochrisiko-KI-System", "AI_SYSTEM"), ("Konformitätsbewertung", "PROCEDURE")]),

    # RISK_TIER
    ("This system is classified as high-risk under Article 9 of the EU AI Act.",
     [("high-risk", "RISK_TIER"), ("Article 9", "ARTICLE"), ("EU AI Act", "REGULATION")]),
    ("Real-time biometric surveillance in public spaces is prohibited under Article 5.",
     [("prohibited", "RISK_TIER"), ("Article 5", "ARTICLE")]),
    ("The chatbot is considered limited risk and only requires transparency measures.",
     [("limited risk", "RISK_TIER")]),
    ("Das System wurde als hochriskant eingestuft.",
     [("hochriskant", "RISK_TIER")]),

    # PROCEDURE
    ("Providers must complete a conformity assessment before market placement.",
     [("Providers", "ACTOR"), ("must complete", "OBLIGATION"), ("conformity assessment", "PROCEDURE")]),
    ("The risk management system must be established before deployment.",
     [("risk management system", "PROCEDURE")]),
    ("Technical documentation must be maintained for the lifetime of the system.",
     [("Technical documentation", "PROCEDURE")]),
    ("Anbieter müssen eine Konformitätsbewertung gemäß Artikel 43 durchführen.",
     [("Anbieter", "ACTOR"), ("Konformitätsbewertung", "PROCEDURE"), ("Artikel 43", "ARTICLE")]),

    # REGULATION
    ("The EU AI Act introduces a risk-based framework for AI systems.",
     [("EU AI Act", "REGULATION")]),
    ("Compliance requires adherence to both the EU AI Act and GDPR.",
     [("EU AI Act", "REGULATION"), ("GDPR", "REGULATION")]),
    ("Die DSGVO und das KI-Gesetz gelten gemeinsam für diese Anwendung.",
     [("DSGVO", "REGULATION"), ("KI-Gesetz", "REGULATION")]),

    # PROHIBITED_USE
    ("Social scoring by public authorities is explicitly banned under Article 5.",
     [("Social scoring", "PROHIBITED_USE"), ("Article 5", "ARTICLE")]),
    ("Emotion recognition in the workplace is prohibited under the EU AI Act.",
     [("Emotion recognition in the workplace", "PROHIBITED_USE"), ("EU AI Act", "REGULATION")]),
    ("Real-time biometric surveillance in public spaces violates Article 5.",
     [("Real-time biometric surveillance in public spaces", "PROHIBITED_USE"), ("Article 5", "ARTICLE")]),
    ("Subliminal manipulation of users is a prohibited AI practice.",
     [("Subliminal manipulation", "PROHIBITED_USE")]),
    ("Social Scoring durch Behörden ist nach Artikel 5 verboten.",
     [("Social Scoring", "PROHIBITED_USE"), ("Artikel 5", "ARTICLE")]),
]


def _build_gold() -> list[dict]:
    """Convert raw gold list to annotation dicts with verified char offsets."""
    records = []
    for text, spans in _RAW_GOLD:
        entities = [_e(text, span, label) for span, label in spans]
        records.append({"text": text, "entities": entities})
    return records


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(nlp, records: list[dict]) -> tuple[dict, list[dict]]:
    """Compute per-label and overall P/R/F1 against gold records.

    Returns (metrics_dict, mismatches_list).
    """
    tp: dict[str, int] = {l: 0 for l in ENTITY_LABELS}
    fp: dict[str, int] = {l: 0 for l in ENTITY_LABELS}
    fn: dict[str, int] = {l: 0 for l in ENTITY_LABELS}
    mismatches: list[dict] = []

    for rec in records:
        doc = nlp(rec["text"])
        pred_spans = {(e.start_char, e.end_char, e.label_) for e in doc.ents}
        gold_spans = {(e["start"], e["end"], e["label"]) for e in rec["entities"]}

        for span in gold_spans:
            if span in pred_spans:
                tp[span[2]] = tp.get(span[2], 0) + 1
            else:
                fn[span[2]] = fn.get(span[2], 0) + 1
                mismatches.append({
                    "type": "fn",
                    "text": rec["text"],
                    "span": rec["text"][span[0]:span[1]],
                    "label": span[2],
                })

        for span in pred_spans:
            if span not in gold_spans:
                fp[span[2]] = fp.get(span[2], 0) + 1
                mismatches.append({
                    "type": "fp",
                    "text": rec["text"],
                    "span": rec["text"][span[0]:span[1]],
                    "label": span[2],
                })

    per_label: dict[str, dict] = {}
    total_tp = total_fp = total_fn = 0

    for label in ENTITY_LABELS:
        t, f_p, f_n = tp[label], fp[label], fn[label]
        total_tp += t; total_fp += f_p; total_fn += f_n
        p  = t / (t + f_p) if (t + f_p) else 0.0
        r  = t / (t + f_n) if (t + f_n) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label[label] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
            "tp": t, "fp": f_p, "fn": f_n,
        }

    overall_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) else 0.0

    metrics = {
        "overall_f1": round(overall_f1, 4),
        "overall_p":  round(overall_p, 4),
        "overall_r":  round(overall_r, 4),
        "per_label":  per_label,
    }
    return metrics, mismatches


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(verbose: bool = False) -> dict:
    if MODEL_PATH is None:
        return _skip("NER model not found — run ./run.sh setup first")

    try:
        import spacy
    except ImportError:
        return _skip("spacy not installed")

    nlp = spacy.load(str(MODEL_PATH))

    # Verify all 8 labels are present in the model
    ner = nlp.get_pipe("ner")
    missing_labels = set(ENTITY_LABELS) - set(ner.labels)
    if missing_labels:
        return {
            "eval": "ner",
            "status": "fail",
            "reason": f"Model is missing labels: {missing_labels}",
        }

    gold = _build_gold()
    metrics, mismatches = _score(nlp, gold)

    overall_f1 = metrics["overall_f1"]
    weak_labels = {
        lbl: m["f1"]
        for lbl, m in metrics["per_label"].items()
        if m["f1"] < 0.70
    }
    failing_labels = {
        lbl: m["f1"]
        for lbl, m in metrics["per_label"].items()
        if m["f1"] < 0.50
    }

    if overall_f1 < 0.60 or failing_labels:
        status = "fail"
    elif overall_f1 < 0.80 or weak_labels:
        status = "warn"
    else:
        status = "pass"

    if verbose:
        print(f"\n  {'Label':<22} {'P':>6} {'R':>6} {'F1':>6}  TP  FP  FN")
        print("  " + "─" * 58)
        for lbl, m in metrics["per_label"].items():
            flag = " ←" if m["f1"] < 0.70 else ""
            print(f"  {lbl:<22} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                  f"{m['f1']:>6.3f}{flag}  {m['tp']:>2}  {m['fp']:>2}  {m['fn']:>2}")
        print("  " + "─" * 58)
        print(f"  {'Overall':<22} {metrics['overall_p']:>6.3f} "
              f"{metrics['overall_r']:>6.3f} {overall_f1:>6.3f}")

        if mismatches and verbose:
            print(f"\n  First 5 mismatches:")
            for m in mismatches[:5]:
                icon = "FP" if m["type"] == "fp" else "FN"
                print(f"    [{icon}] {m['label']}: '{m['span']}' in \"{m['text'][:70]}\"")

    results = {
        "eval":          "ner",
        "status":        status,
        "overall_f1":    overall_f1,
        "overall_p":     metrics["overall_p"],
        "overall_r":     metrics["overall_r"],
        "per_label":     metrics["per_label"],
        "n_gold":        len(gold),
        "weak_labels":   weak_labels,
        "failing_labels": failing_labels,
        "n_mismatches":  len(mismatches),
        "threshold_overall_f1":  0.80,
        "threshold_per_label_f1": 0.70,
    }

    out_path = RESULTS_DIR / "ner.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _skip(reason: str) -> dict:
    return {"eval": "ner", "status": "skip", "reason": reason}


def print_report(r: dict) -> None:
    if r["status"] == "skip":
        print(f"  [SKIP] {r['reason']}")
        return

    status_icon = "✓" if r["status"] == "pass" else ("!" if r["status"] == "warn" else "✗")
    print(f"\n  {'─'*52}")
    print(f"  {status_icon} NER Model Evaluation — 8 Entity Labels")
    print(f"  {'─'*52}")
    print(f"  Overall F1  : {r['overall_f1']*100:.1f}%  (threshold ≥ 80%)")
    print(f"  Precision   : {r['overall_p']*100:.1f}%")
    print(f"  Recall      : {r['overall_r']*100:.1f}%")
    print(f"  Gold samples: {r['n_gold']}")

    print(f"\n  {'Label':<22} {'F1':>6}")
    print("  " + "─" * 32)
    for lbl, m in r.get("per_label", {}).items():
        flag = " ← below 0.70" if m["f1"] < 0.70 else ""
        icon = "✓" if m["f1"] >= 0.70 else "!"
        print(f"  {icon} {lbl:<20} {m['f1']*100:>5.1f}%{flag}")

    if r.get("weak_labels"):
        print(f"\n  Weak labels (F1 < 70%): {list(r['weak_labels'].keys())}")
    if r.get("failing_labels"):
        print(f"  Failing labels (F1 < 50%): {list(r['failing_labels'].keys())}")


# ── pytest ──────────────────────────────────────────────────────────────────

def test_ner_all_labels_present() -> None:
    """pytest: Trained model must contain all 8 expected NER labels."""
    if MODEL_PATH is None:
        import pytest
        pytest.skip("NER model not found — run ./run.sh setup first")
    try:
        import spacy
    except ImportError:
        import pytest
        pytest.skip("spacy not installed — run pip install spacy")
    nlp = spacy.load(str(MODEL_PATH))
    ner = nlp.get_pipe("ner")
    missing = set(ENTITY_LABELS) - set(ner.labels)
    assert not missing, f"NER model missing labels: {missing}"


def test_ner_overall_f1() -> None:
    """pytest: NER overall F1 must be ≥ 0.80 on the gold dataset."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r.get("reason", ""))
    assert r["overall_f1"] >= 0.80, (
        f"NER overall F1 {r['overall_f1']:.4f} is below 0.80. "
        f"Weak labels: {r.get('weak_labels')}. Re-train with more data."
    )


def test_ner_no_label_below_50_pct() -> None:
    """pytest: No individual NER label should have F1 below 0.50."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r.get("reason", ""))
    assert not r.get("failing_labels"), (
        f"NER labels below 50% F1: {r['failing_labels']}. "
        "These labels need more training data or data quality fixes."
    )


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER model evaluation — 8 entity labels")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("Running NER evaluation …")
    results = run(verbose=args.verbose)
    print_report(results)
    sys.exit(0 if results.get("status") in ("pass", "warn") else 1)
