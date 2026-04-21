#!/usr/bin/env python3
"""Weak supervision pipeline using Snorkel.

This script applies programmatic Labeling Functions (LFs) to compliance
text to generate probabilistic labels for the 8 Article domains.

Usage:
    pip install snorkel pandas numpy
    python training/weak_supervision.py

This produces a probabilistic dataset that can augment or denoise the
BERT training data.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
    from snorkel.labeling.model import LabelModel
except ImportError:
    print("Snorkel is not installed. Run: pip install snorkel>=0.9.8")
    import sys
    sys.exit(1)

# Paths
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "training" / "data" / "clause_labels.jsonl"
OUT_PATH = ROOT / "training" / "data" / "weak_labels.jsonl"

# Label mapping
ABSTAIN = -1
LABELS = {
    "risk_management": 0,
    "data_governance": 1,
    "technical_documentation": 2,
    "record_keeping": 3,
    "transparency": 4,
    "human_oversight": 5,
    "security": 6,
    "unrelated": 7,
}
ID2LABEL = {v: k for k, v in LABELS.items()}

# ── Labeling Functions ────────────────────────────────────────────────────────

@labeling_function()
def lf_risk_keywords(x):
    keywords = ["risk", "hazard", "mitigation", "harm", "risiko", "gefahr"]
    return LABELS["risk_management"] if any(k in x.text.lower() for k in keywords) else ABSTAIN

@labeling_function()
def lf_data_keywords(x):
    keywords = ["dataset", "training data", "bias", "ground truth", "datensatz", "trainingsdaten"]
    return LABELS["data_governance"] if any(k in x.text.lower() for k in keywords) else ABSTAIN

@labeling_function()
def lf_tech_docs_keywords(x):
    keywords = ["architecture", "design", "specification", "model card", "architektur", "spezifikation"]
    return LABELS["technical_documentation"] if any(k in x.text.lower() for k in keywords) else ABSTAIN

@labeling_function()
def lf_record_keeping_keywords(x):
    keywords = ["log", "audit", "record", "traceability", "protokoll", "aufzeichnung"]
    return LABELS["record_keeping"] if any(k in x.text.lower() for k in keywords) else ABSTAIN

@labeling_function()
def lf_transparency_keywords(x):
    keywords = ["disclose", "instruction", "inform", "transparency", "informieren", "transparenz"]
    return LABELS["transparency"] if any(k in x.text.lower() for k in keywords) else ABSTAIN

@labeling_function()
def lf_human_oversight_keywords(x):
    keywords = ["human", "override", "stop button", "intervene", "mensch", "eingreifen"]
    return LABELS["human_oversight"] if any(k in x.text.lower() for k in keywords) else ABSTAIN

@labeling_function()
def lf_security_keywords(x):
    keywords = ["cybersecurity", "attack", "robustness", "fail-safe", "angriff", "sicherheit"]
    return LABELS["security"] if any(k in x.text.lower() for k in keywords) else ABSTAIN

@labeling_function()
def lf_unrelated_keywords(x):
    # If the text has words very common in generic business docs but lacks AI words
    ai_words = ["ai", "model", "algorithm", "system", "ki", "modell", "algorithmus"]
    if any(k in x.text.lower() for k in ai_words):
        return ABSTAIN
    generic_words = ["marketing", "hr", "vacation", "finance", "urlaub", "finanzen"]
    return LABELS["unrelated"] if any(k in x.text.lower() for k in generic_words) else ABSTAIN

# In a real weak supervision pipeline, we would also add LFs that hit Ollama models
# or use regex patterns.

LFS = [
    lf_risk_keywords, lf_data_keywords, lf_tech_docs_keywords,
    lf_record_keeping_keywords, lf_transparency_keywords,
    lf_human_oversight_keywords, lf_security_keywords, lf_unrelated_keywords
]

def load_data(path: Path) -> pd.DataFrame:
    records = []
    if not path.exists():
        print(f"Data file not found: {path}")
        return pd.DataFrame()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)

def main():
    print(f"Loading data from {DATA_PATH}...")
    df = load_data(DATA_PATH)
    if df.empty:
        print("No data to process. Exiting.")
        return

    # Keep a subset to simulate an unlabeled pool or to denoise existing data
    df_unlabeled = df[['text', 'lang']].copy()

    print("Applying Labeling Functions...")
    applier = PandasLFApplier(lfs=LFS)
    L = applier.apply(df=df_unlabeled)

    print("\nLF Analysis:")
    analysis = LFAnalysis(L=L, lfs=LFS).lf_summary()
    print(analysis)

    print("\nTraining LabelModel to denoise weak signals...")
    label_model = LabelModel(cardinality=8, verbose=True)
    label_model.fit(L_train=L, n_epochs=500, log_freq=100, seed=42)

    print("Predicting probabilistic labels...")
    preds = label_model.predict(L=L, tie_break_policy="abstain")
    probs = label_model.predict_proba(L=L)

    # Filter out abstains
    mask = preds != ABSTAIN
    df_labeled = df_unlabeled.iloc[mask].copy()
    df_labeled["label"] = [ID2LABEL[p] for p in preds[mask]]
    
    # Store maximum confidence score from the label model
    df_labeled["confidence"] = [max(prob) for prob in probs[mask]]

    print(f"\nGenerated {len(df_labeled)} weakly labeled examples.")
    
    # Save output
    out_records = df_labeled.to_dict(orient="records")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Saved weakly labeled data to {OUT_PATH}")

if __name__ == "__main__":
    main()
