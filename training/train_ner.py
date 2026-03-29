#!/usr/bin/env python3
"""Train a spaCy NER model for EU AI Act entity recognition.

Extracts entities: ARTICLE, OBLIGATION, RISK_TIER, REGULATION
from compliance document text.

Usage:
    pip install spacy
    python -m spacy download de_core_news_sm
    python training/train_ner.py \
        --data training/data/ner_annotations.jsonl \
        --output training/spacy_ner_model \
        --epochs 30

The trained model is saved as a spaCy pipeline and wrapped in a Triton
Python backend at model_repository/spacy_ner/1/model.py.
"""

import argparse
import json
import random
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.training import Example


ENTITY_LABELS = ["ARTICLE", "OBLIGATION", "RISK_TIER", "REGULATION"]


def load_annotations(path: str) -> list[dict]:
    """Load NER annotations from JSONL."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_doc_bin(records: list[dict], nlp) -> DocBin:
    """Convert annotation records to a spaCy DocBin."""
    db = DocBin()
    for record in records:
        text = record["text"]
        entities = record.get("entities", [])
        doc = nlp.make_doc(text)
        spans = []
        for ent in entities:
            span = doc.char_span(ent["start"], ent["end"], label=ent["label"])
            if span is not None:
                spans.append(span)
        doc.ents = spans
        db.add(doc)
    return db


def main() -> None:
    parser = argparse.ArgumentParser(description="Train spaCy NER for EU AI Act entities")
    parser.add_argument("--data", default="training/data/ner_annotations.jsonl")
    parser.add_argument("--output", default="training/spacy_ner_model")
    parser.add_argument("--base-model", default="de_core_news_sm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading base model: {args.base_model}")
    nlp = spacy.load(args.base_model, disable=["tagger", "parser", "lemmatizer"])

    # Add NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for label in ENTITY_LABELS:
        ner.add_label(label)

    print(f"Loading annotations from {args.data}")
    records = load_annotations(args.data)
    print(f"  Loaded {len(records)} annotated sentences")

    # Split 80/20
    random.shuffle(records)
    cut = int(len(records) * 0.8)
    train_records = records[:cut]
    dev_records   = records[cut:]

    train_db = build_doc_bin(train_records, nlp)
    dev_db   = build_doc_bin(dev_records,   nlp)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    train_db.to_disk(output_path / "train.spacy")
    dev_db.to_disk(  output_path / "dev.spacy")

    print(f"\nTraining data saved to {output_path}")
    print("Run spaCy training with:")
    print(f"  python -m spacy train config.cfg --output {output_path}/model \\")
    print(f"    --paths.train {output_path}/train.spacy \\")
    print(f"    --paths.dev   {output_path}/dev.spacy")
    print("\nAlternatively, use the quick train loop below for prototyping:")

    # Quick in-process training loop (no config.cfg needed)
    optimizer = nlp.initialize()
    other_pipes = [p for p in nlp.pipe_names if p != "ner"]
    with nlp.disable_pipes(*other_pipes):
        for epoch in range(args.epochs):
            random.shuffle(train_records)
            losses: dict = {}
            for record in train_records:
                doc = nlp.make_doc(record["text"])
                spans = []
                for ent in record.get("entities", []):
                    span = doc.char_span(ent["start"], ent["end"], label=ent["label"])
                    if span:
                        spans.append(span)
                doc.ents = spans
                example = Example(nlp(record["text"]), doc)
                nlp.update([example], drop=0.3, losses=losses, sgd=optimizer)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} — NER loss: {losses.get('ner', 0):.4f}")

    # Save final model
    model_path = output_path / "model-final"
    nlp.to_disk(model_path)
    print(f"\nNER model saved to {model_path}")
    print("Copy to model_repository/spacy_ner/1/ for Triton deployment.")


if __name__ == "__main__":
    main()
