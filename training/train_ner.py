#!/usr/bin/env python3
"""Train a spaCy NER model for EU AI Act entity recognition.

Extracts entities: ARTICLE, OBLIGATION, RISK_TIER, REGULATION
from compliance document text.

Usage:
    pip install spacy tqdm
    python training/train_ner.py \
        --data training/data/ner_annotations.jsonl \
        --output training/spacy_ner_model \
        --epochs 30

Uses spacy.blank("de") to avoid the spacy-lookups-data dependency on de_core_news_sm.
The blank German model has the correct tokenisation rules without requiring lookup tables.

The trained model is saved as a spaCy pipeline and wrapped in a Triton
Python backend at model_repository/spacy_ner/1/model.py.
"""

import argparse
import json
import random
import time
from pathlib import Path

import spacy
import spacy.util
from spacy.tokens import DocBin
from spacy.training import Example


ENTITY_LABELS = ["ARTICLE", "OBLIGATION", "RISK_TIER", "REGULATION"]

RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[32m"
AMBER = "\033[33m"
CYAN  = "\033[36m"
DIM   = "\033[2m"
RED   = "\033[31m"


def _c(col: str, txt: str) -> str:
    return f"{col}{txt}{RESET}"


def _bar(current: int, total: int, width: int = 28) -> str:
    """Return an ASCII progress bar string."""
    filled = int(width * current / max(total, 1))
    return f"[{'#' * filled}{'.' * (width - filled)}] {current}/{total}"


def _resolve_spans(record: dict, nlp) -> list[tuple[int, int, str]]:
    """Return deduplicated (start_char, end_char, label) tuples for one record.

    Resolves all char spans to token spans, drops any that don't align with
    token boundaries, then greedily keeps the longest non-overlapping span.
    """
    doc = nlp.make_doc(record["text"])
    candidates = []
    for ent in record.get("entities", []):
        span = doc.char_span(ent["start"], ent["end"], label=ent["label"])
        if span is not None:
            candidates.append(span)
    candidates.sort(key=lambda s: s.end - s.start, reverse=True)
    accepted, occupied = [], set()
    for span in candidates:
        tokens = set(range(span.start, span.end))
        if not (tokens & occupied):
            accepted.append((span.start_char, span.end_char, span.label_))
            occupied |= tokens
    return accepted


def _make_example(record: dict, nlp) -> Example:
    spans = _resolve_spans(record, nlp)
    return Example.from_dict(nlp.make_doc(record["text"]), {"entities": spans})


def _eval_f1(nlp, records: list[dict]) -> tuple[float, dict]:
    """Run NER on records and return (overall_f1, ents_per_type)."""
    from spacy.scorer import Scorer
    scorer = Scorer()
    examples = []
    for record in records:
        ref_doc = nlp.make_doc(record["text"])
        ref_doc.ents = [
            ref_doc.char_span(s, e, label=l)
            for s, e, l in _resolve_spans(record, nlp)
            if ref_doc.char_span(s, e, label=l) is not None
        ]
        examples.append(Example(nlp(record["text"]), ref_doc))
    scores = scorer.score(examples)
    return round(scores.get("ents_f", 0.0), 4), scores.get("ents_per_type", {})


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
    """Convert annotation records to a spaCy DocBin.

    Overlapping spans (E1010) are resolved greedily: spans are sorted by
    length descending so longer/more-specific spans win, and any span whose
    token range overlaps an already-accepted span is dropped.
    """
    db = DocBin()
    skipped = 0
    for record in records:
        text = record["text"]
        entities = record.get("entities", [])
        doc = nlp.make_doc(text)

        # Resolve all char spans to token spans, drop any that don't align
        candidates = []
        for ent in entities:
            span = doc.char_span(ent["start"], ent["end"], label=ent["label"])
            if span is not None:
                candidates.append(span)

        # Greedy non-overlapping selection: longest span wins
        candidates.sort(key=lambda s: s.end - s.start, reverse=True)
        accepted = []
        occupied: set[int] = set()
        for span in candidates:
            tokens = set(range(span.start, span.end))
            if tokens & occupied:
                skipped += 1
                continue
            accepted.append(span)
            occupied |= tokens

        doc.ents = accepted
        db.add(doc)

    if skipped:
        print(_c(DIM, f"  Dropped {skipped} overlapping spans during DocBin build."))
    return db


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train spaCy NER for EU AI Act entities")
    parser.add_argument("--data", default="training/data/ner_annotations.jsonl")
    parser.add_argument("--output", default="training/spacy_ner_model")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Max epochs — early stopping may trigger earlier (default: 60)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size for training updates (default: 32)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience in epochs on dev F1 (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    random.seed(args.seed)

    print(_c(BOLD, "\n  NER Training -- EU AI Act Entity Recognition"))
    print(_c(DIM, "  -" * 30))

    # Use spacy.blank("de") — correct German tokenisation without needing
    # spacy-lookups-data or de_core_news_sm's lexeme_norm tables.
    print(_c(DIM, "  Loading blank German model (spacy.blank('de'))..."))
    nlp = spacy.blank("de")

    # Add NER component from scratch
    ner = nlp.add_pipe("ner")
    for label in ENTITY_LABELS:
        ner.add_label(label)
    print(_c(DIM, f"  Entity labels: {', '.join(ENTITY_LABELS)}"))

    print(f"\n  Loading annotations from {args.data}")
    records = load_annotations(args.data)
    print(_c(DIM, f"  Loaded {len(records)} annotated sentences"))

    if not records:
        print(_c(RED, "  ERROR: No annotations found. Check your JSONL file."))
        return

    # 80/20 train/dev split
    random.shuffle(records)
    cut = int(len(records) * 0.8)
    train_records = records[:cut]
    dev_records   = records[cut:]
    print(_c(DIM, f"  Train: {len(train_records)} / Dev: {len(dev_records)}"))

    # Save DocBin files
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    train_db = build_doc_bin(train_records, nlp)
    dev_db   = build_doc_bin(dev_records,   nlp)
    train_db.to_disk(output_path / "train.spacy")
    dev_db.to_disk(  output_path / "dev.spacy")

    # ── Pre-build training examples once (avoids redundant span resolution per epoch)
    print(_c(DIM, "  Building training examples..."))
    train_examples = [_make_example(r, nlp) for r in train_records]

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\n  Starting training: max {args.epochs} epochs, batch={args.batch_size}, "
          f"dropout={args.dropout}, patience={args.patience}")
    print(_c(DIM, "  -" * 30))

    optimizer = nlp.initialize()
    other_pipes = [p for p in nlp.pipe_names if p != "ner"]

    t_start = time.time()
    best_f1 = 0.0
    best_loss = 0.0
    best_epoch = 0
    patience_count = 0
    model_best_path = output_path / "model-best"

    with nlp.disable_pipes(*other_pipes):
        for epoch in range(1, args.epochs + 1):
            random.shuffle(train_examples)
            losses: dict = {}
            for batch in spacy.util.minibatch(train_examples, size=args.batch_size):
                nlp.update(batch, drop=args.dropout, losses=losses, sgd=optimizer)

            ner_loss = losses.get("ner", 0.0)

            # Per-epoch dev F1 — drives best-model saving and early stopping
            dev_f1, _ = _eval_f1(nlp, dev_records)
            improved = dev_f1 > best_f1

            if improved:
                best_f1 = dev_f1
                best_loss = ner_loss
                best_epoch = epoch
                patience_count = 0
                nlp.to_disk(model_best_path)
            else:
                patience_count += 1

            # Progress reporting every 5 epochs
            if epoch % 5 == 0 or epoch == 1 or improved:
                elapsed = time.time() - t_start
                bar = _bar(epoch, args.epochs)
                f1_colour = GREEN if dev_f1 >= 0.90 else AMBER if dev_f1 >= 0.75 else RED
                marker = _c(GREEN, " ← best") if improved else ""
                print(
                    f"  Epoch {epoch:>3}/{args.epochs}  {bar}"
                    f"  loss={ner_loss:.4f}  dev_F1={_c(f1_colour, f'{dev_f1:.4f}')}{marker}"
                    + _c(DIM, f"  {elapsed:.0f}s")
                )

            # Early stopping
            if patience_count >= args.patience:
                print(_c(AMBER, f"\n  Early stopping at epoch {epoch} "
                                f"(no dev F1 improvement for {args.patience} epochs)"))
                break

    total_time = time.time() - t_start
    print(_c(DIM, "\n  -" * 30))
    print(_c(GREEN, f"  Training complete in {total_time:.1f}s"))
    print(_c(GREEN, f"  Best dev F1: {best_f1:.4f} at epoch {best_epoch}"))

    # Load best model for evaluation and final save
    print(_c(DIM, f"  Loading best model from epoch {best_epoch}..."))
    nlp.from_disk(model_best_path)

    # Save as model-final (the best checkpoint, not the last epoch)
    model_path = output_path / "model-final"
    nlp.to_disk(model_path)
    print(_c(GREEN, f"  Model saved to {model_path}"))
    print(_c(DIM,   "  Copy to model_repository/spacy_ner/1/ for Triton deployment."))

    # -- Evaluation on dev set -------------------------------------------------
    print(_c(BOLD, "\n  Evaluating on dev set..."))
    _, ents_per_type = _eval_f1(nlp, dev_records)
    # Re-fetch full scores for overall metrics
    from spacy.scorer import Scorer
    scorer = Scorer()
    examples_dev = [
        Example(nlp(r["text"]),
                _make_example(r, nlp).reference)
        for r in dev_records
    ]
    scores = scorer.score(examples_dev)
    ents_per_type = scores.get("ents_per_type", {})

    print(f"  {'Label':<20} {'P':>6} {'R':>6} {'F1':>6}")
    print("  " + "-" * 42)
    per_label = []
    for label in ENTITY_LABELS:
        s = ents_per_type.get(label, {})
        p  = round(s.get("p",  0.0), 4)
        r  = round(s.get("r",  0.0), 4)
        f1 = round(s.get("f",  0.0), 4)
        per_label.append({"label": label, "precision": p, "recall": r, "f1": f1})
        colour = GREEN if f1 >= 0.85 else AMBER if f1 >= 0.70 else RED
        print(_c(colour, f"  {label:<20} {p:>6.3f} {r:>6.3f} {f1:>6.3f}"))

    overall_f1 = round(scores.get("ents_f", 0.0), 4)
    overall_p  = round(scores.get("ents_p", 0.0), 4)
    overall_r  = round(scores.get("ents_r", 0.0), 4)
    print("  " + "-" * 42)
    overall_colour = GREEN if overall_f1 >= 0.90 else AMBER if overall_f1 >= 0.75 else RED
    print(_c(overall_colour, f"  {'Overall':<20} {overall_p:>6.3f} {overall_r:>6.3f} {overall_f1:>6.3f}"))

    # Save metrics JSON
    metrics_payload = {
        "overall_f1":  overall_f1,
        "overall_p":   overall_p,
        "overall_r":   overall_r,
        "per_label":   per_label,
        "labels":      ENTITY_LABELS,
        "val_size":    len(dev_records),
        "train_size":  len(train_records),
        "final_loss":  round(best_loss, 4),
    }
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(_c(GREEN, f"\n  Metrics saved to {metrics_path}"))


if __name__ == "__main__":
    main()
