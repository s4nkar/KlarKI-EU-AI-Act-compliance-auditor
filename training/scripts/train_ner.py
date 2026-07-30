#!/usr/bin/env python3
"""Train a spaCy NER model for EU AI Act entity recognition.

Extracts entities:
  ARTICLE, OBLIGATION, ACTOR, AI_SYSTEM, RISK_TIER, PROCEDURE, REGULATION, PROHIBITED_USE

Uses de_core_news_lg as the backbone (pre-trained German word vectors + tok2vec).
Falls back to spacy.blank("de") if de_core_news_lg is not installed.

Install backbone:
    python -m spacy download de_core_news_lg

Usage:
    python training/train_ner.py \
        --data training/data/ner_annotations.jsonl \
        --output training/artifacts/spacy_ner_model

The trained model is saved as a spaCy pipeline and wrapped in a Triton
Python backend at model_repository/spacy_ner/1/model.py.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Windows cp1252 stdout can't encode non-ASCII symbols (e.g. ←, —).
# Reconfigure to UTF-8 so progress output renders correctly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import spacy
import spacy.util
from spacy.tokens import DocBin
from spacy.training import Example


ENTITY_LABELS = ["ARTICLE", "OBLIGATION", "ACTOR", "AI_SYSTEM", "RISK_TIER", "PROCEDURE", "REGULATION", "PROHIBITED_USE"]

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
        # alignment_mode="expand": when an offset lands mid-token (e.g. the
        # German tokenizer fuses a sentence-final "43." into one token, so the
        # exact span "Article 43" can't align), expand to the enclosing token
        # boundary instead of silently dropping the span. Strict mode was
        # discarding ~25% of ARTICLE spans (all sentence-final), so the model
        # never learned end-of-sentence article references.
        span = doc.char_span(ent["start"], ent["end"], label=ent["label"],
                             alignment_mode="expand")
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


def _audit_span_alignment(records: list[dict], nlp) -> dict:
    """Count annotated spans that spaCy cannot align to token boundaries.

    `doc.char_span(start, end)` returns None when an offset falls mid-token
    (e.g. a tagger truncated 'Hochrisiko-KI-System' inside the inflected
    'Hochrisiko-KI-Systeme'). Those spans are silently dropped from training
    AND from the dev score, so dev F1 can look perfect while real coverage
    quietly shrinks. We surface the drop counts so the failure is visible.
    """
    from collections import Counter
    requested: Counter = Counter()
    dropped: Counter = Counter()
    for rec in records:
        doc = nlp.make_doc(rec["text"])
        for ent in rec.get("entities", []):
            label = ent["label"]
            requested[label] += 1
            # Mirror training's alignment_mode so the audit reports spans that
            # are *actually* dropped (those that can't align even when expanded).
            if doc.char_span(ent["start"], ent["end"], label=label,
                             alignment_mode="expand") is None:
                dropped[label] += 1
    total_req = sum(requested.values())
    total_drop = sum(dropped.values())
    rate = (total_drop / total_req) if total_req else 0.0
    colour = RED if rate > 0.05 else AMBER if rate > 0.01 else GREEN
    print(_c(colour, f"  Span alignment: {total_drop}/{total_req} dropped "
                     f"({rate*100:.1f}%) — misaligned offsets are NOT trained"))
    if dropped:
        worst = ", ".join(f"{lbl}={dropped[lbl]}/{requested[lbl]}"
                          for lbl, _ in dropped.most_common(8))
        print(_c(colour, f"    by label: {worst}"))
    if rate > 0.05:
        print(_c(RED, "  [!!] >5% of spans are misaligned — check the data "
                      "generator's entity offsets (word-boundary truncation?)."))
    return {"requested": total_req, "dropped": total_drop, "rate": round(rate, 4)}


def _eval_overlap(nlp, records: list[dict]) -> tuple[float, float, float, dict]:
    """Label-aware OVERLAP scorer — mirrors the CI gate in
    tests/evaluation/eval_ner.py::_score so training selection AND the final
    reported metrics use the same rule the gate asserts (a gold entity counts as
    found if a same-label prediction overlaps it). Keep this in sync with that
    file. Returns (overall_f1, overall_p, overall_r, per_label) where per_label
    maps label -> {precision, recall, f1, tp, fp, fn}.
    """
    def _overlaps(a_s, a_e, b_s, b_e) -> bool:
        return a_s < b_e and b_s < a_e

    tp = {l: 0 for l in ENTITY_LABELS}
    fp = {l: 0 for l in ENTITY_LABELS}
    fn = {l: 0 for l in ENTITY_LABELS}

    for rec in records:
        doc = nlp(rec["text"])
        pred = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
        gold = [(e["start"], e["end"], e["label"]) for e in rec["entities"]]
        for gs, ge, gl in gold:
            if any(pl == gl and _overlaps(gs, ge, ps, pe) for ps, pe, pl in pred):
                tp[gl] = tp.get(gl, 0) + 1
            else:
                fn[gl] = fn.get(gl, 0) + 1
        for ps, pe, pl in pred:
            if not any(gl == pl and _overlaps(ps, pe, gs, ge) for gs, ge, gl in gold):
                fp[pl] = fp.get(pl, 0) + 1

    per_label: dict[str, dict] = {}
    tot_tp = tot_fp = tot_fn = 0
    for label in ENTITY_LABELS:
        t, f_p, f_n = tp[label], fp[label], fn[label]
        tot_tp += t; tot_fp += f_p; tot_fn += f_n
        p = t / (t + f_p) if (t + f_p) else 0.0
        r = t / (t + f_n) if (t + f_n) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label[label] = {"precision": round(p, 4), "recall": round(r, 4),
                            "f1": round(f1, 4), "tp": t, "fp": f_p, "fn": f_n}

    op = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0.0
    orr = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0.0
    of1 = 2 * op * orr / (op + orr) if (op + orr) else 0.0
    return round(of1, 4), round(op, 4), round(orr, 4), per_label


def load_annotations(path: str) -> list[dict]:
    """Load NER annotations from JSONL."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _case_augment(records: list[dict], prob: float, rng: random.Random) -> tuple[list[dict], int]:
    """Lowercase a random subset of records wholesale.

    Entity offsets are unaffected — str.lower() is length-preserving for the
    EN/DE alphabets used here. Guards against the model keying off surface
    capitalization instead of the words themselves (e.g. it previously only
    recognised some PROHIBITED_USE phrases when Title-Cased, never in their
    equally-valid lowercase mid-sentence form — a training-data casing
    artifact, not a real distinction). Applied to TRAIN only, at low
    probability, so natural casing signals that matter (German nouns are
    always capitalized; REGULATION acronyms like GDPR) stay dominant.
    """
    augmented = []
    n = 0
    for rec in records:
        if rng.random() < prob:
            augmented.append({**rec, "text": rec["text"].lower()})
            n += 1
        else:
            augmented.append(rec)
    return augmented, n


def build_doc_bin(records: list[dict], nlp) -> DocBin:
    """Convert annotation records to a spaCy DocBin.

    Overlapping spans are resolved via _resolve_spans (greedy, longest wins).
    """
    db = DocBin()
    for record in records:
        doc = nlp.make_doc(record["text"])
        doc.ents = [
            sp for sp in (
                doc.char_span(s, e, label=l)
                for s, e, l in _resolve_spans(record, nlp)
            )
            if sp is not None
        ]
        db.add(doc)
    return db


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train spaCy NER for EU AI Act entities")
    parser.add_argument("--data", default="training/data/ner_annotations.jsonl")
    parser.add_argument("--output", default="training/artifacts/spacy_ner_model")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Max epochs — early stopping may trigger earlier (default: 60)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size for training updates (default: 32)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience in epochs on dev F1 (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--case-augment-prob", type=float, default=0.08,
                        help="Fraction of TRAIN examples lowercased wholesale for case "
                             "robustness (dev set untouched; default: 0.08; 0 disables)")
    args = parser.parse_args()

    # Full determinism: spacy.util.fix_random_seed seeds Python random, NumPy and
    # (if present) torch, so the split shuffle AND spaCy's tok2vec init + dropout
    # are reproducible. Seeding only Python `random` left model init and dropout
    # unseeded, causing run-to-run dev-F1 swings right at the eval gate.
    random.seed(args.seed)
    spacy.util.fix_random_seed(args.seed)

    print(_c(BOLD, "\n  NER Training -- EU AI Act Entity Recognition"))
    print(_c(DIM, "  -" * 30))

    # Load de_core_news_lg as backbone (pre-trained German tok2vec + word vectors).
    # Falls back to blank German model if not installed.
    try:
        nlp = spacy.load("de_core_news_lg", exclude=["ner"])
        backbone = "de_core_news_lg (pre-trained tok2vec + word vectors)"
    except OSError:
        print(_c(AMBER, "  WARNING: de_core_news_lg not found — using blank German model."))
        print(_c(AMBER, "  Install for better accuracy: python -m spacy download de_core_news_lg"))
        nlp = spacy.blank("de")
        backbone = "spacy.blank('de')"
    print(_c(DIM, f"  Backbone: {backbone}"))

    # Add fresh NER head on top of the backbone
    ner = nlp.add_pipe("ner", last=True)
    for label in ENTITY_LABELS:
        ner.add_label(label)
    print(_c(DIM, f"  Entity labels: {', '.join(ENTITY_LABELS)}"))

    print(f"\n  Loading annotations from {args.data}")
    records = load_annotations(args.data)
    print(_c(DIM, f"  Loaded {len(records)} annotated sentences"))

    # Surface which generator(s) produced this data — there are currently 3
    # (scripts/generate_ner_data.py, local-datagen, local-datagen-V2) and
    # silently training on an unexpected one has caused real confusion before.
    from collections import Counter
    gen_counts = Counter(r.get("generator", "unknown") for r in records)
    if len(gen_counts) > 1 or "unknown" in gen_counts:
        colour = AMBER
    else:
        colour = GREEN
    print(_c(colour, f"  Generator provenance: {dict(gen_counts)}"))

    if not records:
        print(_c(RED, "  ERROR: No annotations found. Check your JSONL file."))
        return

    # Pre-flight: surface any annotated spans spaCy can't align to tokens.
    # These are silently dropped from training otherwise.
    _audit_span_alignment(records, nlp)

    # 80/20 train/dev split
    random.shuffle(records)
    cut = int(len(records) * 0.8)
    train_records = records[:cut]
    dev_records   = records[cut:]
    print(_c(DIM, f"  Train: {len(train_records)} / Dev: {len(dev_records)}"))

    if args.case_augment_prob > 0:
        train_records, n_aug = _case_augment(
            train_records, args.case_augment_prob, random.Random(args.seed))
        print(_c(DIM, f"  Case augmentation: lowercased {n_aug}/{len(train_records)} "
                      f"train examples (p={args.case_augment_prob}) for case robustness"))

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

    # de_core_news_lg tries to load lexeme_norm lookup tables during initialize()
    # which requires spacy-lookups-data. NER doesn't need these tables — clear them.
    nlp.config["initialize"]["lookups"] = None
    optimizer = nlp.initialize()
    # Keep tok2vec active so NER benefits from pre-trained representations.
    # Disable only task-specific heads irrelevant to NER (tagger, parser, etc.).
    # No-op for blank model which has no tok2vec.
    skip_in_training = {"ner", "tok2vec"}
    other_pipes = [p for p in nlp.pipe_names if p not in skip_in_training]

    t_start = time.time()
    best_f1 = 0.0
    best_loss: float | None = None
    best_epoch = 0
    patience_count = 0
    model_best_path = output_path / "model-best"

    with nlp.disable_pipes(*other_pipes):
        for epoch in range(1, args.epochs + 1):
            random.shuffle(train_examples)
            losses: dict = {}
            for batch in spacy.util.minibatch(train_examples, size=args.batch_size):
                nlp.update(batch, drop=args.dropout, losses=losses, sgd=optimizer)

            train_loss = losses.get("ner", 0.0)

            # Per-epoch dev F1 — selection uses the OVERLAP metric (same rule as
            # the eval-suite gate) so the promoted checkpoint maximises what it is
            # gated on.
            dev_f1, _, _, _ = _eval_overlap(nlp, dev_records)
            improved = dev_f1 > best_f1

            if improved:
                best_f1 = dev_f1
                best_loss = train_loss
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

                # Train loss colouring — high loss late in training suggests underfitting
                loss_colour = GREEN if train_loss < 0.5 else AMBER if train_loss < 1.0 else RED
                loss_str = _c(loss_colour, f"train_loss={train_loss:.4f}")

                print(
                    f"  Epoch {epoch:>3}/{args.epochs}  {bar}"
                    f"  {loss_str}  dev_F1={_c(f1_colour, f'{dev_f1:.4f}')}{marker}"
                    + _c(DIM, f"  {elapsed:.0f}s")
                )

                if train_loss > 1.0 and dev_f1 < 0.60:
                    print(_c(AMBER, "    [!]  High train loss + low F1 — model may be UNDERFITTING"))

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
    if best_epoch == 0 or not model_best_path.exists():
        print(_c(AMBER, "  WARNING: No best model checkpoint found — using final epoch state."))
    else:
        print(_c(DIM, f"  Loading best model from epoch {best_epoch}..."))
        nlp.from_disk(model_best_path)

    # Save as model-final (the best checkpoint, not the last epoch).
    # Disable the non-NER backbone pipes before saving so they persist as
    # disabled in config.cfg. If they run at inference they overwrite doc.ents
    # and ARTICLE/OBLIGATION/PROHIBITED_USE entities silently vanish — the same
    # interference this script already works around during its own eval.
    save_disable = [p for p in nlp.pipe_names if p not in {"ner", "tok2vec"}]
    if save_disable:
        nlp.select_pipes(disable=save_disable)
    model_path = output_path / "model-final"
    nlp.to_disk(model_path)
    print(_c(GREEN, f"  Model saved to {model_path} (active pipes: {nlp.pipe_names})"))
    print(_c(DIM,   "  Copy to model_repository/spacy_ner/1/ for Triton deployment."))

    # -- Evaluation on dev set -------------------------------------------------
    # Uses the label-aware OVERLAP scorer (same rule as tests/evaluation/eval_ner.py
    # and as best-model selection above), so the reported overall_f1 — which
    # version_manager uses to rank NER promotions — matches the CI gate exactly
    # instead of the stricter span-exact number it used to report.
    # After the final select_pipes(disable=...) above, non-NER pipes are already
    # inactive, so nlp(text) won't let the tagger overwrite doc.ents.
    print(_c(BOLD, "\n  Evaluating on dev set (label-aware overlap)..."))
    overall_f1, overall_p, overall_r, per_label_map = _eval_overlap(nlp, dev_records)

    print(f"  {'Label':<20} {'P':>6} {'R':>6} {'F1':>6}")
    print("  " + "-" * 42)
    per_label = []
    for label in ENTITY_LABELS:
        s  = per_label_map.get(label, {})
        p  = round(s.get("precision", 0.0), 4)
        r  = round(s.get("recall",    0.0), 4)
        f1 = round(s.get("f1",        0.0), 4)
        per_label.append({"label": label, "precision": p, "recall": r, "f1": f1})
        colour = GREEN if f1 >= 0.85 else AMBER if f1 >= 0.70 else RED
        print(_c(colour, f"  {label:<20} {p:>6.3f} {r:>6.3f} {f1:>6.3f}"))

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
        "final_loss":  round(best_loss, 4) if best_loss is not None else None,
    }
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(_c(GREEN, f"\n  Metrics saved to {metrics_path}"))


if __name__ == "__main__":
    main()
