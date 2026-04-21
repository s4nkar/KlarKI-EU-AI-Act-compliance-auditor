#!/usr/bin/env python3
"""Patch the trained spaCy NER model with a rule-based EntityRuler.

Places an EntityRuler AFTER the learned NER pipe with overwrite_ents=True
so that known regulatory phrases are matched deterministically, overriding
anything the NER learned (or failed to learn) for those tokens.

The NER model still runs first and catches entities the ruler doesn't cover.

Usage:
    python scripts/patch_ner_model.py
    python scripts/patch_ner_model.py --model training/artifacts/spacy_ner_model/model-final
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def build_patterns() -> list[dict]:
    patterns: list[dict] = []

    # ── ARTICLE ────────────────────────────────────────────────────────────────
    # LIKE_NUM matches "9", "10.", "43." etc.
    # The German tokenizer fuses trailing sentence-period with numbers ("10." = one token),
    # so LIKE_NUM is intentional here — gold data includes the period in end-of-sentence spans.
    patterns += [
        {"label": "ARTICLE", "pattern": [{"LOWER": "article"}, {"LIKE_NUM": True}]},
        {"label": "ARTICLE", "pattern": [{"LOWER": "artikel"}, {"LIKE_NUM": True}]},
        {"label": "ARTICLE", "pattern": [{"LOWER": "art"}, {"TEXT": "."}, {"LIKE_NUM": True}]},
        {"label": "ARTICLE", "pattern": [{"LOWER": "gdpr"}, {"LOWER": "article"}, {"LIKE_NUM": True}]},
        {"label": "ARTICLE", "pattern": [{"LOWER": "annex"}, {"LIKE_NUM": True}]},
        {"label": "ARTICLE", "pattern": [{"LOWER": "annex"}, {"TEXT": {"REGEX": "^[IVX]+$"}}]},
    ]

    # ── OBLIGATION ─────────────────────────────────────────────────────────────
    patterns += [
        {"label": "OBLIGATION", "pattern": [{"LOWER": "must"}]},
        {"label": "OBLIGATION", "pattern": [{"LOWER": "shall"}]},
        {"label": "OBLIGATION", "pattern": [{"LOWER": "are"}, {"LOWER": "required"}, {"LOWER": "to"}]},
        {"label": "OBLIGATION", "pattern": [{"LOWER": "is"}, {"LOWER": "required"}, {"LOWER": "to"}]},
        {"label": "OBLIGATION", "pattern": [{"LOWER": "müssen"}]},
        {"label": "OBLIGATION", "pattern": [{"LOWER": "muss"}]},
        {"label": "OBLIGATION", "pattern": [{"LOWER": "soll"}]},
        {"label": "OBLIGATION", "pattern": [{"LOWER": "sollen"}]},
        {"label": "OBLIGATION", "pattern": [{"LOWER": "sind"}, {"LOWER": "verpflichtet"}]},
    ]

    # ── ACTOR ──────────────────────────────────────────────────────────────────
    actors_en = [
        "provider", "providers", "operator", "operators",
        "importer", "importers", "manufacturer", "manufacturers",
        "distributor", "distributors", "deployer", "deployers",
        "notified body", "notified bodies",
        "conformity assessment body", "market surveillance authority",
        "national competent authority",
    ]
    actors_de = [
        "Anbieter", "Betreiber", "Einführer", "Hersteller",
        "Händler", "Inverkehrbringer", "Bevollmächtigter",
    ]
    for a in actors_en + actors_de:
        patterns.append({"label": "ACTOR", "pattern": a})

    # ── AI_SYSTEM ──────────────────────────────────────────────────────────────
    # "high-risk" stays free for RISK_TIER — ruler captures "AI system" separately.
    # Phrase patterns handle compound forms; token pattern handles bare "AI system/model".
    for s in [
        "general-purpose AI model", "general-purpose AI models",
        "general-purpose AI system", "general-purpose AI systems",
        "emotion recognition system", "emotion recognition systems",
        "remote biometric identification system",
        "real-time remote biometric identification system",
        "biometric identification system", "social credit system",
        "Hochrisiko-KI-System", "KI-System", "KI-Systeme", "KI-Modell",
    ]:
        patterns.append({"label": "AI_SYSTEM", "pattern": s})
    patterns += [
        {"label": "AI_SYSTEM", "pattern": [{"LOWER": "ai"}, {"LOWER": {"IN": ["system", "systems", "model", "models"]}}]},
    ]

    # ── RISK_TIER ──────────────────────────────────────────────────────────────
    risk_tiers = [
        "high-risk", "prohibited", "unacceptable risk",
        "limited risk", "minimal risk", "low risk",
        "hochriskant", "verboten", "hohes Risiko",
        "minimales Risiko", "begrenztes Risiko",
    ]
    for t in risk_tiers:
        patterns.append({"label": "RISK_TIER", "pattern": t})

    # ── PROCEDURE ──────────────────────────────────────────────────────────────
    procedures = [
        "conformity assessment", "conformity assessments",
        "risk management system", "risk management systems",
        "technical documentation",
        "post-market monitoring",
        "fundamental rights impact assessment",
        "data protection impact assessment",
        "human rights impact assessment",
        "quality management system",
        "incident reporting",
        "market surveillance",
        "Konformitätsbewertung", "Risikomanagementsystem",
        "technische Dokumentation", "Marktüberwachung",
        "Grundrechte-Folgenabschätzung",
    ]
    for p in procedures:
        patterns.append({"label": "PROCEDURE", "pattern": p})

    # ── REGULATION ─────────────────────────────────────────────────────────────
    regulations = [
        "EU AI Act", "AI Act",
        "Artificial Intelligence Act",
        "General Data Protection Regulation",
        "GDPR", "DSGVO",
        "KI-Gesetz", "KI-Verordnung", "EU-KI-Gesetz",
        "NIS2", "NIS 2 Directive",
        "Product Liability Directive",
        "Machinery Regulation",
    ]
    for r in regulations:
        patterns.append({"label": "REGULATION", "pattern": r})

    # ── PROHIBITED_USE ─────────────────────────────────────────────────────────
    prohibited = [
        "social scoring",
        "emotion recognition in the workplace",
        "emotion recognition in educational institutions",
        "real-time biometric surveillance in public spaces",
        "real-time biometric surveillance",
        "subliminal manipulation",
        "subliminal techniques",
        "mass surveillance of natural persons",
        "exploitation of vulnerabilities",
        "biometric categorisation system",
        "Social Scoring",
        "Emotionserkennung am Arbeitsplatz",
        "Echtzeit-Biometrie-Überwachung",
    ]
    for p in prohibited:
        patterns.append({"label": "PROHIBITED_USE", "pattern": p})

    return patterns


def patch(model_path: Path) -> None:
    import spacy

    print(f"Loading model from {model_path} ...")
    nlp = spacy.load(str(model_path))
    print(f"  Pipes: {nlp.pipe_names}")

    # Remove stale ruler if already patched
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
        print("  Removed existing entity_ruler (will re-add)")

    # Add ruler AFTER ner so it overrides NER output for known phrases
    ruler = nlp.add_pipe(
        "entity_ruler",
        after="ner",
        config={"overwrite_ents": True, "phrase_matcher_attr": "LOWER"},
    )

    patterns = build_patterns()
    ruler.add_patterns(patterns)
    print(f"  Added {len(patterns)} patterns across 8 labels")

    nlp.to_disk(str(model_path))
    print(f"  Saved patched model to {model_path}")

    # Quick smoke test
    print("\nSmoke test:")
    checks = [
        ("The operator shall maintain a complete audit trail.", {"OBLIGATION", "ACTOR"}),
        ("high-risk AI system must comply with Article 9 of the EU AI Act.", {"AI_SYSTEM", "ARTICLE", "REGULATION"}),
        ("social scoring is banned under Article 5.", {"PROHIBITED_USE", "ARTICLE"}),
        ("Providers must complete a conformity assessment.", {"ACTOR", "PROCEDURE", "OBLIGATION"}),
        ("Compliance with the EU AI Act and GDPR is mandatory.", {"REGULATION"}),
    ]
    for text, expected in checks:
        doc = nlp(text)
        found = {e.label_ for e in doc.ents}
        missing = expected - found
        status = "PASS" if not missing else f"FAIL (missing {missing})"
        ents = [(e.text, e.label_) for e in doc.ents]
        print(f"  [{status}] {text[:60]}")
        if missing:
            print(f"         found={ents}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent.parent / "training" / "spacy_ner_model" / "model-final"),
    )
    args = parser.parse_args()
    patch(Path(args.model))


if __name__ == "__main__":
    main()
