#!/usr/bin/env python3
"""Deterministic NER training data generator for EU AI Act compliance auditing.

Replaces Ollama-based synthetic generation with two fast, dependency-free methods:

  1. Real regulatory corpus extraction — sentences from data/regulatory/**/*.txt
     with entity offsets computed by regex / phrase matching.

  2. Deterministic template expansion — controlled vocabularies × sentence
     templates produce thousands of clean, multi-entity samples in seconds.
     Entity offsets are tracked during string assembly (exact, no str.find()).

Entity labels (8 total):
  ARTICLE        — Article references          "Article 9", "Artikel 13", "Art. 14"
  OBLIGATION     — Compliance duty phrases     "must document", "shall maintain"
  ACTOR          — Regulatory role actors      "providers", "operators", "importers"
  AI_SYSTEM      — Named AI system types       "high-risk AI system", "general-purpose AI model"
  RISK_TIER      — Risk classifications        "high-risk", "prohibited", "unacceptable risk"
  PROCEDURE      — Named compliance procedures "conformity assessment", "risk management system"
  REGULATION     — Regulation names            "EU AI Act", "GDPR", "DSGVO"
  PROHIBITED_USE — Article 5 banned practices  "social scoring", "emotion recognition in the workplace"

Output format: training/data/ner_annotations.jsonl — unchanged from prior version,
so train_ner.py continues to work without modification.

Usage:
    python scripts/generate_ner_data.py
    python scripts/generate_ner_data.py --n-templates 5000
    python scripts/generate_ner_data.py --overwrite
    python scripts/generate_ner_data.py --n-templates 100   # smoke test
"""

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
REGULATORY_DIR = ROOT / "data" / "regulatory"

# ---------------------------------------------------------------------------
# Controlled vocabularies — EN
# ---------------------------------------------------------------------------

_VOCABS_EN: dict[str, list[str]] = {
    "ACTOR": [
        "providers", "operators", "importers", "manufacturers", "distributors",
        "deployers", "authorised representatives", "national competent authorities",
        "notified bodies", "market surveillance authorities", "users",
        "the provider", "the operator", "the importer", "the manufacturer",
        "the deployer", "the authorised representative",
    ],
    "OBLIGATION": [
        "must establish", "shall maintain", "must implement",
        "must document", "shall ensure", "must notify",
        "must conduct", "shall submit", "must retain", "must verify",
        "need to demonstrate", "must undergo", "shall register",
        "must draw up", "shall draw up", "must provide", "shall provide",
        "must assess", "shall assess", "must be documented",
        "shall be conducted", "must be retained", "are required to",
        "are obliged to", "have to register",
    ],
    "AI_SYSTEM": [
        "high-risk AI system", "general-purpose AI model",
        "remote biometric identification system", "emotion recognition system",
        "prohibited AI system", "biometric categorisation system",
        "AI system used in critical infrastructure", "AI system for credit scoring",
        "AI system for recruitment", "AI system used in education",
        "AI system used in employment", "real-time biometric surveillance system",
    ],
    "RISK_TIER": [
        "high-risk", "prohibited", "limited risk", "minimal risk",
        "unacceptable risk", "low risk",
    ],
    "PROCEDURE": [
        "conformity assessment", "risk management system", "technical documentation",
        "post-market monitoring", "data governance", "quality management system",
        "fundamental rights impact assessment", "human oversight mechanism",
        "incident reporting procedure", "logging and record-keeping system",
        "transparency obligations", "market surveillance procedure",
    ],
    "ARTICLE": (
        [f"Article {n}" for n in list(range(5, 21)) + [35, 43, 47]]
        + [f"Art. {n}" for n in [9, 10, 11, 12, 13, 14, 15]]
    ),
    "REGULATION": [
        "EU AI Act", "AI Act", "Artificial Intelligence Act", "GDPR",
        "General Data Protection Regulation", "NIS2", "Cyber Resilience Act",
    ],
    "PROHIBITED_USE": [
        "social scoring", "emotion recognition in the workplace",
        "emotion recognition in educational institutions",
        "real-time remote biometric identification in public spaces",
        "real-time biometric surveillance in public spaces",
        "subliminal manipulation", "exploitation of vulnerabilities",
        "predictive policing based solely on profiling",
        "biometric categorisation to infer sensitive characteristics",
        "manipulation of behaviour without awareness",
        "mass surveillance of natural persons",
        "indiscriminate scraping of facial images",
    ],
}

# ---------------------------------------------------------------------------
# Controlled vocabularies — DE
# ---------------------------------------------------------------------------

_VOCABS_DE: dict[str, list[str]] = {
    "ACTOR": [
        "Anbieter", "Betreiber", "Einführer", "Hersteller", "Händler",
        "Bevollmächtigte", "zuständige nationale Behörden", "benannte Stellen",
        "Marktüberwachungsbehörden", "der Anbieter", "der Betreiber",
        "der Einführer", "der Hersteller", "der Bevollmächtigte",
    ],
    "OBLIGATION": [
        "müssen einrichten", "müssen sicherstellen", "müssen dokumentieren",
        "müssen implementieren", "müssen melden", "müssen prüfen",
        "sind verpflichtet", "müssen nachweisen", "müssen unterhalten",
        "müssen durchführen", "müssen registrieren", "müssen erstellen",
        "muss sicherstellen", "muss dokumentieren", "muss einrichten",
        "muss melden", "muss prüfen", "muss nachweisen", "muss erstellen",
        "sind gehalten", "ist verpflichtet",
    ],
    "AI_SYSTEM": [
        "Hochrisiko-KI-System", "KI-System mit allgemeinem Verwendungszweck",
        "verbotenes KI-System", "biometrisches Fernidentifizierungssystem",
        "Emotionserkennungssystem", "KI-System für die Kreditwürdigkeitsprüfung",
        "KI-System im Bereich der kritischen Infrastruktur",
        "KI-System im Bildungsbereich", "biometrisches Kategorisierungssystem",
    ],
    "RISK_TIER": [
        "hochriskant", "verboten", "begrenztes Risiko", "minimales Risiko",
        "unannehmbares Risiko", "Hochrisiko",
    ],
    "PROCEDURE": [
        "Konformitätsbewertung", "Risikomanagementsystem", "technische Dokumentation",
        "Überwachung nach dem Inverkehrbringen", "Datenverwaltung",
        "Qualitätsmanagementsystem", "Folgenabschätzung für Grundrechte",
        "menschliche Aufsicht", "Vorfallmeldesystem", "Marktüberwachungsverfahren",
        "Protokollierungssystem", "Transparenzpflichten",
    ],
    "ARTICLE": (
        [f"Artikel {n}" for n in list(range(5, 21)) + [35, 43, 47]]
        + [f"Art. {n}" for n in [9, 10, 11, 12, 13, 14, 15]]
    ),
    "REGULATION": [
        "KI-Gesetz", "EU-KI-Verordnung", "DSGVO", "Datenschutz-Grundverordnung",
        "NIS2", "Cyber Resilience Act",
    ],
    "PROHIBITED_USE": [
        "Social Scoring", "Emotionserkennung am Arbeitsplatz",
        "Emotionserkennung in Bildungseinrichtungen",
        "biometrische Echtzeit-Fernidentifizierung im öffentlichen Raum",
        "unterschwellige Manipulation", "Ausnutzung von Schwachstellen",
        "prädiktive Polizeiarbeit auf Basis von Profiling",
        "biometrische Kategorisierung sensibler Merkmale",
        "Manipulation des Verhaltens ohne Bewusstsein",
        "Massenüberwachung natürlicher Personen",
        "unterschiedslose Erfassung von Gesichtsbildern",
    ],
}

# ---------------------------------------------------------------------------
# Sentence templates — EN
# Slots are {LABEL_NAME}. Every slot is tracked as an entity span.
# ---------------------------------------------------------------------------

_TEMPLATES_EN: list[str] = [
    "{ACTOR} {OBLIGATION} the {PROCEDURE} under {ARTICLE} of the {REGULATION}.",
    "Under {ARTICLE} of the {REGULATION}, {ACTOR} {OBLIGATION} a {PROCEDURE}.",
    "{ARTICLE} of the {REGULATION} requires {ACTOR} to {OBLIGATION} a {PROCEDURE}.",
    "The {REGULATION} mandates that {ACTOR} {OBLIGATION} the {PROCEDURE}.",
    "{ACTOR} deploying a {AI_SYSTEM} {OBLIGATION} a {PROCEDURE}.",
    "A {RISK_TIER} {AI_SYSTEM} requires {ACTOR} to {OBLIGATION} a {PROCEDURE}.",
    "{ACTOR} {OBLIGATION} that the {AI_SYSTEM} complies with {ARTICLE}.",
    "Under {ARTICLE}, {ACTOR} {OBLIGATION} the {PROCEDURE} for the {AI_SYSTEM}.",
    "The {REGULATION} classifies this as a {RISK_TIER} {AI_SYSTEM}.",
    "{ACTOR} {OBLIGATION} a {PROCEDURE} before placing the {AI_SYSTEM} on the market.",
    "For a {RISK_TIER} {AI_SYSTEM}, {ACTOR} {OBLIGATION} a {PROCEDURE} per {ARTICLE}.",
    "{ARTICLE} of the {REGULATION} obliges {ACTOR} to {OBLIGATION} a {PROCEDURE}.",
    "The {PROCEDURE} required under {ARTICLE} must be completed by {ACTOR}.",
    "{ACTOR} are responsible for the {PROCEDURE} of any {AI_SYSTEM}.",
    "Pursuant to {ARTICLE} of the {REGULATION}, {ACTOR} {OBLIGATION} a {PROCEDURE}.",
    "Any {RISK_TIER} {AI_SYSTEM} requires {ACTOR} to {OBLIGATION} the {PROCEDURE}.",
    "{ACTOR} {OBLIGATION} technical evidence of compliance with {ARTICLE} of the {REGULATION}.",
    "The {PROCEDURE} defined in {ARTICLE} applies to all {RISK_TIER} systems.",
    "Before deployment, {ACTOR} {OBLIGATION} the {PROCEDURE} required by the {REGULATION}.",
    "{ACTOR} {OBLIGATION} that the {AI_SYSTEM} documentation satisfies {ARTICLE} standards.",
    "The {REGULATION} requires a {PROCEDURE} for each {AI_SYSTEM} operated by {ACTOR}.",
    "{ACTOR} {OBLIGATION} a {PROCEDURE} in accordance with {ARTICLE} of the {REGULATION}.",
    "A {PROCEDURE} under {ARTICLE} is mandatory for {ACTOR} using a {AI_SYSTEM}.",
    "When deploying a {RISK_TIER} {AI_SYSTEM}, {ACTOR} {OBLIGATION} the {PROCEDURE}.",
    "{ACTOR} {OBLIGATION} to register the {AI_SYSTEM} under {ARTICLE} of the {REGULATION}.",
    # RISK_TIER-focused templates to balance label frequency
    "This system is classified as {RISK_TIER} under {ARTICLE} of the {REGULATION}.",
    "A {RISK_TIER} {AI_SYSTEM} falls under the requirements of the {REGULATION}.",
    "{ACTOR} {OBLIGATION} declare whether the {AI_SYSTEM} is {RISK_TIER} under {ARTICLE}.",
    "The {REGULATION} defines this {AI_SYSTEM} as {RISK_TIER} and subject to {PROCEDURE}.",
    "{ACTOR} {OBLIGATION} verify the {RISK_TIER} classification before market placement.",
    "Under {ARTICLE}, all {RISK_TIER} systems deployed by {ACTOR} require a {PROCEDURE}.",
    "The {RISK_TIER} classification triggers additional {PROCEDURE} obligations for {ACTOR}.",
    "A system is deemed {RISK_TIER} when {ACTOR} deploy it for purposes listed in {ARTICLE}.",
    "{ACTOR} {OBLIGATION} reassess the {RISK_TIER} status of the {AI_SYSTEM} periodically.",
    "The {PROCEDURE} required by {ARTICLE} applies only to {RISK_TIER} systems.",
    # PROHIBITED_USE templates — Article 5 banned practices
    "{PROHIBITED_USE} is explicitly banned under {ARTICLE} of the {REGULATION}.",
    "{ACTOR} {OBLIGATION} ensure the {AI_SYSTEM} does not employ {PROHIBITED_USE}.",
    "The {REGULATION} classifies {PROHIBITED_USE} as {RISK_TIER} under {ARTICLE}.",
    "{ACTOR} {OBLIGATION} verify that the {AI_SYSTEM} does not involve {PROHIBITED_USE}.",
    "Any {AI_SYSTEM} capable of {PROHIBITED_USE} is {RISK_TIER} under {ARTICLE}.",
    "{ARTICLE} of the {REGULATION} prohibits {PROHIBITED_USE} in all contexts.",
    "{ACTOR} {OBLIGATION} certify the absence of {PROHIBITED_USE} per {ARTICLE}.",
    "The {PROCEDURE} includes a check for {PROHIBITED_USE} as required by {ARTICLE}.",
    "Under {ARTICLE}, {ACTOR} {OBLIGATION} screen the {AI_SYSTEM} for {PROHIBITED_USE}.",
    "{PROHIBITED_USE} triggers an automatic {RISK_TIER} classification under {ARTICLE}.",
]

# ---------------------------------------------------------------------------
# Sentence templates — DE
# ---------------------------------------------------------------------------

_TEMPLATES_DE: list[str] = [
    "{ACTOR} {OBLIGATION} eine {PROCEDURE} gemäß {ARTICLE} der {REGULATION}.",
    "Gemäß {ARTICLE} der {REGULATION} {OBLIGATION} {ACTOR} eine {PROCEDURE}.",
    "{ARTICLE} der {REGULATION} verpflichtet {ACTOR} zur Durchführung einer {PROCEDURE}.",
    "Die {REGULATION} schreibt vor, dass {ACTOR} eine {PROCEDURE} {OBLIGATION}.",
    "{ACTOR}, die ein {AI_SYSTEM} einsetzen, {OBLIGATION} eine {PROCEDURE}.",
    "Ein {RISK_TIER} {AI_SYSTEM} erfordert, dass {ACTOR} eine {PROCEDURE} {OBLIGATION}.",
    "{ACTOR} {OBLIGATION} die Konformität des {AI_SYSTEM} mit {ARTICLE} sicherstellen.",
    "Nach {ARTICLE} {OBLIGATION} {ACTOR} die {PROCEDURE} für alle {AI_SYSTEM}e.",
    "Die {REGULATION} stuft das System als {RISK_TIER} {AI_SYSTEM} ein.",
    "{ACTOR} {OBLIGATION} eine {PROCEDURE} vor dem Inverkehrbringen des {AI_SYSTEM}.",
    "Für ein {RISK_TIER} {AI_SYSTEM} {OBLIGATION} {ACTOR} eine {PROCEDURE} nach {ARTICLE}.",
    "{ARTICLE} der {REGULATION} verpflichtet {ACTOR} zur {PROCEDURE}.",
    "Die gemäß {ARTICLE} erforderliche {PROCEDURE} ist von {ACTOR} durchzuführen.",
    "{ACTOR} sind für die {PROCEDURE} jedes {AI_SYSTEM} verantwortlich.",
    "Gemäß {ARTICLE} der {REGULATION} {OBLIGATION} {ACTOR} eine {PROCEDURE} erstellen.",
    "Ein {RISK_TIER} {AI_SYSTEM} setzt voraus, dass {ACTOR} eine {PROCEDURE} {OBLIGATION}.",
    "{ACTOR} {OBLIGATION} den Nachweis der Konformität gemäß {ARTICLE} der {REGULATION}.",
    "Die {PROCEDURE} nach {ARTICLE} gilt für alle {RISK_TIER} Systeme.",
    "Vor der Inbetriebnahme {OBLIGATION} {ACTOR} die {PROCEDURE} der {REGULATION}.",
    "{ACTOR} {OBLIGATION} das {AI_SYSTEM} nach {ARTICLE} der {REGULATION} registrieren.",
    # RISK_TIER-focused DE templates
    "Dieses System wird gemäß {ARTICLE} der {REGULATION} als {RISK_TIER} eingestuft.",
    "Ein {RISK_TIER} {AI_SYSTEM} unterliegt den Anforderungen der {REGULATION}.",
    "{ACTOR} {OBLIGATION} die {RISK_TIER}-Einstufung des {AI_SYSTEM} vor der Markteinführung prüfen.",
    "Die {REGULATION} legt fest, dass ein {RISK_TIER} {AI_SYSTEM} einer {PROCEDURE} bedarf.",
    "Die {RISK_TIER}-Klassifizierung löst zusätzliche {PROCEDURE}-Pflichten für {ACTOR} aus.",
    "Nach {ARTICLE} müssen alle {RISK_TIER} Systeme von {ACTOR} einer {PROCEDURE} unterzogen werden.",
    "{ACTOR} {OBLIGATION} den {RISK_TIER}-Status des {AI_SYSTEM} regelmäßig neu bewerten.",
    "Die {PROCEDURE} gemäß {ARTICLE} gilt ausschließlich für {RISK_TIER} Systeme.",
    # PROHIBITED_USE DE templates — Article 5 verbotene Praktiken
    "{PROHIBITED_USE} ist nach {ARTICLE} der {REGULATION} ausdrücklich verboten.",
    "{ACTOR} {OBLIGATION} sicherstellen, dass das {AI_SYSTEM} keine {PROHIBITED_USE} einsetzt.",
    "Die {REGULATION} stuft {PROHIBITED_USE} als {RISK_TIER} nach {ARTICLE} ein.",
    "{ACTOR} {OBLIGATION} die Abwesenheit von {PROHIBITED_USE} gemäß {ARTICLE} bestätigen.",
    "Jedes {AI_SYSTEM}, das {PROHIBITED_USE} einsetzt, gilt als {RISK_TIER} nach {ARTICLE}.",
    "Die {PROCEDURE} umfasst eine Prüfung auf {PROHIBITED_USE} gemäß {ARTICLE}.",
    "Gemäß {ARTICLE} {OBLIGATION} {ACTOR} das {AI_SYSTEM} auf {PROHIBITED_USE} prüfen.",
    "{PROHIBITED_USE} löst automatisch eine {RISK_TIER}-Einstufung nach {ARTICLE} aus.",
]

# ---------------------------------------------------------------------------
# Template filling — position-tracked, no str.find() required
# ---------------------------------------------------------------------------

_SLOT_RE = re.compile(r'\{([A-Z_]+)\}')


def _fill_template(
    template: str,
    vocabs: dict[str, list[str]],
    rng: random.Random,
) -> tuple[str, list[dict]]:
    """Fill a template string with random vocab values.

    Entity character offsets are computed during string assembly — exact and
    guaranteed correct without any post-hoc search.

    Returns (sentence_text, entities) where entities is a list of
    {"start": int, "end": int, "label": str} dicts.
    """
    parts: list[str] = []
    entities: list[dict] = []
    pos = 0
    last_end = 0

    for m in _SLOT_RE.finditer(template):
        static = template[last_end:m.start()]
        parts.append(static)
        pos += len(static)
        last_end = m.end()

        slot = m.group(1)
        if slot in vocabs:
            value = rng.choice(vocabs[slot])
            entities.append({"start": pos, "end": pos + len(value), "label": slot})
            parts.append(value)
            pos += len(value)
        else:
            # Unknown slot — leave placeholder as literal text
            parts.append(m.group(0))
            pos += len(m.group(0))

    parts.append(template[last_end:])
    return "".join(parts), entities


def generate_template_records(n: int, seed: int = 42) -> list[dict]:
    """Generate up to n unique records from EN + DE templates with random vocab fills."""
    rng = random.Random(seed)
    template_pool = (
        [(t, _VOCABS_EN) for t in _TEMPLATES_EN]
        + [(t, _VOCABS_DE) for t in _TEMPLATES_DE]
    )

    records: list[dict] = []
    seen: set[str] = set()
    max_attempts = n * 15

    for _ in range(max_attempts):
        if len(records) >= n:
            break
        tmpl, vocabs = rng.choice(template_pool)
        text, entities = _fill_template(tmpl, vocabs, rng)
        if text not in seen and entities:
            seen.add(text)
            records.append({"text": text, "entities": entities, "source": "generated"})

    return records


# ---------------------------------------------------------------------------
# Regulatory corpus extraction — regex + phrase matching for all 7 labels
# ---------------------------------------------------------------------------

def _parse_txt_sections(path: Path) -> dict[str, str]:
    """Parse === EN === / === DE === sections from a regulatory txt file."""
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("=== ") and stripped.endswith(" ==="):
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = stripped[4:-4].lower()
            lines = []
        elif current is not None:
            lines.append(line)
    if current is not None and lines:
        sections[current] = "\n".join(lines).strip()
    return sections


def _split_regulatory_sentences(text: str) -> list[str]:
    """Split regulatory article text into sentences, skipping the title line."""
    lines = text.splitlines()
    skip_title = True
    body_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if skip_title:
            skip_title = False
            continue
        body_lines.append(stripped)
    body = " ".join(body_lines)
    parts = re.split(r'(?<=[.!?])\s+', body)
    results: list[str] = []
    for part in parts:
        part = re.sub(r'^[\(\[]?(?:[ivxIVX]+|[a-z0-9]{1,3})[\)\]\.]\s+', '', part).strip()
        if len(part) >= 40:
            results.append(part)
    return results


# --- Entity detection patterns ---

_ARTICLE_RE = re.compile(
    r'\b(?:GDPR\s+)?(?:Article|Artikel|Art\.)\s+\d+(?:\(\d+\))?\b',
    re.IGNORECASE,
)

_OBLIGATION_RE_EN = re.compile(
    r'\b(?:'
    r'(?:shall|must)(?:\s+be)?\s+\w+'
    r'|are\s+required\s+to'
    r'|is\s+required\s+to'
    r'|are\s+obliged\s+to'
    r'|is\s+obliged\s+to'
    r'|have\s+to\s+\w+'
    r'|has\s+to\s+\w+'
    r'|need\s+to\s+\w+'
    r')\b',
    re.IGNORECASE,
)

_OBLIGATION_RE_DE = re.compile(
    r'\b(?:'
    r'(?:müssen?|sollen?)\s+\w+'
    r'|ist\s+verpflichtet'
    r'|sind\s+verpflichtet'
    r'|ist\s+zu\s+\w+'
    r'|sind\s+zu\s+\w+'
    r'|sind\s+gehalten'
    r'|ist\s+gehalten'
    r')\b',
    re.IGNORECASE,
)

_ACTOR_RE_EN = re.compile(
    r'\b(?:'
    r'providers?|operators?|importers?|manufacturers?|distributors?|deployers?'
    r'|notified\s+bod(?:y|ies)'
    r'|national\s+competent\s+authorit(?:y|ies)'
    r'|market\s+surveillance\s+authorit(?:y|ies)'
    r'|authorised\s+representatives?'
    r')\b',
    re.IGNORECASE,
)

_ACTOR_RE_DE = re.compile(
    r'\b(?:'
    r'Anbieter|Betreiber|Einführer|Hersteller|Händler|Bevollmächtigte[rn]?'
    r'|benannte\s+Stellen?'
    r'|zuständige(?:n)?\s+(?:nationale(?:n)?\s+)?Behörden?'
    r'|Marktüberwachungsbehörden?'
    r')\b',
)

# Ordered longest-first to prevent partial matches
_AI_SYSTEM_PHRASES_EN: list[str] = [
    "real-time biometric identification system",
    "remote biometric identification system",
    "general-purpose AI model",
    "biometric categorisation system",
    "emotion recognition system",
    "AI system used in critical infrastructure",
    "AI system for credit scoring",
    "AI system for recruitment",
    "AI system used in education",
    "AI system used in employment",
    "high-risk AI system",
    "prohibited AI system",
]
_AI_SYSTEM_PHRASES_DE: list[str] = [
    "KI-System mit allgemeinem Verwendungszweck",
    "biometrisches Fernidentifizierungssystem",
    "biometrisches Kategorisierungssystem",
    "Emotionserkennungssystem",
    "Hochrisiko-KI-System",
    "verbotenes KI-System",
]

_RISK_TIER_PHRASES_EN: list[str] = [
    "unacceptable risk", "high-risk", "limited risk", "minimal risk", "low risk", "prohibited",
]
_RISK_TIER_PHRASES_DE: list[str] = [
    "unannehmbares Risiko", "begrenztes Risiko", "minimales Risiko",
    "hochriskant", "Hochrisiko", "verboten",
]

_PROCEDURE_PHRASES_EN: list[str] = [
    "fundamental rights impact assessment",
    "conformity assessment procedure",
    "post-market monitoring system",
    "quality management system",
    "risk management system",
    "conformity assessment",
    "technical documentation",
    "post-market monitoring",
    "data governance",
    "human oversight",
    "incident reporting",
    "logging system",
    "transparency obligations",
]
_PROCEDURE_PHRASES_DE: list[str] = [
    "Folgenabschätzung für Grundrechte",
    "Konformitätsbewertungsverfahren",
    "Qualitätsmanagementsystem",
    "Risikomanagementsystem",
    "Konformitätsbewertung",
    "technische Dokumentation",
    "Überwachung nach dem Inverkehrbringen",
    "Datenverwaltung",
    "Protokollierungssystem",
    "menschliche Aufsicht",
    "Transparenzpflichten",
]

_REGULATION_PHRASES: list[str] = [
    "General Data Protection Regulation",
    "Artificial Intelligence Act",
    "Datenschutz-Grundverordnung",
    "Cyber Resilience Act",
    "EU AI Act",
    "AI Act",
    "GDPR",
    "DSGVO",
    "KI-Gesetz",
    "NIS2",
]

# Ordered longest-first so multi-word phrases match before substrings
_PROHIBITED_USE_PHRASES_EN: list[str] = [
    "real-time remote biometric identification in public spaces",
    "real-time biometric surveillance in public spaces",
    "emotion recognition in educational institutions",
    "emotion recognition in the workplace",
    "biometric categorisation to infer sensitive characteristics",
    "predictive policing based solely on profiling",
    "manipulation of behaviour without awareness",
    "indiscriminate scraping of facial images",
    "mass surveillance of natural persons",
    "exploitation of vulnerabilities",
    "subliminal manipulation",
    "social scoring",
]
_PROHIBITED_USE_PHRASES_DE: list[str] = [
    "biometrische Echtzeit-Fernidentifizierung im öffentlichen Raum",
    "Emotionserkennung in Bildungseinrichtungen",
    "Emotionserkennung am Arbeitsplatz",
    "biometrische Kategorisierung sensibler Merkmale",
    "prädiktive Polizeiarbeit auf Basis von Profiling",
    "Manipulation des Verhaltens ohne Bewusstsein",
    "unterschiedslose Erfassung von Gesichtsbildern",
    "Massenüberwachung natürlicher Personen",
    "Ausnutzung von Schwachstellen",
    "unterschwellige Manipulation",
    "Social Scoring",
]


def _find_entities(text: str, lang: str) -> list[dict]:
    """Find all 8 NER entity labels in a sentence.

    Returns [{"start": int, "end": int, "label": str}, ...] with verified offsets.
    Overlapping spans are dropped (longest match wins).
    """
    candidates: list[tuple[int, int, str]] = []
    text_lower = text.lower()

    # ARTICLE
    for m in _ARTICLE_RE.finditer(text):
        candidates.append((m.start(), m.end(), "ARTICLE"))

    # OBLIGATION
    ob_re = _OBLIGATION_RE_DE if lang == "de" else _OBLIGATION_RE_EN
    for m in ob_re.finditer(text):
        candidates.append((m.start(), m.end(), "OBLIGATION"))

    # ACTOR
    actor_re = _ACTOR_RE_DE if lang == "de" else _ACTOR_RE_EN
    for m in actor_re.finditer(text):
        candidates.append((m.start(), m.end(), "ACTOR"))

    # AI_SYSTEM — phrase matching, longest first
    ai_phrases = _AI_SYSTEM_PHRASES_DE if lang == "de" else _AI_SYSTEM_PHRASES_EN
    for phrase in sorted(ai_phrases, key=len, reverse=True):
        idx = text_lower.find(phrase.lower())
        while idx != -1:
            candidates.append((idx, idx + len(phrase), "AI_SYSTEM"))
            idx = text_lower.find(phrase.lower(), idx + 1)

    # RISK_TIER — phrase matching, longest first
    risk_phrases = _RISK_TIER_PHRASES_DE if lang == "de" else _RISK_TIER_PHRASES_EN
    for phrase in sorted(risk_phrases, key=len, reverse=True):
        idx = text_lower.find(phrase.lower())
        while idx != -1:
            candidates.append((idx, idx + len(phrase), "RISK_TIER"))
            idx = text_lower.find(phrase.lower(), idx + 1)

    # PROCEDURE — phrase matching, longest first
    proc_phrases = _PROCEDURE_PHRASES_DE if lang == "de" else _PROCEDURE_PHRASES_EN
    for phrase in sorted(proc_phrases, key=len, reverse=True):
        idx = text_lower.find(phrase.lower())
        while idx != -1:
            candidates.append((idx, idx + len(phrase), "PROCEDURE"))
            idx = text_lower.find(phrase.lower(), idx + 1)

    # PROHIBITED_USE — Article 5 banned practices, longest-first
    prohibited_phrases = _PROHIBITED_USE_PHRASES_DE if lang == "de" else _PROHIBITED_USE_PHRASES_EN
    for phrase in sorted(prohibited_phrases, key=len, reverse=True):
        idx = text_lower.find(phrase.lower())
        while idx != -1:
            candidates.append((idx, idx + len(phrase), "PROHIBITED_USE"))
            idx = text_lower.find(phrase.lower(), idx + 1)

    # REGULATION — case-sensitive (EU AI Act vs ai act are distinct)
    for phrase in sorted(_REGULATION_PHRASES, key=len, reverse=True):
        idx = text.find(phrase)
        while idx != -1:
            candidates.append((idx, idx + len(phrase), "REGULATION"))
            idx = text.find(phrase, idx + 1)

    if not candidates:
        return []

    # Sort by start position, prefer longer spans on tie
    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))

    # Greedy non-overlapping selection
    kept: list[tuple[int, int, str]] = []
    for start, end, label in candidates:
        if not any(s <= start < e or s < end <= e for s, e, _ in kept):
            kept.append((start, end, label))

    return [{"start": s, "end": e, "label": lbl} for s, e, lbl in sorted(kept)]


def extract_regulatory_ner_records(regulatory_dir: Path) -> list[dict]:
    """Extract NER-annotated records from all data/regulatory/**/*.txt files.

    Processes EN and DE sections. Keeps only sentences with at least one entity.
    """
    records: list[dict] = []
    for txt_file in sorted(regulatory_dir.glob("**/*.txt")):
        sections = _parse_txt_sections(txt_file)
        for lang_code in ("en", "de"):
            text = sections.get(lang_code, "")
            if not text:
                continue
            for sentence in _split_regulatory_sentences(text):
                entities = _find_entities(sentence, lang_code)
                if entities:
                    records.append({
                        "text": sentence,
                        "entities": entities,
                        "source": "regulatory",
                    })
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate spaCy NER training data (deterministic — no LLM required)"
    )
    parser.add_argument(
        "--n-templates", type=int, default=5000,
        help="Number of template-generated records to produce (default: 5000)",
    )
    parser.add_argument("--output", default="training/data/ner_annotations.jsonl")
    parser.add_argument("--overwrite", action="store_true",
                        help="Clear the output file before writing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible template fills (default: 42)")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing records to avoid duplicates
    existing_texts: set[str] = set()
    if output_path.exists() and not args.overwrite:
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_texts.add(json.loads(line)["text"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"Loaded {len(existing_texts)} existing records (will skip duplicates)")
    elif args.overwrite and output_path.exists():
        output_path.unlink()
        print("Overwrite mode: cleared existing file")

    total_written = 0

    # --- Stage 1: Real regulatory corpus ---
    print("\nExtracting real regulatory sentences...")
    reg_records = extract_regulatory_ner_records(REGULATORY_DIR)
    reg_written = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for rec in reg_records:
            if rec["text"] not in existing_texts:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                existing_texts.add(rec["text"])
                reg_written += 1
    skipped = len(reg_records) - reg_written
    print(f"  Wrote {reg_written} regulatory records"
          + (f" ({skipped} already existed)" if skipped else ""))
    total_written += reg_written

    # --- Stage 2: Template expansion ---
    print(f"\nGenerating {args.n_templates} template records (seed={args.seed})...")
    tmpl_records = generate_template_records(args.n_templates, seed=args.seed)
    tmpl_written = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for rec in tmpl_records:
            if rec["text"] not in existing_texts:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                existing_texts.add(rec["text"])
                tmpl_written += 1
    dupes = len(tmpl_records) - tmpl_written
    print(f"  Wrote {tmpl_written} template records"
          + (f" ({dupes} duplicates skipped)" if dupes else ""))
    total_written += tmpl_written

    total_lines = sum(1 for line in open(output_path, encoding="utf-8") if line.strip())
    print(f"\nDone. Total records in file: {total_lines} ({total_written} new this run)")
    print(f"Labels: ARTICLE, OBLIGATION, ACTOR, AI_SYSTEM, RISK_TIER, PROCEDURE, REGULATION")
    print(f"Output: {output_path}")
    print(f"\nNext: python training/train_ner.py --data {output_path}")


if __name__ == "__main__":
    main()
