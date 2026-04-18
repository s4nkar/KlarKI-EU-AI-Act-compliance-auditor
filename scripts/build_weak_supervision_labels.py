#!/usr/bin/env python3
"""Build training labels using weak supervision from regulatory text.

Reads data/regulatory/**/*.txt files and auto-labels sentences using
deterministic rule/regex patterns — no LLM required. Labels are APPENDED
to the existing seed JSONL files (use --overwrite to reset).

This replaces pure LLM synthetic generation for the specialist classifiers
with a grounded, reproducible pipeline that leverages the actual regulatory text.

Three label sets produced:
  actor_labels.jsonl      — provider / deployer / importer / distributor
  risk_labels.jsonl       — high_risk / not_high_risk
  prohibited_labels.jsonl — prohibited / not_prohibited

Usage:
    python scripts/build_weak_supervision_labels.py
    python scripts/build_weak_supervision_labels.py --overwrite
    python scripts/build_weak_supervision_labels.py --type actor
    python scripts/build_weak_supervision_labels.py --min-len 60 --max-len 600
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "regulatory"
OUTPUT_DIR = ROOT / "training" / "data"

# ── Sentence splitter ─────────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÜÄÖ\(\"\'])")


def _sentences(text: str, min_len: int, max_len: int) -> Iterator[str]:
    """Split text into sentences; filter by character length."""
    for sent in _SENT_SPLIT.split(text):
        sent = sent.strip()
        if min_len <= len(sent) <= max_len:
            yield sent


# ── Article 3 actor patterns (mirrors actor_classifier.py) ───────────────────

_ACTOR_RULES: list[tuple[str, str]] = [
    # provider
    (r"\b(?:we\s+)?(?:have\s+)?developed\b", "provider"),
    (r"\bplaced?\s+on\s+the\s+market\b", "provider"),
    (r"\bin\s+Verkehr\s+gebracht\b", "provider"),
    (r"\bprovider\s+means\b", "provider"),
    (r"\bAnbieter\b.*\bverlegt\b|\bAnbieter\b.*\bEntwickl", "provider"),
    (r"\bunder\s+its\s+own\s+name\s+or\s+trademark\b", "provider"),
    (r"\beigenen\s+Namen\s+oder\s+(?:ihrer|seiner)\s+eigenen\s+Marke\b", "provider"),
    (r"\bputs?\s+it\s+into\s+service\s+under\s+its\s+own\b", "provider"),
    (r"\bdevelops?\s+an?\s+AI\s+system\b", "provider"),
    (r"\bgeneral[-\s]purpose\s+AI\s+model\b.*\bplaces?\b", "provider"),
    # deployer
    (r"\bdeployer\s+means\b", "deployer"),
    (r"\bBetreiber\b.*\bverantwortung\b|\bBetreiber\b.*\bVerwendung\b", "deployer"),
    (r"\buses?\s+an?\s+AI\s+system\s+under\s+its\s+authority\b", "deployer"),
    (r"\bin\s+eigener\s+Verantwortung\s+verwendet\b", "deployer"),
    (r"\boperates?\s+(?:the\s+)?AI\s+system\s+under\s+(?:its|their)\s+authority\b", "deployer"),
    (r"\bdeployer\b.*\b(?:operates|uses|deploys)\b", "deployer"),
    # importer
    (r"\bimporter\s+means\b", "importer"),
    (r"\bEinführer\b", "importer"),
    (r"\bestablished\s+in\s+the\s+Union\b.*\bplaces?\b", "importer"),
    (r"\bin\s+der\s+Union\s+niedergelassen\b.*\bin\s+Verkehr\b", "importer"),
    (r"\bestablished\s+outside\s+the\s+Union\b", "importer"),
    (r"\baußerhalb\s+der\s+Union\s+niedergelass", "importer"),
    # distributor
    (r"\bdistributor\s+means\b", "distributor"),
    (r"\bHändler\b", "distributor"),
    (r"\bmakes?\s+an?\s+AI\s+system\s+available\s+on\s+the\s+(?:Union\s+)?market\b", "distributor"),
    (r"\bauf\s+dem\s+(?:Unions)?markt\s+bereitstellt\b", "distributor"),
    (r"\bwithout\s+affecting\s+its\s+properties\b", "distributor"),
    (r"\bohne\s+seine\s+Eigenschaften\s+zu\s+verändern\b", "distributor"),
    (r"\bin\s+the\s+supply\s+chain\b.*\bdistributor\b", "distributor"),
]

_ACTOR_COMPILED = [(re.compile(p, re.I), label) for p, label in _ACTOR_RULES]


def _label_actor(text: str) -> str | None:
    votes: dict[str, int] = {}
    for pattern, label in _ACTOR_COMPILED:
        if pattern.search(text):
            votes[label] = votes.get(label, 0) + 1
    if not votes:
        return None
    return max(votes, key=lambda k: votes[k])


# ── Article 6 + Annex III risk patterns ──────────────────────────────────────

_HIGH_RISK_PATTERNS: list[str] = [
    r"\bhigh[-\s]risk\s+AI\s+system\b",
    r"\bHochrisiko[-\s]KI[-\s]System\b",
    r"\bAnnex\s+III\b",
    r"\bAnnex\s+I\s+product\b",
    r"\bbiometric\s+(?:identification|recognition|categoris)",
    r"\bGesichtserkennung\b",
    r"\bcritical\s+infrastructure\b",
    r"\bkritische\s+Infrastruktur\b",
    r"\bstudent\s+(?:admission|selection|assessment)\b",
    r"\bZulassung\s+von\s+Studierenden\b",
    r"\b(?:hiring|firing|recruitment)\s+(?:decision|AI|system|tool)\b",
    r"\bKreditvergabe\b",
    r"\b(?:credit|insurance|benefit)\s+(?:scoring|eligibility|decision)\b",
    r"\blaw\s+enforcement\b.*\b(?:AI|predict|profil)\b",
    r"\bStrafverfolgung\b.*\b(?:KI|Vorhersage|Profil)\b",
    r"\b(?:migration|asylum|border\s+control)\b.*\b(?:AI|system|decision)\b",
    r"\bjustice\b.*\b(?:AI|decision|prediction)\b",
    r"\brichterliche\s+Entscheidung\b",
    r"\bArticle\s+6\b.*\bhigh[-\s]risk\b",
    r"\bhigh[-\s]risk\b.*\bArticle\s+6\b",
    r"\bArtikel\s+6\b.*\bHochrisiko\b",
    r"\bconformity\s+assessment\b",
    r"\bKonformitätsbewertung\b",
]

_NOT_HIGH_RISK_PATTERNS: list[str] = [
    r"\b(?:spam\s+filter|chatbot|recommendation\s+engine)\b",
    r"\bcontent\s+(?:generation|moderation)\s+(?:AI|system|tool)\b",
    r"\btranslation\s+(?:AI|system|tool|software)\b",
    r"\bpredictive\s+maintenance\b",
    r"\bcustomer\s+service\s+(?:bot|AI|chatbot)\b",
    r"\binventory\s+(?:optimis|management)\b.*\bAI\b",
    r"\bnot\s+(?:considered\s+)?(?:a\s+)?high[-\s]risk\b",
    r"\bminimal\s+risk\s+AI\b",
    r"\blimited\s+risk\s+AI\b",
    r"\btransparency\s+obligation\b.*\bnot\b",
    r"\bArticle\s+52\b.*\btransparency\b",
    r"\bexcluded\s+from\s+(?:the\s+)?(?:high[-\s]risk|Annex\s+III)\b",
    r"\bKundendienst[-\s]Bot\b",
    r"\bSpam[-\s]Filter\b",
    r"\bInhaltsempfehlung\b",
    r"\bweder\s+(?:als\s+)?Hochrisiko\b",
]

_HIGH_RISK_COMPILED = [re.compile(p, re.I) for p in _HIGH_RISK_PATTERNS]
_NOT_HIGH_RISK_COMPILED = [re.compile(p, re.I) for p in _NOT_HIGH_RISK_PATTERNS]


def _label_risk(text: str) -> str | None:
    hr_hits = sum(1 for p in _HIGH_RISK_COMPILED if p.search(text))
    nhr_hits = sum(1 for p in _NOT_HIGH_RISK_COMPILED if p.search(text))
    if hr_hits == 0 and nhr_hits == 0:
        return None
    if hr_hits > nhr_hits:
        return "high_risk"
    if nhr_hits > hr_hits:
        return "not_high_risk"
    return None  # tie — skip ambiguous


# ── Article 5 prohibited practice patterns ────────────────────────────────────

_PROHIBITED_PATTERNS: list[str] = [
    r"\bsubliminal\s+technique[s]?\b",
    r"\bsocial\s+scor(?:ing|e)\b",
    r"\breal[-\s]time\s+(?:remote\s+)?biometric\s+identification\b",
    r"\bemotion\s+recogni(?:tion|se|ze)\b.*\b(?:workplace|school|education|educational)\b",
    r"\b(?:workplace|school|education)\b.*\bemotion\s+recogni(?:tion|se|ze)\b",
    r"\bmanipulat\w+\s+(?:human\s+)?behaviour\b.*\b(?:harm|damage|circumvent)\b",
    r"\bexploit\s+(?:the\s+)?vulnerabilit\w+\b.*\b(?:group|person|individual)\b",
    r"\bEchtzeitbiometrie\b",
    r"\bSocial[-\s]Scoring\b",
    r"\bunterbewusste\s+Techniken\b",
    r"\bEmotionserkennung\b.*\b(?:Arbeitsplatz|Bildungseinrichtung|Schule)\b",
    r"\bverboten(?:e|en|er|em)?\b.*\b(?:KI|Praktik|Anwendung)\b",
    r"\bprohibited\s+(?:AI\s+)?practice[s]?\b",
    r"\bArticle\s+5\b.*\bprohibit\b",
    r"\bprohibit\b.*\bArticle\s+5\b",
    r"\bArtikel\s+5\b.*\bverboten\b",
    r"\bmass\s+(?:surveillance|monitoring)\b.*\b(?:AI|biometric|face)\b",
]

_NOT_PROHIBITED_PATTERNS: list[str] = [
    r"\bbiometric\s+authentication\b",
    r"\baccess\s+control\b.*\bbiometric\b",
    r"\bsentiment\s+analysis\b.*\bcustomer\b",
    r"\bfraud\s+detection\b",
    r"\b(?:medical|clinical)\s+diagnosis\b.*\bemotion\b",
    r"\bwith\s+(?:explicit\s+)?(?:informed\s+)?consent\b.*\bemotion\b",
    r"\bcredit\s+scoring\b(?!\s+is\s+prohibit)",
    r"\bpersonali[sz]ed\s+recommendation\b",
    r"\bcontent\s+moderation\b",
    r"\bsafety\s+monitoring\b.*\bfactor(?:y|ies)\b",
    r"\bborder\s+control\b.*\blegal\s+basis\b",
    r"\bnot\s+(?:a\s+)?prohibited\b",
    r"\bnicht\s+(?:als\s+)?verboten\b",
    r"\blegal\s+(?:basis|framework)\b.*\bbiometric\b",
    r"\bzulässig\b.*\b(?:biometrisch|Erkennung)\b",
]

_PROHIBITED_COMPILED = [re.compile(p, re.I) for p in _PROHIBITED_PATTERNS]
_NOT_PROHIBITED_COMPILED = [re.compile(p, re.I) for p in _NOT_PROHIBITED_PATTERNS]


def _label_prohibited(text: str) -> str | None:
    p_hits = sum(1 for pat in _PROHIBITED_COMPILED if pat.search(text))
    np_hits = sum(1 for pat in _NOT_PROHIBITED_COMPILED if pat.search(text))
    if p_hits == 0 and np_hits == 0:
        return None
    if p_hits > np_hits:
        return "prohibited"
    if np_hits > p_hits:
        return "not_prohibited"
    return None


# ── File parsing ──────────────────────────────────────────────────────────────

_LANG_SECTION = re.compile(r"^=== (EN|DE) ===\s*$", re.M)


def _parse_regulatory_file(path: Path) -> list[tuple[str, str]]:
    """Return [(text, lang), ...] sentence pairs from a regulatory .txt file."""
    raw = path.read_text(encoding="utf-8")

    # Split on language section markers
    parts = _LANG_SECTION.split(raw)
    # parts: [preamble, "EN", en_text, "DE", de_text, ...]
    results: list[tuple[str, str]] = []
    i = 1
    while i + 1 < len(parts):
        lang_code = parts[i].strip().lower()
        body = parts[i + 1]
        if lang_code in ("en", "de"):
            results.append((body.strip(), lang_code))
        i += 2

    return results


# ── Main pipeline ─────────────────────────────────────────────────────────────

LABELERS = {
    "actor": (_label_actor, "actor_labels.jsonl"),
    "risk": (_label_risk, "risk_labels.jsonl"),
    "prohibited": (_label_prohibited, "prohibited_labels.jsonl"),
}


def _load_existing(path: Path) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    texts = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    texts.add(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return frozenset(texts)


def build_labels(
    classifier_type: str,
    min_len: int,
    max_len: int,
    overwrite: bool,
    verbose: bool,
) -> None:
    labeler, output_file = LABELERS[classifier_type]
    output_path = OUTPUT_DIR / output_file

    existing_texts = frozenset() if overwrite else _load_existing(output_path)
    print(f"\n[{classifier_type.upper()}] {output_path.name}")
    print(f"  Existing examples: {len(existing_texts)}")

    new_examples: list[dict] = []
    label_counts: dict[str, int] = {}

    reg_files = sorted(DATA_DIR.rglob("*.txt"))
    if not reg_files:
        print(f"  WARNING: No regulatory .txt files found in {DATA_DIR}")
        return

    for reg_file in reg_files:
        lang_sections = _parse_regulatory_file(reg_file)
        for body, lang in lang_sections:
            for sent in _sentences(body, min_len, max_len):
                if sent in existing_texts:
                    continue
                label = labeler(sent)
                if label is None:
                    continue
                if verbose:
                    print(f"    [{label}] {sent[:80]!r}")
                new_examples.append({
                    "text": sent,
                    "label": label,
                    "lang": lang,
                    "source": "regulatory",
                })
                label_counts[label] = label_counts.get(label, 0) + 1

    # Deduplicate within new_examples (same sentence from multiple files)
    seen: set[str] = set(existing_texts)
    deduped: list[dict] = []
    for ex in new_examples:
        if ex["text"] not in seen:
            seen.add(ex["text"])
            deduped.append(ex)

    mode = "w" if overwrite else "a"
    with open(output_path, mode, encoding="utf-8") as f:
        for ex in deduped:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  New examples added: {len(deduped)}")
    print(f"  Label distribution: {label_counts}")
    print(f"  Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build weak supervision training labels from regulatory text (no LLM)"
    )
    parser.add_argument(
        "--type", choices=["actor", "risk", "prohibited", "all"], default="all",
        help="Which classifier labels to build (default: all)",
    )
    parser.add_argument("--min-len", type=int, default=60,
                        help="Minimum sentence character length (default: 60)")
    parser.add_argument("--max-len", type=int, default=600,
                        help="Maximum sentence character length (default: 600)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Clear output files before writing (default: append)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each labeled sentence")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    types = list(LABELERS.keys()) if args.type == "all" else [args.type]
    for t in types:
        build_labels(
            classifier_type=t,
            min_len=args.min_len,
            max_len=args.max_len,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )

    print("\nDone. Next step:")
    print("  python scripts/generate_specialist_training_data.py  # augment with LLM synthetic")
    print("  python training/train_specialist_classifiers.py --type actor")


if __name__ == "__main__":
    main()
