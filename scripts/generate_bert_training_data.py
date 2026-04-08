#!/usr/bin/env python3
"""Auto-generate BERT classifier training data from EU AI Act + GDPR regulatory text.

Uses Ollama (the same LLM already running) to produce synthetic labeled
compliance document sentences for each of the 8 article domains.

Each prompt includes the actual regulation text as reference, so generated
examples are grounded in the official requirements rather than generic
descriptions. This produces more accurate, varied, and realistic training data.

Target: 400 examples per class x 8 classes x 2 languages = 6,400 examples.
Generated examples are APPENDED to clause_labels.jsonl (hand-crafted examples
are preserved). Duplicates are deduplicated by text content.

Usage:
    # Default: 400 per class, EN + DE
    python scripts/generate_bert_training_data.py

    # Faster smoke test (20 per class)
    python scripts/generate_bert_training_data.py --n-per-class 20

    # Overwrite instead of append
    python scripts/generate_bert_training_data.py --overwrite

    # Custom Ollama host
    python scripts/generate_bert_training_data.py --ollama-host http://localhost:11434

Run this BEFORE train_classifier.py.
setup.py runs it automatically as the 'generate-data' stage.
"""

import argparse
import json
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
REGULATORY_DIR = ROOT / "data" / "regulatory"

# Each classifier label maps to a primary regulation article.
# The article text is loaded and included in the generation prompt as reference,
# grounding synthetic examples in the official regulation language.
DOMAIN_ARTICLE_MAP: dict[str, tuple[str, int] | None] = {
    "risk_management":         ("eu_ai_act", 9),
    "data_governance":         ("eu_ai_act", 10),
    "technical_documentation": ("eu_ai_act", 11),
    "record_keeping":          ("eu_ai_act", 12),
    "transparency":            ("eu_ai_act", 13),
    "human_oversight":         ("eu_ai_act", 14),
    "security":                ("eu_ai_act", 15),
    "unrelated":               None,
}

DOMAINS = [
    {
        "label": "risk_management",
        "article": "Article 9",
        "description": "Identifying, evaluating, and mitigating risks of an AI system. Key concepts: risk register, hazard identification, risk analysis, residual risk, foreseeable misuse, risk mitigation measures, lifecycle risk assessment, safety risk, harm potential.",
        "not_confused_with": "technical_documentation (which describes HOW the system works, not what can go wrong). A sentence about risk management will mention hazards, risk levels, mitigation, or potential harms — not system architecture or component descriptions.",
    },
    {
        "label": "data_governance",
        "article": "Article 10",
        "description": "Quality, sourcing, and management of training/validation data. Key concepts: training data quality, dataset documentation, bias examination, data representativeness, data labelling, data collection processes, validation datasets, ground truth.",
        "not_confused_with": "technical_documentation (which covers the full system design, not specifically data). Sentences are about the DATA ITSELF — its quality, collection, labelling, bias.",
    },
    {
        "label": "technical_documentation",
        "article": "Article 11",
        "description": "Written documentation describing the AI system design and architecture. Key concepts: system architecture, component specifications, design documents, model cards, Annex IV documentation, conformity assessment, development methodology, system description.",
        "not_confused_with": "risk_management (hazards/harms) and human_oversight (human control). A sentence about technical_documentation will mention documents, specifications, architecture diagrams, design files, or system descriptions — NOT risks or human control.",
    },
    {
        "label": "record_keeping",
        "article": "Article 12",
        "description": "Automated logging and retention of operational records. Key concepts: audit logs, event logs, automatic recording, log retention policy, traceability, prediction logs, timestamp records, monitoring logs, logging requirements.",
        "not_confused_with": "technical_documentation (design docs) or transparency (user-facing disclosure). Sentences are specifically about LOGS that are automatically generated during system operation.",
    },
    {
        "label": "transparency",
        "article": "Article 13",
        "description": "Providing information to users and deployers about the AI system. Key concepts: user disclosure, instructions for use, AI disclosure notice, capability limitations, automated decision notification, deployer information package, transparency obligations.",
        "not_confused_with": "human_oversight (which is about control/override, not disclosure) and technical_documentation (internal design docs, not user-facing info). Sentences are about INFORMING users or the public.",
    },
    {
        "label": "human_oversight",
        "article": "Article 14",
        "description": "Mechanisms for humans to monitor, intervene in, or override AI decisions. Key concepts: human review, override capability, stop/pause button, human-in-the-loop, oversight procedures, manual intervention, deactivation mechanism, competent natural person.",
        "not_confused_with": "transparency (which is about disclosure) and technical_documentation (which describes system components). Sentences are specifically about a HUMAN BEING ABLE TO CONTROL or stop the system.",
    },
    {
        "label": "security",
        "article": "Article 15",
        "description": "Resilience and protection of the AI system against attacks and failures. Key concepts: adversarial robustness, cybersecurity, data poisoning prevention, fail-safe mechanisms, accuracy under attack, resilience testing, security vulnerabilities, evasion attacks.",
        "not_confused_with": "risk_management (general harms) and technical_documentation (system design). Sentences focus specifically on ATTACKS, cyber threats, robustness against adversarial inputs, or failsafe protections.",
    },
    {
        "label": "unrelated",
        "article": "none",
        "description": "General business text completely unrelated to AI compliance. Examples: financial reports, quarterly results, HR leave policies, marketing campaigns, supply chain logistics, office administration, customer service scripts.",
        "not_confused_with": "Any AI compliance class. Must NOT mention AI, machine learning, algorithms, risk management, data governance, or regulatory compliance.",
    },
]

LANGUAGES = [
    ("en", "English"),
    ("de", "German"),
]

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


def load_article_text(regulation: str, article_num: int, lang: str = "en") -> str:
    """Load article text from data/regulatory/<regulation>/article_<num>.txt.

    Args:
        regulation: Regulation folder name, e.g. 'eu_ai_act'.
        article_num: Article number, e.g. 9.
        lang: Language code ('en' or 'de'). Falls back to 'en' if DE not found.

    Returns:
        Article text string, or empty string if file not found.
    """
    path = REGULATORY_DIR / regulation / f"article_{article_num}.txt"
    if not path.exists():
        return ""
    try:
        sections = _parse_txt_sections(path)
        return sections.get(lang, sections.get("en", ""))
    except Exception:
        return ""


_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0)


def ollama_generate(host: str, model: str, prompt: str) -> str:
    """Synchronous Ollama generate call."""
    payload = {"model": model, "prompt": prompt, "stream": False, "format": "json"}
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(f"{host}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


def parse_json_list(raw: str) -> list[str]:
    """Extract a JSON array from LLM output, tolerating surrounding text."""
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(s) for s in result if s]
        if isinstance(result, dict):
            for key in ("sentences", "examples", "texts", "items", "data"):
                if key in result and isinstance(result[key], list):
                    return [str(s) for s in result[key] if s]
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(s) for s in result if s]
        except json.JSONDecodeError:
            pass

    # Line-by-line fallback
    lines = []
    for line in raw.splitlines():
        line = line.strip().strip('",').strip('"').strip("'")
        if len(line) > 20:
            lines.append(line)
    return lines


def build_prompt(domain: dict, language_name: str, lang_code: str, n: int) -> str:
    """Build a generation prompt anchored to the actual regulation text.

    Loads article text from data/regulatory/ and includes a truncated excerpt
    as reference. This grounds the generated examples in official language rather
    than vague domain descriptions, producing more accurate training data.
    """
    regulation, article_num = DOMAIN_ARTICLE_MAP.get(domain["label"]) or (None, None)

    # Load regulation reference text (EN always; DE for German prompts)
    article_ref = ""
    if regulation and article_num:
        ref_lang = lang_code if lang_code in ("en", "de") else "en"
        text = load_article_text(regulation, article_num, ref_lang)
        if not text:
            text = load_article_text(regulation, article_num, "en")
        if text:
            # Truncate to ~800 chars at a sentence boundary
            if len(text) > 800:
                truncated = text[:800].rsplit(".", 1)[0] + "."
            else:
                truncated = text
            article_ref = f"\nReference regulation text:\n{truncated}\n"

    lang_instruction = f"Generate all {n} sentences in {language_name}."

    if domain["label"] == "unrelated":
        return (
            f'You are generating training data for a compliance classification model.\n'
            f'\n'
            f'Label: UNRELATED\n'
            f'\n'
            f'Generate {n} short sentences (10-30 words) that could appear in ordinary '
            f'business documents such as financial reports, HR policies, marketing copy, '
            f'supply-chain procedures, or administrative memos. '
            f'These sentences must NOT relate to AI compliance, AI systems, risk management, '
            f'data governance, or cybersecurity. {lang_instruction}\n'
            f'\n'
            f'Requirements:\n'
            f'- Vary topics widely (finance, HR, marketing, operations, legal admin)\n'
            f'- Keep sentences independent and realistic\n'
            f'- Do not mention AI, machine learning, or EU AI Act\n'
            f'\n'
            f'Return ONLY a JSON array of {n} strings:\n'
            f'{{"sentences": ["...", "..."]}}'
        )

    not_confused = domain.get("not_confused_with", "")
    disambiguation = f"\nIMPORTANT — do NOT generate sentences that could belong to other classes:\n{not_confused}\n" if not_confused else ""

    return (
        f'You are generating training data for a compliance classification model.\n'
        f'\n'
        f'Label: {domain["label"].upper()}\n'
        f'Regulation: EU AI Act {domain["article"]}\n'
        f'Description: {domain["description"]}\n'
        f'{article_ref}'
        f'{disambiguation}\n'
        f'Generate {n} short sentences (10-30 words) that could appear in an AI system\'s '
        f'compliance documentation, internal policy, or audit report. {lang_instruction}\n'
        f'\n'
        f'Requirements:\n'
        f'- Each sentence must UNAMBIGUOUSLY belong to {domain["label"].upper()} — a human expert must agree\n'
        f'- Use realistic compliance language (policy declarations, process steps, audit findings)\n'
        f'- Avoid repeating phrases or synonymous rewrites\n'
        f'- Vary sentence structure (statements, imperatives, passive constructions)\n'
        f'- Keep sentences independent — each must stand alone\n'
        f'- Do NOT copy or paraphrase the reference text directly\n'
        f'\n'
        f'Return ONLY a JSON object:\n'
        f'{{"sentences": ["...", "..."]}}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-generate BERT training data via Ollama")
    parser.add_argument("--n-per-class", type=int, default=400,
                        help="Examples to generate per class per language (default: 400)")
    parser.add_argument("--output",  default="training/data/clause_labels.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output file instead of appending")
    parser.add_argument("--ollama-host",  default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="phi3:mini")
    parser.add_argument("--languages",   default="en,de",
                        help="Comma-separated language codes to generate (default: en,de)")
    parser.add_argument("--batch-size",  type=int, default=20,
                        help="Examples to request per Ollama call (default: 20)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    active_langs = [(c, n) for c, n in LANGUAGES if c in args.languages.split(",")]

    # Load existing examples to avoid duplicates
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
        print(f"Loaded {len(existing_texts)} existing examples (will skip duplicates)")
    elif args.overwrite and output_path.exists():
        output_path.unlink()
        print("Overwrite mode: cleared existing file")

    # Verify Ollama is reachable
    print(f"\nChecking Ollama at {args.ollama_host}...")
    try:
        with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
            resp = client.get(f"{args.ollama_host}/api/tags")
            resp.raise_for_status()
        print("  Ollama is ready.")
    except Exception as exc:
        print(f"  ERROR: Cannot reach Ollama: {exc}")
        print("  Start containers first: docker compose up -d")
        raise SystemExit(1)

    # Report which article files were found
    print("\nRegulatory reference files:")
    for domain in DOMAINS:
        article_ref = DOMAIN_ARTICLE_MAP.get(domain["label"])
        if article_ref:
            reg, num = article_ref
            text = load_article_text(reg, num, "en")
            status = f"found ({len(text)} chars)" if text else "NOT FOUND (will use description only)"
            print(f"  {domain['label']}: {reg}/article_{num}.txt — {status}")
        else:
            print(f"  {domain['label']}: no article reference (unrelated class)")

    total_written = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for domain in DOMAINS:
            for lang_code, lang_name in active_langs:
                needed = args.n_per_class
                generated: list[str] = []

                print(f"\n  [{domain['label']}] [{lang_code}] generating {needed} examples...")

                attempts = 0
                max_attempts = (needed // args.batch_size + 2) * 2

                while len(generated) < needed and attempts < max_attempts:
                    batch_n = min(args.batch_size, needed - len(generated) + 5)
                    prompt = build_prompt(domain, lang_name, lang_code, batch_n)

                    try:
                        raw = ollama_generate(args.ollama_host, args.ollama_model, prompt)
                        batch = parse_json_list(raw)

                        new_count = 0
                        for text in batch:
                            text = text.strip()
                            if len(text) < 20:
                                continue
                            if text not in existing_texts and text not in generated:
                                generated.append(text)
                                new_count += 1

                        print(f"    batch {attempts+1}: got {len(batch)}, kept {new_count} new "
                              f"(total {len(generated)}/{needed})")

                    except Exception as exc:
                        print(f"    WARNING: Ollama call failed: {exc}")
                        time.sleep(2)

                    attempts += 1

                written = 0
                for text in generated[:needed]:
                    record = {"text": text, "label": domain["label"], "lang": lang_code, "source": "generated"}
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    existing_texts.add(text)
                    written += 1

                total_written += written
                print(f"    wrote {written} examples for [{domain['label']}][{lang_code}]")

    print(f"\nDone.")
    print(f"  Written: {total_written} new examples")
    print(f"  Output:  {output_path}")

    total_lines = sum(1 for line in open(output_path, encoding="utf-8") if line.strip())
    print(f"  Total in file: {total_lines} examples")
    print(f"\nNext: python training/train_classifier.py --data {output_path}")


if __name__ == "__main__":
    main()
