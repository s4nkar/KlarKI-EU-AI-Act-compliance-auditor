#!/usr/bin/env python3
"""Auto-generate spaCy NER training data from EU AI Act + GDPR regulatory text.

Uses Ollama to produce synthetic annotated sentences for 4 entity labels:
  ARTICLE      -- Article references  (e.g. "Article 9", "Artikel 13")
  OBLIGATION   -- Compliance duties   (e.g. "must document", "shall maintain")
  RISK_TIER    -- Risk classifications (e.g. "high-risk", "prohibited")
  REGULATION   -- Regulation names    (e.g. "EU AI Act", "GDPR", "DSGVO")

Ollama generates sentences with entity text spans. Character offsets are
computed programmatically via str.find() -- never trusted from the LLM.
Only spans that can be found verbatim in the sentence text are kept.

Usage:
    python scripts/generate_ner_data.py

    # Faster smoke test
    python scripts/generate_ner_data.py --n-per-label 10

    # Custom Ollama
    python scripts/generate_ner_data.py --ollama-host http://localhost:11434

Run BEFORE training/train_ner.py.
setup.py runs it automatically as the 'generate-ner-data' stage.
"""

import argparse
import json
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent

_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0)

ENTITY_CONFIGS = [
    {
        "label": "ARTICLE",
        "description": "References to specific regulation articles, e.g. 'Article 9', 'Article 13', 'Artikel 10', 'Art. 14', 'GDPR Article 35'.",
        "examples_en": [
            '{"text": "Article 9 establishes the risk management system requirements.", "entities": [{"span": "Article 9", "label": "ARTICLE"}]}',
            '{"text": "Both Article 13 and Article 14 address operator responsibilities.", "entities": [{"span": "Article 13", "label": "ARTICLE"}, {"span": "Article 14", "label": "ARTICLE"}]}',
        ],
        "examples_de": [
            '{"text": "Artikel 9 legt die Anforderungen an das Risikomanagement fest.", "entities": [{"span": "Artikel 9", "label": "ARTICLE"}]}',
            '{"text": "Gemaess Artikel 13 sind Transparenzpflichten zu erfuellen.", "entities": [{"span": "Artikel 13", "label": "ARTICLE"}]}',
        ],
    },
    {
        "label": "OBLIGATION",
        "description": "Compliance obligations or duties: phrases like 'must document', 'shall maintain', 'are required to', 'obligated to report', 'must implement'.",
        "examples_en": [
            '{"text": "Providers must document all training data sources.", "entities": [{"span": "must document", "label": "OBLIGATION"}]}',
            '{"text": "The operator shall maintain an audit trail of system decisions.", "entities": [{"span": "shall maintain", "label": "OBLIGATION"}]}',
        ],
        "examples_de": [
            '{"text": "Anbieter muessen alle Trainingsdatenquellen dokumentieren.", "entities": [{"span": "muessen alle Trainingsdatenquellen dokumentieren", "label": "OBLIGATION"}]}',
            '{"text": "Der Betreiber muss ein Auditprotokoll fuehren.", "entities": [{"span": "muss ein Auditprotokoll fuehren", "label": "OBLIGATION"}]}',
        ],
    },
    {
        "label": "RISK_TIER",
        "description": "Risk classification labels: 'high-risk', 'prohibited', 'limited risk', 'minimal risk', 'hochriskant', 'verboten', 'begrenztes Risiko', 'minimales Risiko'.",
        "examples_en": [
            '{"text": "This AI system has been classified as high-risk under the EU AI Act.", "entities": [{"span": "high-risk", "label": "RISK_TIER"}]}',
            '{"text": "Real-time biometric surveillance in public spaces is prohibited.", "entities": [{"span": "prohibited", "label": "RISK_TIER"}]}',
        ],
        "examples_de": [
            '{"text": "Dieses System wurde als hochriskant eingestuft.", "entities": [{"span": "hochriskant", "label": "RISK_TIER"}]}',
            '{"text": "Die Verwendung von Social-Scoring-Systemen ist verboten.", "entities": [{"span": "verboten", "label": "RISK_TIER"}]}',
        ],
    },
    {
        "label": "REGULATION",
        "description": "Regulation or directive names: 'EU AI Act', 'AI Act', 'GDPR', 'General Data Protection Regulation', 'DSGVO', 'Datenschutz-Grundverordnung', 'KI-Gesetz', 'Artificial Intelligence Act'.",
        "examples_en": [
            '{"text": "The EU AI Act introduces a risk-based framework for AI systems.", "entities": [{"span": "EU AI Act", "label": "REGULATION"}]}',
            '{"text": "Compliance requires adherence to both the EU AI Act and GDPR.", "entities": [{"span": "EU AI Act", "label": "REGULATION"}, {"span": "GDPR", "label": "REGULATION"}]}',
        ],
        "examples_de": [
            '{"text": "Das KI-Gesetz fuehrt einen risikobasierten Rahmen fuer KI ein.", "entities": [{"span": "KI-Gesetz", "label": "REGULATION"}]}',
            '{"text": "Die DSGVO und das KI-Gesetz gelten gemeinsam fuer diese Anwendung.", "entities": [{"span": "DSGVO", "label": "REGULATION"}, {"span": "KI-Gesetz", "label": "REGULATION"}]}',
        ],
    },
]

LANGUAGES = [("en", "English"), ("de", "German")]


def ollama_generate(host: str, model: str, prompt: str) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False, "format": "json"}
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(f"{host}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


def check_ollama(host: str) -> None:
    try:
        with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
            resp = client.get(f"{host}/api/tags")
            resp.raise_for_status()
        print("  Ollama is ready.")
    except Exception as exc:
        print(f"  ERROR: Cannot reach Ollama: {exc}")
        raise SystemExit(1)


def build_prompt(config: dict, lang_code: str, lang_name: str, n: int) -> str:
    label = config["label"]
    description = config["description"]
    examples = config[f"examples_{lang_code}"]
    example_block = "\n".join(examples)

    return (
        f"You are generating NER training data for EU AI Act compliance documents.\n"
        f"\n"
        f"Entity label: {label}\n"
        f"Description: {description}\n"
        f"\n"
        f"Examples of correct output:\n"
        f"{example_block}\n"
        f"\n"
        f"Generate {n} NEW sentences in {lang_name} that each contain at least one {label} entity.\n"
        f"Each sentence should be realistic compliance document language (policy texts, audit reports, technical documentation).\n"
        f"\n"
        f"Rules:\n"
        f"- Each 'span' value MUST appear verbatim in the 'text' field (exact substring match).\n"
        f"- Sentences should be 10-40 words long.\n"
        f"- Vary sentence structure and vocabulary.\n"
        f"- Do NOT copy the example sentences.\n"
        f"\n"
        f"Return ONLY a JSON array of {n} objects, each with 'text' and 'entities' fields:\n"
        f'{{"records": [{{"text": "...", "entities": [{{"span": "...", "label": "{label}"}}]}}]}}'
    )


def _extract_records_from_raw(raw: str) -> list[dict]:
    """Try multiple strategies to extract records from LLM JSON output."""
    # Strategy 1: parse as-is
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "records" in data:
            return data["records"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Strategy 2: find first JSON object
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            data = json.loads(raw[start:end])
            if isinstance(data, dict) and "records" in data:
                return data["records"]
        except json.JSONDecodeError:
            pass

    # Strategy 3: find all individual JSON objects in the output
    records = []
    for match in re.finditer(r'\{[^{}]*"text"[^{}]*"entities"[^{}]*\}', raw, re.DOTALL):
        try:
            obj = json.loads(match.group())
            if "text" in obj and "entities" in obj:
                records.append(obj)
        except json.JSONDecodeError:
            pass

    return records


def parse_records(raw: str, label: str) -> list[dict]:
    """Parse LLM output into annotation records with verified char offsets.

    Only keeps spans that are found verbatim in the sentence text.
    Returns records in the format expected by train_ner.py:
      {"text": "...", "entities": [{"start": int, "end": int, "label": str}]}
    """
    raw_records = _extract_records_from_raw(raw)
    results = []

    for rec in raw_records:
        text = str(rec.get("text", "")).strip()
        if not text or len(text) < 15:
            continue

        entities_raw = rec.get("entities", [])
        entities = []

        for ent in entities_raw:
            span_text = str(ent.get("span", "")).strip()
            ent_label = str(ent.get("label", label)).strip()
            if not span_text:
                continue
            idx = text.find(span_text)
            if idx == -1:
                # Try case-insensitive search and use the actual substring
                lower_idx = text.lower().find(span_text.lower())
                if lower_idx != -1:
                    actual = text[lower_idx:lower_idx + len(span_text)]
                    entities.append({"start": lower_idx, "end": lower_idx + len(actual), "label": ent_label})
            else:
                entities.append({"start": idx, "end": idx + len(span_text), "label": ent_label})

        if entities:  # only keep sentences where at least one entity was resolved
            results.append({"text": text, "entities": entities})

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate spaCy NER training data via Ollama")
    parser.add_argument("--n-per-label", type=int, default=40,
                        help="Sentences per label per language (default: 40, total ~320)")
    parser.add_argument("--output", default="training/data/ner_annotations.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="phi3:mini")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Sentences to request per Ollama call (default: 10)")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Checking Ollama at {args.ollama_host}...")
    check_ollama(args.ollama_host)

    # Load existing to avoid duplicates
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

    with open(output_path, "a", encoding="utf-8") as out_f:
        for config in ENTITY_CONFIGS:
            for lang_code, lang_name in LANGUAGES:
                needed = args.n_per_label
                collected: list[dict] = []

                print(f"\n  [{config['label']}] [{lang_code}] generating {needed} sentences...")

                max_attempts = (needed // args.batch_size + 3) * 3
                attempts = 0

                while len(collected) < needed and attempts < max_attempts:
                    batch_n = min(args.batch_size, needed - len(collected) + 3)
                    prompt = build_prompt(config, lang_code, lang_name, batch_n)

                    try:
                        raw = ollama_generate(args.ollama_host, args.ollama_model, prompt)
                        batch = parse_records(raw, config["label"])

                        new_count = 0
                        for rec in batch:
                            if rec["text"] not in existing_texts:
                                t = rec["text"]
                                if not any(c["text"] == t for c in collected):
                                    collected.append(rec)
                                    new_count += 1

                        print(f"    batch {attempts + 1}: got {len(batch)} valid, kept {new_count} new "
                              f"(total {len(collected)}/{needed})")

                    except Exception as exc:
                        print(f"    WARNING: Ollama call failed: {exc}")
                        time.sleep(2)

                    attempts += 1

                written = 0
                for rec in collected[:needed]:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    existing_texts.add(rec["text"])
                    written += 1

                total_written += written
                print(f"    wrote {written} records for [{config['label']}][{lang_code}]")

    total_lines = sum(1 for line in open(output_path, encoding="utf-8") if line.strip())
    print(f"\nDone. Total in file: {total_lines} records ({total_written} new)")
    print(f"Output: {output_path}")
    print(f"\nNext: python training/train_ner.py --data {output_path}")


if __name__ == "__main__":
    main()
