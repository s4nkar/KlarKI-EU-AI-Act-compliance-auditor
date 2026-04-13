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
import asyncio
import json
import re
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


def _get_article_domain(path: Path) -> str | None:
    """Read the DOMAIN: header from the first 8 lines of a regulatory txt file."""
    for line in path.read_text(encoding="utf-8").splitlines()[:8]:
        if line.strip().startswith("DOMAIN:"):
            return line.split(":", 1)[1].strip()
    return None


def _split_regulatory_sentences(text: str) -> list[str]:
    """Split a regulatory article section into individual sentences.

    Skips the first non-empty line (article title), strips list markers
    like (a), (b), (i), 1., 2., and discards sentences under 40 chars.
    """
    lines = text.splitlines()
    skip_title = True
    body_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if skip_title:
            skip_title = False
            continue  # skip "Article N — Title" line
        body_lines.append(stripped)

    body = " ".join(body_lines)
    parts = re.split(r'(?<=[.!?])\s+', body)
    results: list[str] = []
    for part in parts:
        # Strip list item prefixes: (a), (b), (i), (ii), 1., 2., etc.
        part = re.sub(r'^[\(\[]?(?:[ivxIVX]+|[a-z0-9]{1,3})[\)\]\.]\s+', '', part).strip()
        if len(part) >= 40:
            results.append(part)
    return results


def extract_regulatory_sentences(regulatory_dir: Path, lang_codes: set[str]) -> list[dict]:
    """Extract labeled sentences directly from data/regulatory/ article files.

    Each file's DOMAIN: header determines the classifier label.
    Files whose domain is not in the 8 trained classes are skipped (e.g. GDPR articles).
    Returns records with source='regulatory'.
    """
    valid_labels = {d["label"] for d in DOMAINS}
    records: list[dict] = []

    for txt_file in sorted(regulatory_dir.glob("**/*.txt")):
        domain = _get_article_domain(txt_file)
        if not domain or domain not in valid_labels:
            continue
        sections = _parse_txt_sections(txt_file)
        for lang_code in sorted(lang_codes):
            text = sections.get(lang_code, "")
            if not text:
                continue
            for sentence in _split_regulatory_sentences(text):
                records.append({
                    "text": sentence,
                    "label": domain,
                    "lang": lang_code,
                    "source": "regulatory",
                })

    return records


async def ollama_generate(
    client: httpx.AsyncClient,
    host: str,
    model: str,
    prompt: str,
    num_predict: int,
) -> str:
    """Async Ollama generate call using a shared client session."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "-1m",
        "options": {"num_predict": num_predict},
    }
    resp = await client.post(f"{host}/api/generate", json=payload)
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


def build_prompt(
    domain: dict,
    language_name: str,
    n: int,
    real_examples: list[str] | None = None,
) -> str:
    """Build a generation prompt anchored to real regulatory sentences.

    Uses actual sentences extracted from data/regulatory/ as few-shot style
    anchors. This is stronger than showing raw article text because the LLM
    sees exactly what format and register correct training examples should have.
    """
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
            f'Return ONLY a JSON object:\n'
            f'{{"sentences": ["...", "..."]}}'
        )

    # Few-shot examples block — real sentences from the regulatory corpus
    examples_block = ""
    if real_examples:
        sample = real_examples[:5]
        lines = "\n".join(f'  {i + 1}. "{ex}"' for i, ex in enumerate(sample))
        examples_block = (
            f"\nReal {language_name} sentences from the regulatory corpus "
            f"(use these as style and register anchors — do NOT copy them verbatim):\n"
            f"{lines}\n"
        )

    not_confused = domain.get("not_confused_with", "")
    disambiguation = (
        f"\nIMPORTANT — do NOT generate sentences that could belong to other classes:\n"
        f"{not_confused}\n"
    ) if not_confused else ""

    return (
        f'You are generating training data for a compliance classification model.\n'
        f'\n'
        f'Label: {domain["label"].upper()}\n'
        f'Regulation: EU AI Act {domain["article"]}\n'
        f'Description: {domain["description"]}\n'
        f'{examples_block}'
        f'{disambiguation}\n'
        f'Generate {n} NEW sentences (10-30 words) that could appear in an AI system\'s '
        f'compliance documentation, internal policy, or audit report. {lang_instruction}\n'
        f'\n'
        f'Requirements:\n'
        f'- Each sentence must UNAMBIGUOUSLY belong to {domain["label"].upper()} — a human expert must agree\n'
        f'- Match the register of the example sentences above (formal, precise, regulatory)\n'
        f'- Avoid repeating phrases or synonymous rewrites\n'
        f'- Vary sentence structure (statements, imperatives, passive constructions)\n'
        f'- Keep sentences independent — each must stand alone\n'
        f'- Do NOT copy or closely paraphrase the example sentences\n'
        f'\n'
        f'Return ONLY a JSON object:\n'
        f'{{"sentences": ["...", "..."]}}'
    )


async def _generate_domain_lang(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    domain: dict,
    lang_code: str,
    lang_name: str,
    needed: int,
    batch_size: int,
    host: str,
    model: str,
    existing_snapshot: frozenset,
    num_predict: int,
    real_examples: list[str] | None = None,
) -> tuple[str, str, list[str]]:
    """Generate `needed` synthetic sentences for one (domain, language) pair.

    The semaphore is acquired PER REQUEST (not per task) so concurrent tasks
    interleave correctly and don't all queue up against Ollama at once.
    Returns (label, lang_code, texts).
    """
    generated: list[str] = []
    seen: set[str] = set(existing_snapshot)
    # More headroom: at least 6× the minimum batches needed
    max_attempts = max((needed // batch_size + 3) * 6, 20)
    attempts = 0

    print(f"  [{domain['label']}][{lang_code}] task started — need {needed}, "
          f"max_attempts={max_attempts}")

    while len(generated) < needed and attempts < max_attempts:
        batch_n = min(batch_size, needed - len(generated) + 5)
        prompt = build_prompt(domain, lang_name, batch_n, real_examples)
        try:
            # Acquire semaphore only for the duration of the HTTP request
            async with semaphore:
                raw = await ollama_generate(client, host, model, prompt, num_predict)
            batch = parse_json_list(raw)
            new_count = 0
            for text in batch:
                text = text.strip()
                if len(text) >= 20 and text not in seen:
                    generated.append(text)
                    seen.add(text)
                    new_count += 1
            print(f"  [{domain['label']}][{lang_code}] batch {attempts + 1}: "
                  f"got {len(batch)}, kept {new_count} (total {len(generated)}/{needed})")
        except Exception as exc:
            print(f"  [{domain['label']}][{lang_code}] WARNING batch {attempts + 1}: {exc}")
            await asyncio.sleep(2)
        attempts += 1

    if len(generated) < needed:
        print(f"  [{domain['label']}][{lang_code}] reached max_attempts ({max_attempts}) "
              f"with {len(generated)}/{needed} collected")

    return domain["label"], lang_code, generated[:needed]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-generate BERT training data via Ollama (async)")
    parser.add_argument("--n-per-class", type=int, default=400,
                        help="Examples to generate per class per language (default: 400)")
    parser.add_argument("--output", default="training/data/clause_labels.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="phi3:mini")
    parser.add_argument("--languages", default="en,de",
                        help="Comma-separated language codes (default: en,de)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Sentences per Ollama call (default: 10; raise to 20-30 on GPU)")
    parser.add_argument("--semaphore", type=int, default=2,
                        help="Max concurrent Ollama requests (default: 2; raise to 4-6 on GPU)")
    parser.add_argument("--num-predict", type=int, default=2048,
                        help="Max tokens per Ollama response (default: 2048)")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    active_langs = [(c, n) for c, n in LANGUAGES if c in args.languages.split(",")]

    # Load existing examples
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

    # Verify Ollama is reachable and the model can actually generate
    print(f"\nChecking Ollama at {args.ollama_host}...")
    try:
        _probe_timeout = httpx.Timeout(
            connect=10.0,
            read=120.0,
            write=30.0,
            pool=10.0
        )
        async with httpx.AsyncClient(timeout=_probe_timeout) as probe:
            # 1. Check server is up and model is listed
            tags_resp = await probe.get(f"{args.ollama_host}/api/tags")
            tags_resp.raise_for_status()
            models = [m["name"] for m in tags_resp.json().get("models", [])]
            if not any(args.ollama_model.split(":")[0] in m for m in models):
                print(f"  WARNING: model '{args.ollama_model}' not found in Ollama.")
                print(f"  Available: {models or '(none)'}")
                print(f"  Pull it first: docker exec klarki-ollama ollama pull {args.ollama_model}")
                raise SystemExit(1)
            print(f"  Model '{args.ollama_model}' found.")

            # 2. Fire a tiny real generate request to confirm the model can respond
            print(f"  Sending test generation request (may take 30–120 s on CPU)...")
            test_payload = {
                "model": args.ollama_model,
                "prompt": 'Return a JSON object: {"ok": true}',
                "stream": False,
                "format": "json",
                "options": {"num_predict": 20},
            }
            gen_resp = await probe.post(f"{args.ollama_host}/api/generate", json=test_payload)
            gen_resp.raise_for_status()
            print(f"  Ollama generate: OK (response: {gen_resp.json().get('response','')[:60]!r})")
    except SystemExit:
        raise
    except Exception as exc:
        print(f"  ERROR: {exc}")
        print("  Start containers first: docker compose up -d")
        raise SystemExit(1)

    # Report article reference files
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

    # Inject real regulatory sentences and build per-(domain, lang) example map
    print("\nExtracting real regulatory sentences...")
    active_lang_codes = {c for c, _ in active_langs}
    reg_records = extract_regulatory_sentences(REGULATORY_DIR, active_lang_codes)

    # Build few-shot anchor map: {(label, lang_code): [sentence, ...]}
    real_examples_map: dict[tuple[str, str], list[str]] = {}
    for rec in reg_records:
        key = (rec["label"], rec["lang"])
        real_examples_map.setdefault(key, []).append(rec["text"])

    reg_written = 0
    with open(output_path, "a", encoding="utf-8") as out_f:
        for rec in reg_records:
            if rec["text"] not in existing_texts:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                existing_texts.add(rec["text"])
                reg_written += 1
    print(f"  Wrote {reg_written} regulatory sentences "
          f"({len(reg_records) - reg_written} already existed)")
    for (lbl, lc), examples in sorted(real_examples_map.items()):
        print(f"  [{lbl}][{lc}] {len(examples)} real sentences available as few-shot anchors")

    # Snapshot before concurrent tasks — each task gets an immutable view
    # so they never race on the shared set during generation
    existing_snapshot = frozenset(existing_texts)

    # Launch all (domain × language) tasks concurrently under the semaphore
    n_tasks = len(DOMAINS) * len(active_langs)
    print(f"\nGenerating {n_tasks} domain×language combinations "
          f"(semaphore={args.semaphore}, batch={args.batch_size}, "
          f"num_predict={args.num_predict})...")

    # 600s read timeout — phi3:mini on CPU can take 3–8 min per batch
    _client_timeout = httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=30.0)
    semaphore = asyncio.Semaphore(args.semaphore)

    # Write incrementally as each task completes — progress is saved even if
    # the script is interrupted before all tasks finish
    total_written = 0
    async with httpx.AsyncClient(timeout=_client_timeout) as client:
        pending_tasks = [asyncio.ensure_future(
            _generate_domain_lang(
                semaphore, client, domain, lang_code, lang_name,
                args.n_per_class, args.batch_size,
                args.ollama_host, args.ollama_model,
                existing_snapshot, args.num_predict,
                real_examples=real_examples_map.get((domain["label"], lang_code)),
            )
        ) for domain in DOMAINS for lang_code, lang_name in active_langs]

        with open(output_path, "a", encoding="utf-8") as out_f:
            for coro in asyncio.as_completed(pending_tasks):
                try:
                    label, lang_code, texts = await coro
                except Exception as exc:
                    print(f"  ERROR: task failed — {exc}")
                    continue
                written = 0
                for text in texts:
                    if text not in existing_texts:
                        record = {"text": text, "label": label, "lang": lang_code, "source": "generated"}
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        existing_texts.add(text)
                        written += 1
                out_f.flush()  # ensure OS writes to disk after each task
                total_written += written
                print(f"  wrote {written} examples for [{label}][{lang_code}] "
                      f"(running total: {total_written})")

    print(f"\nDone.")
    print(f"  Written: {total_written} new examples")
    print(f"  Output:  {output_path}")
    total_lines = sum(1 for line in open(output_path, encoding="utf-8") if line.strip())
    print(f"  Total in file: {total_lines} examples")
    print(f"\nNext: python training/train_classifier.py --data {output_path}")


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
