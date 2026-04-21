#!/usr/bin/env python3
"""Generate training data for the three Phase 3 specialist classifiers.

Generates synthetic examples grounded in the actual EU AI Act regulatory text
for each classifier type:

  actor      — 4-class: provider / deployer / importer / distributor (Article 3)
  risk       — binary:  high_risk / not_high_risk (Article 6 + Annex III)
  prohibited — binary:  prohibited / not_prohibited (Article 5)

Uses Ollama async (same infrastructure as generate_bert_training_data.py).
Output is APPENDED to the existing seed JSONL files — seed examples are preserved.
Deduplicates by text content after gathering.

Usage:
    # Generate all three (default: 200 per class per language)
    python scripts/generate_specialist_training_data.py

    # Single classifier, smoke test
    python scripts/generate_specialist_training_data.py --type actor --n-per-class 20

    # Overwrite (clear seed data + regenerate)
    python scripts/generate_specialist_training_data.py --overwrite

    # Custom Ollama host
    python scripts/generate_specialist_training_data.py --ollama-host http://localhost:11434
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
REGULATORY_DIR = ROOT / "data" / "regulatory"
OUTPUT_DIR = ROOT / "training" / "data"


# ── Classifier configs ────────────────────────────────────────────────────────

CLASSIFIER_CONFIGS: dict[str, dict] = {
    "actor": {
        "output_file": "actor_labels.jsonl",
        "classes": [
            {
                "label": "provider",
                "article": "Article 3(3)",
                "description": (
                    "A natural or legal person who DEVELOPS an AI system and places it on the market "
                    "or puts it into service under their OWN NAME or trademark. "
                    "Key signals: 'we developed', 'our model', 'we trained', 'placed on market', "
                    "'our AI product', 'we built', 'proprietary AI', 'our trademark'. "
                    "NOT a deployer (who merely uses someone else's system)."
                ),
            },
            {
                "label": "deployer",
                "article": "Article 3(4)",
                "description": (
                    "A natural or legal person who USES an AI system developed by someone else, "
                    "under their own authority. Key signals: 'we use a third-party AI', "
                    "'we deployed a vendor system', 'we implemented an external model', "
                    "'we operate the AI under our authority', 'licensed AI', 'off-the-shelf AI'. "
                    "Most organisations are deployers, not providers."
                ),
            },
            {
                "label": "importer",
                "article": "Article 3(6)",
                "description": (
                    "A person ESTABLISHED IN THE EU who places on the market an AI system "
                    "bearing the name/trademark of a person ESTABLISHED OUTSIDE the EU. "
                    "Key signals: 'we import', 'non-EU manufacturer', 'developed outside the EU', "
                    "'bring to EU market', 'established in the Union', 'EU importer'."
                ),
            },
            {
                "label": "distributor",
                "article": "Article 3(7)",
                "description": (
                    "A person in the supply chain who makes the AI system AVAILABLE ON THE MARKET "
                    "WITHOUT MODIFYING ITS PROPERTIES, and who is NEITHER the provider NOR the importer. "
                    "Key signals: 'we distribute', 'we resell', 'we make available', 'without modification', "
                    "'distribution partner', 'retail network for AI'."
                ),
            },
        ],
        "prompt_template": (
            "Generate {n} realistic sentences that clearly identify an organisation in the "
            "EU AI Act actor role of '{label}' ({article}).\n\n"
            "Role definition: {description}\n\n"
            "Requirements:\n"
            "- Each sentence must unambiguously indicate the '{label}' role\n"
            "- Mix formal policy language with operational descriptions\n"
            "- Include real-world contexts: manufacturing, healthcare, finance, HR, public sector\n"
            "- Vary sentence structure and vocabulary\n"
            "- {lang_instruction}\n\n"
            "Return ONLY a JSON array of strings. No explanations.\n"
            'Example format: ["Sentence 1.", "Sentence 2.", ...]'
        ),
    },
    "risk": {
        "output_file": "risk_labels.jsonl",
        "classes": [
            {
                "label": "high_risk",
                "article": "Article 6 + Annex III",
                "description": (
                    "An AI system that falls under one or more Annex III categories: "
                    "(1) biometric identification/categorisation; "
                    "(2) critical infrastructure management; "
                    "(3) education access/assessment; "
                    "(4) employment decisions (hiring, firing, performance); "
                    "(5) essential services (credit, insurance, benefits); "
                    "(6) law enforcement (crime prediction, profiling); "
                    "(7) migration/asylum/border control; "
                    "(8) justice/democratic processes. "
                    "OR a safety component of an Annex I product."
                ),
            },
            {
                "label": "not_high_risk",
                "article": "Article 6 — does NOT apply",
                "description": (
                    "An AI system that does NOT fall under any Annex III category and is NOT a safety "
                    "component of an Annex I product. Examples: chatbots, spam filters, recommendation engines, "
                    "predictive maintenance tools, content generation AI, translation tools, "
                    "inventory optimisation AI, customer service bots. "
                    "These may have transparency obligations (Article 52) but not full Article 9-15 requirements."
                ),
            },
        ],
        "prompt_template": (
            "Generate {n} realistic sentences describing an AI system that is '{label}' "
            "under the EU AI Act ({article}).\n\n"
            "Classification criteria: {description}\n\n"
            "Requirements:\n"
            "- Each sentence must describe a concrete AI use case or system\n"
            "- Be specific about the domain and function of the AI\n"
            "- Include realistic operational details (not just abstract descriptions)\n"
            "- {lang_instruction}\n\n"
            "Return ONLY a JSON array of strings. No explanations.\n"
            'Example format: ["Sentence 1.", "Sentence 2.", ...]'
        ),
    },
    "prohibited": {
        "output_file": "prohibited_labels.jsonl",
        "classes": [
            {
                "label": "prohibited",
                "article": "Article 5",
                "description": (
                    "An AI practice that is ABSOLUTELY PROHIBITED under Article 5: "
                    "(a) subliminal manipulation techniques beyond conscious awareness causing harm; "
                    "(b) exploiting vulnerabilities of specific groups (age, disability, economic situation); "
                    "(c) social scoring by public authorities leading to detrimental treatment; "
                    "(d) real-time remote biometric identification in public spaces for law enforcement; "
                    "(e) emotion recognition in workplaces or educational institutions (except medical/safety). "
                    "These practices are BANNED regardless of risk tier — even high-risk systems cannot do these."
                ),
            },
            {
                "label": "not_prohibited",
                "article": "Article 5 — does NOT apply",
                "description": (
                    "An AI practice that is NOT prohibited under Article 5. "
                    "Examples: biometric authentication for access control (not identification in public); "
                    "sentiment analysis for customer feedback; fraud detection; "
                    "emotion recognition for medical diagnosis with informed consent; "
                    "biometric recognition for border control with legal basis; "
                    "credit scoring (may be high-risk but not prohibited); "
                    "personalised recommendations without manipulation; "
                    "AI content moderation; AI safety monitoring in factories."
                ),
            },
        ],
        "prompt_template": (
            "Generate {n} realistic sentences describing an AI practice that is '{label}' "
            "under Article 5 of the EU AI Act ({article}).\n\n"
            "Classification criteria: {description}\n\n"
            "Requirements:\n"
            "- Each sentence must describe a specific, concrete AI use case\n"
            "- For 'prohibited': make the prohibited element clear but realistic (not cartoonishly evil)\n"
            "- For 'not_prohibited': describe legitimate AI uses that might superficially resemble prohibited ones\n"
            "- {lang_instruction}\n\n"
            "Return ONLY a JSON array of strings. No explanations.\n"
            'Example format: ["Sentence 1.", "Sentence 2.", ...]'
        ),
    },
}


# ── Ollama async generation ───────────────────────────────────────────────────

def _repair_json_array(raw: str) -> list[str]:
    """Extract and repair a JSON string array from a potentially truncated LLM response."""
    # Find the outermost [...] using a greedy match
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        # Try to salvage individual quoted strings even without a wrapper
        items = re.findall(r'"((?:[^"\\]|\\.)*)"\s*[,\]]?', raw)
        return [t.strip() for t in items if t.strip()]

    text = match.group()

    # Attempt direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [s for s in result if isinstance(s, str)]
    except json.JSONDecodeError:
        pass

    # Repair: remove trailing comma before ] and re-try
    repaired = re.sub(r",\s*\]", "]", text)
    try:
        result = json.loads(repaired)
        if isinstance(result, list):
            return [s for s in result if isinstance(s, str)]
    except json.JSONDecodeError:
        pass

    # Last resort: extract all quoted strings from whatever we have
    items = re.findall(r'"((?:[^"\\]|\\.)*)"', text)
    return [t.strip() for t in items if t.strip()]


async def _call_ollama(
    client: httpx.AsyncClient,
    ollama_host: str,
    model: str,
    prompt: str,
    num_predict: int,
) -> list[str]:
    """Call Ollama and parse the JSON array of strings from the response."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "seed": 42,
            "num_predict": num_predict,
        },
        "keep_alive": "-1m",
    }
    try:
        resp = await client.post(f"{ollama_host}/api/generate", json=payload, timeout=120.0)
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        return _repair_json_array(raw)
    except Exception as exc:
        print(f"    [warn] Ollama call failed: {exc}")
        return []


async def _generate_class(
    client: httpx.AsyncClient,
    ollama_host: str,
    model: str,
    classifier_type: str,
    cls: dict,
    lang: str,
    n_per_batch: int,
    n_batches: int,
    existing_texts: frozenset[str],
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Generate training examples for one class in one language."""
    config = CLASSIFIER_CONFIGS[classifier_type]
    lang_instruction = (
        "Write in ENGLISH only."
        if lang == "en"
        else "Write in GERMAN only (formal regulatory style). Use German grammar and vocabulary."
    )

    prompt = config["prompt_template"].format(
        n=n_per_batch,
        label=cls["label"],
        article=cls["article"],
        description=cls["description"],
        lang_instruction=lang_instruction,
    )

    results: list[dict] = []
    for batch in range(n_batches):
        async with semaphore:
            texts = await _call_ollama(client, ollama_host, model, prompt, num_predict=2048)

        new_count = 0
        for text in texts:
            text = text.strip()
            if not text or text in existing_texts:
                continue
            results.append({
                "text": text,
                "label": cls["label"],
                "lang": lang,
                "source": "generated",
            })
            new_count += 1

        print(f"      batch {batch + 1}/{n_batches}: +{new_count} new examples")

    return results


async def generate_classifier(
    classifier_type: str,
    n_per_class: int,
    batch_size: int,
    languages: list[str],
    ollama_host: str,
    model: str,
    semaphore_limit: int,
    overwrite: bool,
) -> None:
    """Generate training data for one classifier type."""
    config = CLASSIFIER_CONFIGS[classifier_type]
    output_path = OUTPUT_DIR / config["output_file"]

    print(f"\n{'='*60}")
    print(f"  Classifier: {classifier_type.upper()}")
    print(f"  Output: {output_path}")
    print(f"  Target: {n_per_class} examples × {len(config['classes'])} classes × {len(languages)} langs")
    print(f"{'='*60}")

    # Load existing examples
    existing: list[dict] = []
    if output_path.exists() and not overwrite:
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        print(f"  Loaded {len(existing)} existing examples (append mode)")
    elif overwrite and output_path.exists():
        print("  --overwrite: clearing existing file")

    existing_texts: frozenset[str] = frozenset(r["text"] for r in existing)

    n_batches = max(1, n_per_class // batch_size)
    semaphore = asyncio.Semaphore(semaphore_limit)

    async with httpx.AsyncClient() as client:
        tasks = []
        for cls in config["classes"]:
            for lang in languages:
                print(f"\n  Generating [{cls['label']}] [{lang}]:")
                tasks.append(
                    _generate_class(
                        client, ollama_host, model,
                        classifier_type, cls, lang,
                        batch_size, n_batches,
                        existing_texts, semaphore,
                    )
                )

        all_batches = await asyncio.gather(*tasks)

    new_examples: list[dict] = []
    for batch in all_batches:
        new_examples.extend(batch)

    # Final deduplication (parallel tasks may overlap)
    seen = set(existing_texts)
    deduped: list[dict] = []
    for ex in new_examples:
        if ex["text"] not in seen:
            seen.add(ex["text"])
            deduped.append(ex)

    all_examples = (existing if not overwrite else []) + deduped

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n  Done: {len(deduped)} new examples added → {len(all_examples)} total in {output_path}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Phase 3 specialist classifier training data via Ollama"
    )
    parser.add_argument(
        "--type", choices=["actor", "risk", "prohibited", "all"], default="all",
        help="Which classifier to generate data for (default: all)",
    )
    parser.add_argument("--n-per-class", type=int, default=200,
                        help="Target examples per class per language (default: 200)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Sentences per Ollama call (default: 5; phi3:mini reliable at ≤5)")
    parser.add_argument("--languages", default="en,de",
                        help="Comma-separated language codes (default: en,de)")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--model", default="phi3:mini")
    parser.add_argument("--semaphore", type=int, default=4,
                        help="Max concurrent Ollama requests (default: 4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Clear output file before generating (default: append)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    types = list(CLASSIFIER_CONFIGS.keys()) if args.type == "all" else [args.type]
    languages = [l.strip() for l in args.languages.split(",")]

    for classifier_type in types:
        await generate_classifier(
            classifier_type=classifier_type,
            n_per_class=args.n_per_class,
            batch_size=args.batch_size,
            languages=languages,
            ollama_host=args.ollama_host,
            model=args.model,
            semaphore_limit=args.semaphore,
            overwrite=args.overwrite,
        )

    print("\nAll done. Next step:")
    print("  python training/train_specialist_classifiers.py --type actor")
    print("  python training/train_specialist_classifiers.py --type risk")
    print("  python training/train_specialist_classifiers.py --type prohibited")


if __name__ == "__main__":
    asyncio.run(main())
