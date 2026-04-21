"""Script to generate a structured JSON Knowledge Graph of obligations from raw text."""

import json
import os
import re
import sys
import asyncio
from pathlib import Path

import httpx


class _Logger:
    def info(self, event, **kw):    print(f"  [info]  {event} {kw or ''}")
    def warning(self, event, **kw): print(f"  [warn]  {event} {kw or ''}")
    def error(self, event, **kw):   print(f"  [error] {event} {kw or ''}", file=sys.stderr)

logger = _Logger()


async def _ollama_health(host: str, model: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{host}/api/tags")
            tags = r.json().get("models", [])
            return any(m.get("name", "").startswith(model) for m in tags)
    except Exception:
        return False


async def _ollama_generate_json(host: str, model: str, prompt: str) -> dict:
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_predict": 2048},
        "keep_alive": "-1m",
    }
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(f"{host}/api/generate", json=payload)
        r.raise_for_status()
        raw = r.json().get("response", "")
    # Extract outermost {...}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}

OLLAMA_HOST = "http://localhost:11434"
# Using phi3:mini for speed, but can be changed via env
MODEL = os.getenv("MODEL", "phi3:mini")

DATA_DIR = Path("data/regulatory")
OUT_DIR = Path("data/obligations")

PROMPT_TEMPLATE = """You are an expert Legal Knowledge Engineer.
Extract discrete, actionable compliance obligations from the following legal text.

Text:
{text}

Instructions:
1. Break down the article into separate obligations if there are multiple independent requirements.
2. Output ONLY a JSON object containing an "obligations" key with an array of objects matching this schema:
{{
  "id": "String (e.g., AIACT_ART9_001, GDPR_ART5_001)",
  "regulation": "{regulation}",
  "article": "{article_name}",
  "title": "String (short description)",
  "actor": ["String (e.g., provider, deployer, controller, processor)"],
  "trigger": ["String (e.g., high-risk AI system, processing personal data)"],
  "requirement_type": "String (mandatory or conditional)",
  "requirement": "String (The strict legal requirement)",
  "evidence_required": ["String (from controlled taxonomy below)"],
  "severity": "String (low, medium, high, critical)",
  "linked_articles": ["String (e.g., Article 10, Annex IV)"],
  "penalty_relevance": true
}}

Evidence Taxonomy (choose only from these if possible, or create new ones if strictly necessary):
risk register, mitigation controls, monitoring procedure, technical documentation, conformity assessment, CE marking declaration, EU database registration, fundamental rights impact assessment, bias audit, data governance documentation, human oversight procedure, transparency notice, logging and audit trail, appeals mechanism, explainability mechanism, safety validation records, incident monitoring procedure, supervisory authority notification, data protection impact assessment.

Output exactly valid JSON.
"""

def extract_en_text(filepath: Path) -> tuple[str, str, str]:
    """Extracts REGULATION, ARTICLE, and === EN === block from the .txt file."""
    content = filepath.read_text(encoding="utf-8")
    
    reg_match = re.search(r"^REGULATION:\s*(.+)$", content, re.MULTILINE)
    art_match = re.search(r"^ARTICLE:\s*(.+)$", content, re.MULTILINE)
    
    regulation = reg_match.group(1).strip() if reg_match else "Unknown"
    article = f"Article {art_match.group(1).strip()}" if art_match else "Unknown Article"
    
    # Extract text under === EN === until === DE === or EOF
    en_block = ""
    en_start = content.find("=== EN ===")
    if en_start != -1:
        de_start = content.find("=== DE ===", en_start)
        if de_start != -1:
            en_block = content[en_start + 10:de_start].strip()
        else:
            en_block = content[en_start + 10:].strip()
            
    return regulation, article, en_block

async def process_file(filepath: Path, host: str, model: str, out_path: Path) -> None:
    regulation, article, text = extract_en_text(filepath)
    if not text:
        logger.warning("no_en_text", file=filepath.name)
        return

    logger.info("processing_article", article=article, regulation=regulation)
    text_chunk = text[:4000]
    prompt = PROMPT_TEMPLATE.format(text=text_chunk, regulation=regulation, article_name=article)

    try:
        result = await _ollama_generate_json(host, model, prompt)
        obligations = result.get("obligations", [])

        if not obligations:
            logger.warning("no_obligations_extracted", file=filepath.name)
            return

        with open(out_path, "a", encoding="utf-8") as f:
            for obs in obligations:
                obs.setdefault("id", f"{regulation.upper()}_{article.upper().replace(' ', '')}_001")
                obs["regulation"] = regulation
                obs["article"] = article
                f.write(json.dumps(obs) + "\n")

        logger.info("saved_obligations", file=filepath.name, count=len(obligations))
    except Exception as e:
        logger.error("extraction_failed", file=filepath.name, error=str(e))


async def main() -> None:
    logger.info("starting_knowledge_graph_generation")

    if not await _ollama_health(OLLAMA_HOST, MODEL):
        logger.error("ollama_unreachable", host=OLLAMA_HOST, model=MODEL)
        sys.exit(1)
        
    # Clear out old generated files to avoid duplication
    for reg in ["eu_ai_act", "gdpr"]:
        out_file = OUT_DIR / reg / "generated_obligations.jsonl"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if out_file.exists():
            out_file.unlink()
            
    for reg_dir in DATA_DIR.iterdir():
        if not reg_dir.is_dir():
            continue
            
        out_file = OUT_DIR / reg_dir.name / "generated_obligations.jsonl"
        
        for file in reg_dir.glob("*.txt"):
            await process_file(file, OLLAMA_HOST, MODEL, out_file)
            
    logger.info("knowledge_graph_generation_complete")

if __name__ == "__main__":
    asyncio.run(main())
