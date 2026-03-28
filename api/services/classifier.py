"""LLM-based chunk classifier.

Classifies each DocumentChunk into an ArticleDomain using few-shot prompting
against Ollama. Sequential (one request at a time) per Ollama constraint.

Phase 5: Swap backend to BERT via Triton gRPC when USE_TRITON=true.
"""

from pathlib import Path

import structlog

from models.schemas import ArticleDomain, DocumentChunk
from services.ollama_client import OllamaClient

logger = structlog.get_logger()

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "classify_chunk.txt"

# Map raw LLM output string → ArticleDomain enum
_LABEL_MAP: dict[str, ArticleDomain] = {
    "risk_management":        ArticleDomain.RISK_MANAGEMENT,
    "data_governance":        ArticleDomain.DATA_GOVERNANCE,
    "technical_documentation": ArticleDomain.TECHNICAL_DOCUMENTATION,
    "record_keeping":         ArticleDomain.RECORD_KEEPING,
    "transparency":           ArticleDomain.TRANSPARENCY,
    "human_oversight":        ArticleDomain.HUMAN_OVERSIGHT,
    "security":               ArticleDomain.SECURITY,
    "unrelated":              ArticleDomain.UNRELATED,
}


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _parse_label(raw: str) -> ArticleDomain:
    """Normalise LLM output to an ArticleDomain, defaulting to UNRELATED."""
    cleaned = raw.strip().lower().replace("-", "_").replace(" ", "_")
    # Strip surrounding punctuation/quotes
    for char in '",\'.!?':
        cleaned = cleaned.strip(char)
    return _LABEL_MAP.get(cleaned, ArticleDomain.UNRELATED)


async def classify_chunks(
    chunks: list[DocumentChunk],
    ollama: OllamaClient,
) -> list[DocumentChunk]:
    """Classify each chunk into an ArticleDomain using LLM few-shot prompting.

    Reads the prompt template from prompts/classify_chunk.txt.
    Updates chunk.domain in-place and returns the updated list.
    Runs sequentially to respect the single-request Ollama constraint.

    Args:
        chunks: List of DocumentChunk objects with text populated.
        ollama: OllamaClient instance for LLM inference.

    Returns:
        Same list with .domain set on each chunk.
    """
    prompt_template = _load_prompt()
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        prompt = prompt_template.replace("{chunk_text}", chunk.text)
        try:
            raw = await ollama.generate(prompt)
            chunk.domain = _parse_label(raw)
        except Exception as exc:
            logger.warning(
                "classify_chunk_failed",
                chunk_id=chunk.chunk_id,
                error=str(exc),
            )
            chunk.domain = ArticleDomain.UNRELATED

        if (i + 1) % 10 == 0 or (i + 1) == total:
            logger.info("classify_progress", done=i + 1, total=total)

    classified = sum(1 for c in chunks if c.domain != ArticleDomain.UNRELATED)
    logger.info("classify_done", total=total, classified=classified)
    return chunks
