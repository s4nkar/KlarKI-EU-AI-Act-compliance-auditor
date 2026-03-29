"""Chunk classifier — Ollama (Phase 2) or Triton BERT (Phase 5).

Backend is selected at runtime via the USE_TRITON config flag:
  - USE_TRITON=false (default): few-shot prompting via Ollama + phi3:mini
  - USE_TRITON=true:            batched BERT inference via Triton gRPC

Both paths return the same list[DocumentChunk] with .domain populated,
so the rest of the pipeline is unaffected by which backend is active.
"""

from pathlib import Path

import structlog

from config import settings
from models.schemas import ArticleDomain, DocumentChunk
from services.ollama_client import OllamaClient

logger = structlog.get_logger()

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "classify_chunk.txt"

# String label → ArticleDomain enum (shared by both backends)
_LABEL_MAP: dict[str, ArticleDomain] = {
    "risk_management":         ArticleDomain.RISK_MANAGEMENT,
    "data_governance":         ArticleDomain.DATA_GOVERNANCE,
    "technical_documentation": ArticleDomain.TECHNICAL_DOCUMENTATION,
    "record_keeping":          ArticleDomain.RECORD_KEEPING,
    "transparency":            ArticleDomain.TRANSPARENCY,
    "human_oversight":         ArticleDomain.HUMAN_OVERSIGHT,
    "security":                ArticleDomain.SECURITY,
    "unrelated":               ArticleDomain.UNRELATED,
}


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _parse_label(raw: str) -> ArticleDomain:
    """Normalise a raw label string to ArticleDomain, defaulting to UNRELATED."""
    cleaned = raw.strip().lower().replace("-", "_").replace(" ", "_")
    for char in '",\'.!?':
        cleaned = cleaned.strip(char)
    return _LABEL_MAP.get(cleaned, ArticleDomain.UNRELATED)


# ── Ollama backend (Phase 2) ──────────────────────────────────────────────────

async def _classify_ollama(chunks: list[DocumentChunk], ollama: OllamaClient) -> list[DocumentChunk]:
    """Sequential few-shot classification via Ollama."""
    prompt_template = _load_prompt()
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        prompt = prompt_template.replace("{chunk_text}", chunk.text)
        try:
            raw = await ollama.generate(prompt)
            chunk.domain = _parse_label(raw)
        except Exception as exc:
            logger.warning("classify_chunk_failed", chunk_id=chunk.chunk_id, error=str(exc))
            chunk.domain = ArticleDomain.UNRELATED

        if (i + 1) % 10 == 0 or (i + 1) == total:
            logger.info("classify_progress", done=i + 1, total=total)

    return chunks


# ── Triton backend (Phase 5) ──────────────────────────────────────────────────

async def _classify_triton(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Batched BERT classification via Triton gRPC."""
    from services.triton_client import TritonClient

    client = TritonClient(
        host=settings.triton_host,
        grpc_port=settings.triton_grpc_port,
    )

    texts = [c.text for c in chunks]

    # Triton accepts up to 32 at a time (per config.pbtxt max_batch_size)
    batch_size = 32
    all_labels: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        labels = await client.classify(batch)
        all_labels.extend(labels)
        logger.info("classify_progress", done=min(i + batch_size, len(texts)), total=len(texts))

    for chunk, label in zip(chunks, all_labels):
        chunk.domain = _LABEL_MAP.get(label, ArticleDomain.UNRELATED)

    return chunks


# ── Public interface ──────────────────────────────────────────────────────────

async def classify_chunks(
    chunks: list[DocumentChunk],
    ollama: OllamaClient,
) -> list[DocumentChunk]:
    """Classify each chunk into an ArticleDomain.

    Delegates to Triton (batched BERT) when USE_TRITON=true,
    otherwise uses Ollama sequential few-shot prompting.

    Args:
        chunks: DocumentChunks with text populated.
        ollama: OllamaClient — used only when USE_TRITON=false.

    Returns:
        Same list with .domain set on every chunk.
    """
    if settings.use_triton:
        logger.info("classify_backend", backend="triton", chunks=len(chunks))
        chunks = await _classify_triton(chunks)
    else:
        logger.info("classify_backend", backend="ollama", chunks=len(chunks))
        chunks = await _classify_ollama(chunks, ollama)

    classified = sum(1 for c in chunks if c.domain != ArticleDomain.UNRELATED)
    logger.info("classify_done", total=len(chunks), classified=classified)
    return chunks
