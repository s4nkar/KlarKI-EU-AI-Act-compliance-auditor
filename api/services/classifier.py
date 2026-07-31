"""Chunk classifier — selects backend at runtime via USE_TRITON config flag.

  USE_TRITON=false (default): few-shot prompting via Ollama + phi3:mini
  USE_TRITON=true:            batched BERT inference via Triton gRPC

Both backends return the same list[DocumentChunk] with .domain populated,
keeping the rest of the pipeline backend-agnostic.
"""

from typing import Callable

import structlog

from config import settings
from models.schemas import ArticleDomain, DocumentChunk
from services.ollama_client import OllamaClient
from services.prompt_registry import load_prompt

logger = structlog.get_logger()

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


def _parse_label(raw: str) -> ArticleDomain:
    """Normalise a raw label string to ArticleDomain, defaulting to UNRELATED."""
    cleaned = raw.strip().lower().replace("-", "_").replace(" ", "_")
    for char in '",\'.!?':
        cleaned = cleaned.strip(char)
    return _LABEL_MAP.get(cleaned, ArticleDomain.UNRELATED)


async def _classify_ollama(
    chunks: list[DocumentChunk],
    ollama: OllamaClient,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[DocumentChunk], str]:
    """Sequential few-shot classification via Ollama."""
    prompt_template = load_prompt("classifier")
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        prompt = prompt_template.replace("{{CHUNK_TEXT}}", chunk.text)
        try:
            raw = await ollama.generate(prompt)
            chunk.domain = _parse_label(raw)
        except Exception as exc:
            logger.warning("classify_chunk_failed", chunk_id=chunk.chunk_id, error=str(exc))
            chunk.domain = ArticleDomain.UNRELATED

        if on_progress:
            on_progress(i + 1, total)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            logger.info("classify_progress", done=i + 1, total=total)

    return chunks, "ollama"


async def _classify_triton(
    chunks: list[DocumentChunk],
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[DocumentChunk], str]:
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
        done = min(i + batch_size, len(texts))
        if on_progress:
            on_progress(done, len(texts))
        logger.info("classify_progress", done=done, total=len(texts))

    for chunk, label in zip(chunks, all_labels):
        chunk.domain = _LABEL_MAP.get(label, ArticleDomain.UNRELATED)

    return chunks, "triton"


async def classify_chunks(
    chunks: list[DocumentChunk],
    ollama: OllamaClient,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[DocumentChunk], str]:
    """Classify each chunk into an ArticleDomain.

    Delegates to Triton (batched BERT) when USE_TRITON=true,
    otherwise uses Ollama sequential few-shot prompting.

    Args:
        chunks: DocumentChunks with text populated.
        ollama: OllamaClient — used only when USE_TRITON=false.
        on_progress: Optional callback(done, total) invoked as classification
            proceeds, so a caller (e.g. the audit pipeline) can surface live
            progress/ETA without this module knowing anything about audits.

    Returns:
        Tuple of (chunks with .domain set, actual backend name used).
        Backend name reflects any fallback that occurred at runtime.
    """
    if settings.use_triton:
        logger.info("classify_backend", backend="triton", chunks=len(chunks))
        try:
            chunks, backend = await _classify_triton(chunks, on_progress)
        except Exception as exc:
            logger.warning(
                "triton_unavailable_fallback",
                error=str(exc),
                fallback="ollama",
            )
            logger.info("classify_backend", backend="ollama_fallback", chunks=len(chunks))
            chunks, backend = await _classify_ollama(chunks, ollama, on_progress)
            backend = f"ollama_fallback/{settings.ollama_model}"
    else:
        logger.info("classify_backend", backend="ollama", chunks=len(chunks))
        chunks, backend = await _classify_ollama(chunks, ollama, on_progress)

    classified = sum(1 for c in chunks if c.domain != ArticleDomain.UNRELATED)
    logger.info("classify_done", total=len(chunks), classified=classified, backend=backend)
    return chunks, backend
