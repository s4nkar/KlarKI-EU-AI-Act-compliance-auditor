"""LLM-based chunk classifier. (Phase 2)

Classifies each DocumentChunk into an ArticleDomain using few-shot prompting
against Ollama. Sequential (one request at a time) per Ollama constraint.

Phase 5: Swap backend to BERT via Triton gRPC when USE_TRITON=true.
"""

import structlog
from models.schemas import ArticleDomain, DocumentChunk
from services.ollama_client import OllamaClient

logger = structlog.get_logger()

DOMAIN_LABELS = {d.value for d in ArticleDomain}


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
    raise NotImplementedError("classifier.classify_chunks — implemented in Phase 2")
