"""RAG retrieval engine — embed chunk → ChromaDB search → top-k.

Prefers same-language regulatory passages, falls back to any language.
Searches both eu_ai_act and compliance_checklist collections.
"""

import structlog

from models.schemas import ArticleDomain, DocumentChunk
from services.chroma_client import ChromaClient
from services.embedding_service import EmbeddingService

logger = structlog.get_logger()

# Map domain → article number for metadata filtering
_DOMAIN_TO_ARTICLE: dict[ArticleDomain, int] = {
    ArticleDomain.RISK_MANAGEMENT:          9,
    ArticleDomain.DATA_GOVERNANCE:          10,
    ArticleDomain.TECHNICAL_DOCUMENTATION:  11,
    ArticleDomain.RECORD_KEEPING:           12,
    ArticleDomain.TRANSPARENCY:             13,
    ArticleDomain.HUMAN_OVERSIGHT:          14,
    ArticleDomain.SECURITY:                 15,
}


async def retrieve_requirements(
    chunk: DocumentChunk,
    embedding_service: EmbeddingService,
    chroma_client: ChromaClient,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve relevant regulatory passages for a document chunk.

    Embeds the chunk text, queries ChromaDB (eu_ai_act + compliance_checklist),
    prefers same-language results, falls back to any language.

    Args:
        chunk: Source document chunk to find regulatory matches for.
        embedding_service: Local embedding model instance.
        chroma_client: ChromaDB async client.
        top_k: Maximum number of regulatory passages to return.

    Returns:
        List of dicts with keys: text, metadata, distance.
    """
    vectors = await embedding_service.embed([chunk.text])
    query_vec = vectors[0]

    results: list[dict] = []

    # Query eu_ai_act — filter by article_num if domain is known
    article_num = _DOMAIN_TO_ARTICLE.get(chunk.domain) if chunk.domain else None
    where_filter = {"article_num": article_num} if article_num else None

    try:
        eu_result = await chroma_client.query(
            collection_name="eu_ai_act",
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where_filter,
        )
        results.extend(_flatten_result(eu_result))
    except Exception as exc:
        logger.warning("rag_eu_query_failed", error=str(exc))

    # Query compliance_checklist
    try:
        checklist_result = await chroma_client.query(
            collection_name="compliance_checklist",
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where_filter,
        )
        results.extend(_flatten_result(checklist_result))
    except Exception as exc:
        logger.warning("rag_checklist_query_failed", error=str(exc))

    if not results:
        return []

    # Sort: same-language first, then by distance (ascending = more similar)
    chunk_lang = chunk.language or "en"
    results.sort(key=lambda r: (
        0 if r["metadata"].get("lang") == chunk_lang else 1,
        r["distance"],
    ))

    logger.debug("rag_retrieved", chunk_id=chunk.chunk_id, results=len(results))
    return results[:top_k]


def _flatten_result(chroma_result: dict) -> list[dict]:
    """Convert raw ChromaDB query result into a flat list of passage dicts."""
    passages = []
    ids = chroma_result.get("ids", [[]])[0]
    docs = chroma_result.get("documents", [[]])[0]
    metas = chroma_result.get("metadatas", [[]])[0]
    distances = chroma_result.get("distances", [[]])[0]

    for doc_id, text, meta, dist in zip(ids, docs, metas, distances):
        if text:
            passages.append({
                "id": doc_id,
                "text": text,
                "metadata": meta or {},
                "distance": dist,
            })
    return passages
