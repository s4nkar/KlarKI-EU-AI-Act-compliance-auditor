"""NER enrichment service — runs the trained spaCy model over document chunks.

Adds `ner_entities` to each chunk's metadata dict, grouped by label:
    {"ARTICLE": ["Article 9"], "OBLIGATION": ["shall maintain"], ...}

Also applies a conservative domain correction for chunks classified as UNRELATED
when NER finds an explicit Article reference pointing to a known domain. This
recovers chunks the domain classifier missed (e.g. short procedural paragraphs
that say "as required by Article 9" without any domain-specific vocabulary).

The spaCy model is lazy-loaded once and cached. If the model directory doesn't
exist (not yet trained), all functions return the chunks unchanged.

Model path: training/artifacts/spacy_ner_model/model-final  (relative to project root)
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import structlog

from config import settings
from models.schemas import ArticleDomain, DocumentChunk

logger = structlog.get_logger()

_ROOT = Path(__file__).parent.parent.parent
_NER_MODEL_PATH = _ROOT / "training" / "artifacts" / "spacy_ner_model" / "model-final"

# Loaded once on first call; False = unavailable
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp if _nlp is not False else None
    if not _NER_MODEL_PATH.exists():
        logger.info("ner_model_not_found", path=str(_NER_MODEL_PATH))
        _nlp = False
        return None
    try:
        import spacy
        _nlp = spacy.load(str(_NER_MODEL_PATH))
        logger.info("ner_model_loaded", path=str(_NER_MODEL_PATH))
        return _nlp
    except Exception as exc:
        logger.warning("ner_model_load_failed", error=str(exc))
        _nlp = False
        return None


# Article number → domain (used for domain correction)
_ARTICLE_TO_DOMAIN: dict[int, ArticleDomain] = {
    9:  ArticleDomain.RISK_MANAGEMENT,
    10: ArticleDomain.DATA_GOVERNANCE,
    11: ArticleDomain.TECHNICAL_DOCUMENTATION,
    12: ArticleDomain.RECORD_KEEPING,
    13: ArticleDomain.TRANSPARENCY,
    14: ArticleDomain.HUMAN_OVERSIGHT,
    15: ArticleDomain.SECURITY,
}

_ARTICLE_NUM_RE = re.compile(r"\b(9|10|11|12|13|14|15)\b")


def _extract_article_nums(entity_texts: list[str]) -> list[int]:
    """Pull article numbers 9-15 from a list of ARTICLE entity strings."""
    nums: list[int] = []
    for text in entity_texts:
        for m in _ARTICLE_NUM_RE.finditer(text):
            n = int(m.group())
            if n in _ARTICLE_TO_DOMAIN:
                nums.append(n)
    return nums


def enrich_chunks_with_ner(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Run NER on every chunk and enrich chunk.metadata in-place.

    For each chunk:
      - Adds metadata["ner_entities"] = {label: [text, ...], ...}
      - If chunk.domain is UNRELATED and NER finds an unambiguous Article
        reference (9–15), corrects domain to the matching ArticleDomain.

    Returns the same list (mutated in place) so callers can chain easily.
    If the NER model is unavailable the chunks are returned unchanged.
    """
    nlp = _get_nlp()
    if nlp is None:
        return chunks

    corrected = 0
    for chunk in chunks:
        try:
            doc = nlp(chunk.text[:1000])  # cap at 1000 chars to limit latency
        except Exception as exc:
            logger.warning("ner_chunk_failed", chunk_id=chunk.chunk_id, error=str(exc))
            continue

        # Group entities by label
        entities: dict[str, list[str]] = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)

        chunk.metadata["ner_entities"] = entities

        # Domain correction — only for UNRELATED chunks with a clear article signal
        if chunk.domain == ArticleDomain.UNRELATED:
            article_nums = _extract_article_nums(entities.get("ARTICLE", []))
            # Only correct when there's exactly one article referenced — ambiguous
            # chunks with multiple references stay UNRELATED for safety.
            if len(set(article_nums)) == 1:
                target_domain = _ARTICLE_TO_DOMAIN[article_nums[0]]
                chunk.domain = target_domain
                corrected += 1
                logger.debug(
                    "ner_domain_correction",
                    chunk_id=chunk.chunk_id,
                    article=article_nums[0],
                    new_domain=target_domain.value,
                )

    if corrected:
        logger.info("ner_enrichment_done", total=len(chunks), domain_corrections=corrected)
    else:
        logger.info("ner_enrichment_done", total=len(chunks), domain_corrections=0)

    return chunks


async def enrich_chunks_with_ner_async(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Async wrapper — dispatches to Triton NER when USE_TRITON=true, local spaCy otherwise.

    Triton Python backend runs spaCy on CPU inside the inference server.
    Falls back to the local model if Triton is unreachable.
    """
    if settings.use_triton:
        try:
            return await _enrich_triton(chunks)
        except Exception as exc:
            logger.warning("triton_ner_fallback", error=str(exc), fallback="local")

    return await asyncio.to_thread(enrich_chunks_with_ner, chunks)


async def _enrich_triton(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    from services.triton_client import TritonClient

    client = TritonClient(host=settings.triton_host, grpc_port=settings.triton_grpc_port)
    texts = [c.text[:1000] for c in chunks]

    batch_size = 16  # matches spacy_ner config.pbtxt max_batch_size
    all_entities: list[dict[str, list[str]]] = []
    for i in range(0, len(texts), batch_size):
        batch_entities = await client.ner(texts[i : i + batch_size])
        all_entities.extend(batch_entities)

    corrected = 0
    for chunk, entities in zip(chunks, all_entities):
        chunk.metadata["ner_entities"] = entities
        if chunk.domain == ArticleDomain.UNRELATED:
            article_nums = _extract_article_nums(entities.get("ARTICLE", []))
            if len(set(article_nums)) == 1:
                chunk.domain = _ARTICLE_TO_DOMAIN[article_nums[0]]
                corrected += 1

    logger.info("ner_enrichment_done", total=len(chunks), domain_corrections=corrected, backend="triton")
    return chunks
