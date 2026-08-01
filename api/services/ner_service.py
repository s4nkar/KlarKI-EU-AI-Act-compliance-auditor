"""NER enrichment service — runs the trained spaCy model over document chunks.

Adds `ner_entities` to each chunk's metadata dict, grouped by label:
    {"ARTICLE": ["Article 9"], "OBLIGATION": ["shall maintain"], ...}

Two-phase design so NER can run before the legal gate:
  Phase 1 — extract_ner_entities_async   writes chunk.metadata["ner_entities"]
                                         does NOT touch chunk.domain
  Phase 2 — apply_ner_domain_correction  reads already-extracted entities and
                                         corrects UNRELATED chunks that contain
                                         an unambiguous Article 9–15 reference.
                                         Must be called after classify_chunks.

The legacy enrich_chunks_with_ner / enrich_chunks_with_ner_async wrappers
remain for any callers that want both phases in one shot.

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

# Only the NER head and the tok2vec it listens to may run at inference. The
# de_core_news_lg backbone also ships tagger/morphologizer/parser/attribute_ruler
# components; when those run they overwrite doc.ents and ARTICLE, OBLIGATION and
# PROHIBITED_USE entities silently vanish. The training script already disables
# them for its own eval — we must do the same wherever the model is invoked.
_NER_KEEP_PIPES = ("ner", "tok2vec")


def load_ner_model(model_path):
    """Load the spaCy NER model with only the NER-relevant pipes enabled."""
    import spacy
    nlp = spacy.load(str(model_path))
    disable = [p for p in nlp.pipe_names if p not in _NER_KEEP_PIPES]
    if disable:
        nlp.select_pipes(disable=disable)
    return nlp


def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp if _nlp is not False else None
    if not _NER_MODEL_PATH.exists():
        logger.info("ner_model_not_found", path=str(_NER_MODEL_PATH))
        _nlp = False
        return None
    try:
        _nlp = load_ner_model(_NER_MODEL_PATH)
        logger.info("ner_model_loaded", path=str(_NER_MODEL_PATH), active_pipes=_nlp.pipe_names)
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


# ── Phase 1: entity extraction ────────────────────────────────────────────────

def _write_entities(chunk: DocumentChunk, doc) -> None:
    entities: dict[str, list[str]] = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)
    chunk.metadata["ner_entities"] = entities


def extract_ner_entities(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Run spaCy NER over every chunk and write entities to chunk.metadata.

    Writes chunk.metadata["ner_entities"] = {label: [text, ...], ...}.
    Does NOT touch chunk.domain — call apply_ner_domain_correction after
    classify_chunks has set chunk.domain.

    Uses nlp.pipe() to batch-process all chunks in one call instead of one
    nlp() call per chunk — spaCy's own internal batching, a free speedup with
    no change to NER output (same model, same per-doc result). Falls back to
    the original one-call-per-chunk loop (with per-chunk error isolation) if
    the batched call itself raises, so a single malformed chunk can never
    take down NER for the rest of the document — pipe() doesn't support
    resuming mid-batch after an exception, so the fallback re-processes
    everything one at a time, exactly as before this change.

    Returns the same list (mutated in place).
    """
    nlp = _get_nlp()
    if nlp is None:
        return chunks

    texts = [chunk.text[:1000] for chunk in chunks]  # cap at 1000 chars to limit latency

    try:
        docs = nlp.pipe(texts)
        for chunk, doc in zip(chunks, docs):
            _write_entities(chunk, doc)
    except Exception as exc:
        logger.warning("ner_batch_failed_falling_back", error=str(exc))
        for chunk in chunks:
            try:
                doc = nlp(chunk.text[:1000])
            except Exception as chunk_exc:
                logger.warning("ner_chunk_failed", chunk_id=chunk.chunk_id, error=str(chunk_exc))
                continue
            _write_entities(chunk, doc)

    logger.info("ner_extraction_done", total=len(chunks))
    return chunks


async def extract_ner_entities_async(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Async Phase-1 wrapper — Triton NER backend when USE_TRITON=true."""
    if settings.use_triton:
        try:
            return await _extract_triton(chunks)
        except Exception as exc:
            logger.warning("triton_ner_fallback", error=str(exc), fallback="local")

    return await asyncio.to_thread(extract_ner_entities, chunks)


async def _extract_triton(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    from services.triton_client import TritonClient

    client = TritonClient(host=settings.triton_host, grpc_port=settings.triton_grpc_port)
    texts = [c.text[:1000] for c in chunks]

    batch_size = 16  # matches spacy_ner config.pbtxt max_batch_size
    all_entities: list[dict[str, list[str]]] = []
    for i in range(0, len(texts), batch_size):
        batch_entities = await client.ner(texts[i : i + batch_size])
        all_entities.extend(batch_entities)

    for chunk, entities in zip(chunks, all_entities):
        chunk.metadata["ner_entities"] = entities

    logger.info("ner_extraction_done", total=len(chunks), backend="triton")
    return chunks


# ── Phase 2: domain correction ────────────────────────────────────────────────

def apply_ner_domain_correction(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Correct UNRELATED chunks using already-extracted NER entities.

    Reads chunk.metadata["ner_entities"] — does NOT re-run spaCy.
    Must be called after classify_chunks has set chunk.domain.

    Only corrects chunks where NER found exactly one unambiguous Article 9–15
    reference. Multi-article chunks stay UNRELATED to avoid wrong assignment.
    """
    corrected = 0
    for chunk in chunks:
        if chunk.domain != ArticleDomain.UNRELATED:
            continue
        entities = chunk.metadata.get("ner_entities", {})
        article_nums = _extract_article_nums(entities.get("ARTICLE", []))
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
        logger.info("ner_domain_correction_done", corrected=corrected)
    return chunks


# ── Legacy combined wrappers ──────────────────────────────────────────────────

def enrich_chunks_with_ner(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Run both NER phases in sequence (entity extraction + domain correction).

    Requires chunk.domain to be already set by classify_chunks.
    Use extract_ner_entities + apply_ner_domain_correction separately when
    NER needs to run before the legal gate.
    """
    chunks = extract_ner_entities(chunks)
    return apply_ner_domain_correction(chunks)


async def enrich_chunks_with_ner_async(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Async combined wrapper — both phases via Triton or local spaCy."""
    chunks = await extract_ner_entities_async(chunks)
    return apply_ner_domain_correction(chunks)
