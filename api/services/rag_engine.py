"""Hybrid RAG retrieval engine — BM25 + vector search + RRF + cross-encoder re-ranking.

Pipeline per query:
  1. BM25 keyword search  (top 10, article-filtered)
  2. Vector semantic search (top 10, article-filtered)
  3. Reciprocal Rank Fusion → candidate pool (~15 unique docs)
  4. Cross-encoder re-ranking → final top-k

Both BM25 and vector search respect the same article_num metadata filter used
in production (domain → article_num mapping). When domain is None (e.g. eval
queries) the filter is omitted and the full corpus is searched.

The BM25 index and cross-encoder are loaded once at startup via
build_bm25_index() and are stored on app.state to be shared across requests.
"""

import asyncio
import re
import structlog

from config import settings
from models.schemas import ArticleDomain, DocumentChunk
from services.chroma_client import ChromaClient
from services.embedding_service import EmbeddingService

logger = structlog.get_logger()

# ── Domain → article number mapping ──────────────────────────────────────────

_DOMAIN_TO_ARTICLE: dict[ArticleDomain, int] = {
    ArticleDomain.RISK_MANAGEMENT:         9,
    ArticleDomain.DATA_GOVERNANCE:         10,
    ArticleDomain.TECHNICAL_DOCUMENTATION: 11,
    ArticleDomain.RECORD_KEEPING:          12,
    ArticleDomain.TRANSPARENCY:            13,
    ArticleDomain.HUMAN_OVERSIGHT:         14,
    ArticleDomain.SECURITY:                15,
}

# Collections queried for regulatory passages
_COLLECTIONS = ("eu_ai_act", "compliance_checklist")

# Number of candidates to pull from each retriever before merging
_CANDIDATES_PER_RETRIEVER = 10

# RRF smoothing constant (standard value — dampens outlier ranks)
_RRF_K = 60


# ── BM25 index (built at startup) ─────────────────────────────────────────────

class _BM25Index:
    """In-memory BM25 index partitioned by collection and article_num.

    Structure:
        _docs[collection][article_num] = [{"id", "text", "metadata", "distance"}, ...]
        _index[collection][article_num] = BM25Okapi instance

    article_num=None holds the full-corpus index for unfiltered queries.
    """

    def __init__(self) -> None:
        self._docs: dict[str, dict] = {}
        self._index: dict[str, dict] = {}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        return re.sub(r"[^\w\s]", " ", text.lower()).split()

    def build(self, collection: str, passages: list[dict]) -> None:
        """Build BM25 index for a collection from a flat list of passage dicts."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25_not_installed", collection=collection)
            return

        # Partition by article_num
        by_article: dict = {}
        for p in passages:
            art = p["metadata"].get("article_num")
            by_article.setdefault(art, []).append(p)
            by_article.setdefault(None, []).append(p)  # full-corpus partition

        self._docs.setdefault(collection, {})
        self._index.setdefault(collection, {})

        for art_num, docs in by_article.items():
            tokens = [self._tokenize(d["text"]) for d in docs]
            self._docs[collection][art_num] = docs
            self._index[collection][art_num] = BM25Okapi(tokens)

        logger.info(
            "bm25_index_built",
            collection=collection,
            total_docs=len(passages),
            partitions=len(by_article) - 1,  # exclude None
        )

    def search(
        self,
        collection: str,
        query: str,
        article_num: int | None,
        top_k: int,
    ) -> list[dict]:
        """Return top-k BM25 results for a query, optionally filtered by article_num."""
        col_index = self._index.get(collection, {})
        col_docs  = self._docs.get(collection, {})

        index = col_index.get(article_num)
        docs  = col_docs.get(article_num, [])

        if index is None or not docs:
            return []

        tokens = self._tokenize(query)
        scores = index.get_scores(tokens)

        # Pair docs with their BM25 score, sort descending
        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for doc, score in ranked[:top_k]:
            if score > 0:
                results.append({**doc, "bm25_score": float(score)})

        return results

    @property
    def ready(self) -> bool:
        return bool(self._index)


# Singleton — populated during app startup
_bm25: _BM25Index = _BM25Index()


# ── Cross-encoder (loaded at startup) ─────────────────────────────────────────

_cross_encoder = None


def _get_cross_encoder():
    """Lazy-load the cross-encoder model (cached after first call)."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("cross_encoder_loaded", model="ms-marco-MiniLM-L-6-v2")
        except Exception as exc:
            logger.warning("cross_encoder_load_failed", error=str(exc))
    return _cross_encoder


# ── Startup initialisation ───────────────────────────────────────────────────

async def build_bm25_index(chroma_client: ChromaClient) -> None:
    """Fetch all documents from ChromaDB collections and build the BM25 index.

    Called once during app lifespan startup. Stores results in the module-level
    _bm25 singleton so retrieve_requirements() can use it immediately.
    """
    for collection in _COLLECTIONS:
        try:
            # Fetch all documents — get() with no filter returns everything
            col = await chroma_client.get_collection(collection)
            raw = await asyncio.to_thread(
                col.get,
                include=["documents", "metadatas"],
            )

            ids   = raw.get("ids", [])
            docs  = raw.get("documents", [])
            metas = raw.get("metadatas", [])

            passages = [
                {
                    "id":       doc_id,
                    "text":     text,
                    "metadata": meta or {},
                    "distance": 0.0,  # placeholder; BM25 has its own score
                }
                for doc_id, text, meta in zip(ids, docs, metas)
                if text
            ]

            _bm25.build(collection, passages)

        except Exception as exc:
            logger.warning("bm25_index_build_failed", collection=collection, error=str(exc))

    # Pre-load cross-encoder to avoid cold-start on first request
    await asyncio.to_thread(_get_cross_encoder)


# ── RRF merge ────────────────────────────────────────────────────────────────

def _rrf_merge(
    bm25_results: list[dict],
    vector_results: list[dict],
    k: int = _RRF_K,
) -> list[dict]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i)

    Deduplicates by passage id. Returns candidates sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    by_id:  dict[str, dict]  = {}

    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        by_id[doc_id]  = doc

    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        by_id[doc_id]  = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [by_id[doc_id] for doc_id, _ in ranked]


# ── Cross-encoder re-ranking ─────────────────────────────────────────────────

def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Score each (query, passage) pair with the cross-encoder and return top-k.

    Falls back to the RRF-ranked order if the cross-encoder is unavailable.
    """
    if not candidates:
        return []

    encoder = _get_cross_encoder()
    if encoder is None:
        logger.warning("cross_encoder_unavailable_using_rrf_order")
        return candidates[:top_k]

    pairs  = [(query, doc["text"]) for doc in candidates]
    scores = encoder.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    logger.debug(
        "cross_encoder_reranked",
        candidates=len(candidates),
        top_k=top_k,
        top_score=round(float(scores[0]), 3) if len(scores) else 0,
    )

    return [doc for doc, _ in ranked[:top_k]]


# ── Main retrieval function ──────────────────────────────────────────────────

async def retrieve_requirements(
    chunk: DocumentChunk,
    embedding_service: EmbeddingService,
    chroma_client: ChromaClient,
    top_k: int = 5,
    applicable_articles: list[int] | None = None,
    regulation: str | None = None,
) -> list[dict]:
    """Retrieve relevant regulatory passages using hybrid BM25 + vector + re-ranking.

    Steps:
      1. BM25 keyword search  — top _CANDIDATES_PER_RETRIEVER, article-filtered
      2. Vector semantic search — top _CANDIDATES_PER_RETRIEVER, article-filtered
      3. RRF merge into a deduplicated candidate pool
      4. Cross-encoder re-ranks the pool
      5. Return top_k results

    Phase 3 additions:
      - applicable_articles: if provided and chunk's article_num is not in this
        list, retrieval is skipped (returns []) — enforces the applicability gate.
      - regulation: if provided (e.g. "eu_ai_act"), added as a metadata filter
        so only passages from the relevant regulation are retrieved.

    Args:
        chunk: Source document chunk to find regulatory matches for.
        embedding_service: Local embedding model instance.
        chroma_client: ChromaDB async client.
        top_k: Number of passages to return after re-ranking.
        applicable_articles: From ApplicabilityResult — skip retrieval for
                             articles not in this list. None = no restriction.
        regulation: Metadata filter value for the 'regulation' field in ChromaDB.

    Returns:
        List of dicts with keys: id, text, metadata, distance.
    """
    article_num = _DOMAIN_TO_ARTICLE.get(chunk.domain) if chunk.domain else None

    # Phase 3: applicability gate — skip retrieval for non-applicable articles
    if applicable_articles is not None and article_num is not None:
        if article_num not in applicable_articles:
            logger.debug(
                "rag_skipped_non_applicable_article",
                article_num=article_num,
                chunk_id=chunk.chunk_id,
            )
            return []

    # Build metadata where-filter: article_num + optional regulation
    where_filter: dict | None = None
    if article_num is not None:
        where_filter = {"article_num": article_num}
        if regulation:
            where_filter["regulation"] = regulation
    elif regulation:
        where_filter = {"regulation": regulation}

    # ── Stage 1: Vector search ────────────────────────────────────────────────
    vectors = await embedding_service.embed([chunk.text])
    query_vec = vectors[0]

    vector_results: list[dict] = []
    for collection in _COLLECTIONS:
        try:
            raw = await chroma_client.query(
                collection_name=collection,
                query_embeddings=[query_vec],
                n_results=_CANDIDATES_PER_RETRIEVER,
                where=where_filter,
            )
            vector_results.extend(_flatten_result(raw))
        except Exception as exc:
            logger.warning("rag_vector_query_failed", collection=collection, error=str(exc))

    # ── Stage 2: BM25 search ──────────────────────────────────────────────────
    bm25_results: list[dict] = []
    chunk_lang = chunk.language or "en"

    if settings.use_opensearch:
        # Server-side BM25 via OpenSearch (supports native metadata filtering)
        from services.opensearch_client import OpenSearchClient
        os_client = OpenSearchClient(
            host=settings.opensearch_host,
            port=settings.opensearch_port,
        )
        for collection in _COLLECTIONS:
            hits = await os_client.search(
                index=collection,
                query=chunk.text,
                article_num=article_num,
                regulation=regulation,
                lang=chunk_lang,
                top_k=_CANDIDATES_PER_RETRIEVER,
            )
            bm25_results.extend(hits)
    elif _bm25.ready:
        for collection in _COLLECTIONS:
            bm25_results.extend(
                _bm25.search(
                    collection=collection,
                    query=chunk.text,
                    article_num=article_num,
                    top_k=_CANDIDATES_PER_RETRIEVER,
                )
            )
            if regulation:
                bm25_results = [
                    r for r in bm25_results
                    if r["metadata"].get("regulation", "") == regulation
                ]
    else:
        logger.warning("bm25_index_not_ready_falling_back_to_vector_only")

    # ── Stage 3: RRF merge ────────────────────────────────────────────────────
    # Prefer same-language results within each list before merging
    for result_list in (vector_results, bm25_results):
        result_list.sort(key=lambda r: (
            0 if r["metadata"].get("lang") == chunk_lang else 1,
            r.get("distance", 0.0),
        ))

    candidates = _rrf_merge(bm25_results, vector_results)

    # ── Stage 4: Cross-encoder re-ranking ─────────────────────────────────────
    # Run blocking inference in a thread so we don't stall the event loop
    final = await asyncio.to_thread(_rerank, chunk.text, candidates, top_k)

    logger.debug(
        "rag_retrieved",
        chunk_id=chunk.chunk_id,
        bm25_candidates=len(bm25_results),
        vector_candidates=len(vector_results),
        merged_candidates=len(candidates),
        returned=len(final),
        article_filter=article_num,
        regulation_filter=regulation,
    )

    return final


def _flatten_result(chroma_result: dict) -> list[dict]:
    """Convert raw ChromaDB query result into a flat list of passage dicts."""
    passages = []
    ids       = chroma_result.get("ids",       [[]])[0]
    docs      = chroma_result.get("documents", [[]])[0]
    metas     = chroma_result.get("metadatas", [[]])[0]
    distances = chroma_result.get("distances", [[]])[0]

    for doc_id, text, meta, dist in zip(ids, docs, metas, distances):
        if text:
            passages.append({
                "id":       doc_id,
                "text":     text,
                "metadata": meta or {},
                "distance": dist,
            })
    return passages
