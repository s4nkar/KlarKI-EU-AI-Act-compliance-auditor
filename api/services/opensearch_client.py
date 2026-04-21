"""OpenSearch BM25 retrieval client — opt-in alternative to in-memory rank_bm25.

When USE_OPENSEARCH=true, rag_engine.py calls this module for keyword search
instead of the in-memory BM25Okapi index. Vector search remains in ChromaDB.

Index layout mirrors ChromaDB collection names:
  - eu_ai_act
  - gdpr
  - compliance_checklist

Each document is stored with the same metadata fields used by ChromaDB:
  regulation, article_num, domain, lang, chunk_index, title

Usage (called from rag_engine.py):
    from services.opensearch_client import OpenSearchClient
    client = OpenSearchClient(host="localhost", port=9200)
    results = await client.search("eu_ai_act", query, article_num=9, top_k=10)
"""

from __future__ import annotations

import asyncio
import structlog

logger = structlog.get_logger()

_INDICES = ("eu_ai_act", "gdpr", "compliance_checklist")


class OpenSearchClient:
    """Thin async wrapper for OpenSearch BM25 full-text search.

    All blocking OpenSearch SDK calls run via asyncio.to_thread so the
    event loop is never blocked.
    """

    def __init__(self, host: str = "localhost", port: int = 9200) -> None:
        self._host = host
        self._port = port
        self._client = None

    def _get_client(self):
        """Lazy-connect to OpenSearch. Returns None if package unavailable."""
        if self._client is not None:
            return self._client
        try:
            from opensearchpy import OpenSearch
            self._client = OpenSearch(
                hosts=[{"host": self._host, "port": self._port}],
                http_compress=True,
                use_ssl=False,
                verify_certs=False,
                timeout=10,
            )
            logger.info("opensearch_connected", host=self._host, port=self._port)
        except ImportError:
            logger.warning(
                "opensearch_py_not_installed",
                hint="pip install opensearch-py",
            )
        except Exception as exc:
            logger.warning("opensearch_connect_failed", error=str(exc))
        return self._client

    # ── Index management ──────────────────────────────────────────────────────

    def _ensure_index(self, client, index: str) -> None:
        """Create index with appropriate mappings if it does not exist."""
        if client.indices.exists(index=index):
            return
        mapping = {
            "mappings": {
                "properties": {
                    "text":        {"type": "text", "analyzer": "standard"},
                    "regulation":  {"type": "keyword"},
                    "article_num": {"type": "integer"},
                    "domain":      {"type": "keyword"},
                    "lang":        {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "title":       {"type": "text"},
                    "doc_id":      {"type": "keyword"},
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
        }
        client.indices.create(index=index, body=mapping)
        logger.info("opensearch_index_created", index=index)

    async def create_indices(self) -> None:
        """Create all three regulatory indices if they do not exist."""
        client = self._get_client()
        if client is None:
            return
        for index in _INDICES:
            await asyncio.to_thread(self._ensure_index, client, index)

    async def index_documents(self, index: str, documents: list[dict]) -> int:
        """Bulk-index a list of document dicts into OpenSearch.

        Each dict must have: id, text, metadata (dict with regulation,
        article_num, domain, lang, chunk_index, title).

        Returns number of documents indexed.
        """
        client = self._get_client()
        if client is None or not documents:
            return 0

        def _bulk() -> int:
            from opensearchpy.helpers import bulk
            await_self = self
            actions = []
            for doc in documents:
                body = {
                    "_index": index,
                    "_id": doc["id"],
                    "_source": {
                        "text": doc["text"],
                        "doc_id": doc["id"],
                        **doc.get("metadata", {}),
                    },
                }
                actions.append(body)
            success, _ = bulk(client, actions, raise_on_error=False)
            return success

        count = await asyncio.to_thread(_bulk)
        logger.info("opensearch_indexed", index=index, count=count)
        return count

    async def delete_index(self, index: str) -> None:
        """Delete an index (called during rebuild)."""
        client = self._get_client()
        if client is None:
            return
        await asyncio.to_thread(
            lambda: client.indices.delete(index=index, ignore_unavailable=True)
        )
        logger.info("opensearch_index_deleted", index=index)

    # ── Search ────────────────────────────────────────────────────────────────

    def _do_search(
        self,
        client,
        index: str,
        query: str,
        article_num: int | None,
        regulation: str | None,
        lang: str | None,
        top_k: int,
    ) -> list[dict]:
        """Execute BM25 match query with optional metadata filters."""
        must: list[dict] = [{"match": {"text": {"query": query, "operator": "or"}}}]
        filters: list[dict] = []

        if article_num is not None:
            filters.append({"term": {"article_num": article_num}})
        if regulation:
            filters.append({"term": {"regulation": regulation}})

        # Boost same-language results without excluding others
        should: list[dict] = []
        if lang:
            should.append({"term": {"lang": {"value": lang, "boost": 1.5}}})

        body: dict = {
            "query": {
                "bool": {
                    "must": must,
                    "filter": filters,
                    "should": should,
                }
            },
            "size": top_k,
        }

        try:
            resp = client.search(index=index, body=body)
        except Exception as exc:
            logger.warning("opensearch_search_failed", index=index, error=str(exc))
            return []

        results = []
        for hit in resp.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            results.append({
                "id":   hit["_id"],
                "text": src.get("text", ""),
                "metadata": {
                    "regulation":  src.get("regulation", ""),
                    "article_num": src.get("article_num"),
                    "domain":      src.get("domain", ""),
                    "lang":        src.get("lang", ""),
                    "chunk_index": src.get("chunk_index", 0),
                    "title":       src.get("title", ""),
                },
                "distance": 0.0,
                "bm25_score": hit.get("_score", 0.0),
            })
        return results

    async def search(
        self,
        index: str,
        query: str,
        article_num: int | None = None,
        regulation: str | None = None,
        lang: str | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Async BM25 full-text search. Returns passage dicts compatible with rag_engine."""
        client = self._get_client()
        if client is None:
            return []
        return await asyncio.to_thread(
            self._do_search, client, index, query, article_num, regulation, lang, top_k
        )

    async def health(self) -> bool:
        """Return True if OpenSearch is reachable."""
        client = self._get_client()
        if client is None:
            return False
        try:
            info = await asyncio.to_thread(client.info)
            return bool(info)
        except Exception:
            return False
