"""Async ChromaDB client wrapper.

Provides a thin, async-friendly interface over the synchronous ChromaDB Python
client. All blocking I/O is executed in a thread-pool via asyncio.to_thread so
FastAPI coroutines are never blocked.

Collections managed:
    eu_ai_act          — Full text of EU AI Act (DE + EN), chunked per article
    gdpr               — Full text of GDPR (DE + EN), chunked per article
    compliance_checklist — ~85 structured requirements from Articles 9–15
"""

import asyncio
from typing import Any

import chromadb
import structlog

logger = structlog.get_logger()

COLLECTIONS = ["eu_ai_act", "gdpr", "compliance_checklist"]


class ChromaClient:
    """Async wrapper around chromadb.HttpClient.

    Args:
        host: Full URL of the ChromaDB service, e.g. 'http://klarki-chromadb:8000'.
    """

    def __init__(self, host: str) -> None:
        # Parse host/port from URL
        from urllib.parse import urlparse
        parsed = urlparse(host)
        chroma_host = parsed.hostname or "localhost"
        chroma_port = parsed.port or 8000
        self._client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        self._host = host
        logger.info("chroma_client_init", host=host)

    # ── Health ──────────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Ping ChromaDB heartbeat endpoint.

        Returns:
            True if ChromaDB responds, False otherwise.
        """
        try:
            await asyncio.to_thread(self._client.heartbeat)
            return True
        except Exception as exc:
            logger.warning("chroma_health_fail", error=str(exc))
            return False

    # ── Collection helpers ──────────────────────────────────────────────────

    async def get_or_create_collection(
        self,
        name: str,
        metadata: dict | None = None,
    ) -> chromadb.Collection:
        """Get collection by name, creating it if it does not exist.

        Args:
            name: Collection name. Must be one of COLLECTIONS.
            metadata: Optional metadata dict passed on creation.

        Returns:
            ChromaDB Collection object.
        """
        return await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=name,
            metadata=metadata or {},
        )

    async def get_collection(self, name: str) -> chromadb.Collection:
        """Fetch an existing collection by name.

        Args:
            name: Collection name.

        Returns:
            ChromaDB Collection object.

        Raises:
            ValueError: If the collection does not exist.
        """
        return await asyncio.to_thread(self._client.get_collection, name=name)

    async def list_collections(self) -> list[str]:
        """Return names of all collections currently in ChromaDB.

        Returns:
            List of collection name strings.
        """
        cols = await asyncio.to_thread(self._client.list_collections)
        # chromadb 1.x returns str names directly; 0.5.x returns Collection objects
        return [c if isinstance(c, str) else c.name for c in cols]

    # ── CRUD ────────────────────────────────────────────────────────────────

    async def upsert(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Upsert documents with embeddings into a collection.

        Args:
            collection_name: Target collection.
            ids: Unique IDs for each document.
            embeddings: Corresponding embedding vectors.
            documents: Raw text for each document.
            metadatas: Metadata dicts aligned with documents.
        """
        col = await self.get_or_create_collection(collection_name)
        await asyncio.to_thread(
            col.upsert,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug("chroma_upsert", collection=collection_name, count=len(ids))

    async def query(
        self,
        collection_name: str,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict[str, Any]:
        """Semantic search against a collection.

        Args:
            collection_name: Collection to query.
            query_embeddings: One or more query vectors.
            n_results: Number of nearest neighbours to return.
            where: Optional ChromaDB metadata filter dict.

        Returns:
            Raw ChromaDB query result dict with keys: ids, documents, metadatas, distances.
        """
        col = await self.get_collection(collection_name)
        kwargs: dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        result = await asyncio.to_thread(col.query, **kwargs)
        logger.debug(
            "chroma_query",
            collection=collection_name,
            n_results=n_results,
            returned=len(result.get("ids", [[]])[0]),
        )
        return result

    async def count(self, collection_name: str) -> int:
        """Return the number of documents in a collection.

        Args:
            collection_name: Collection name.

        Returns:
            Document count as integer.
        """
        col = await self.get_collection(collection_name)
        return await asyncio.to_thread(col.count)

    async def delete_collection(self, name: str) -> None:
        """Delete a collection by name (used in tests / rebuilds).

        Args:
            name: Collection name to delete.
        """
        await asyncio.to_thread(self._client.delete_collection, name=name)
        logger.info("chroma_collection_deleted", name=name)
