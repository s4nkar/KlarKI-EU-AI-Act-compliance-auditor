"""Local sentence-transformers embedding service with in-memory cache.

Model: intfloat/multilingual-e5-small (384-dim, supports DE + EN).
Loaded once at FastAPI startup via the lifespan event — never reloaded per request.
No external API calls; all inference is local CPU.

Caching: each unique text is embedded once per server lifetime.
Cache key = SHA-256(text). Same text → same vector (embeddings are deterministic),
so the cache never returns stale data. Cache lives in RAM; it is cleared on restart,
which is the correct behaviour — no stale embeddings across model updates.
"""

import asyncio
import hashlib

import structlog
from sentence_transformers import SentenceTransformer

from config import settings

logger = structlog.get_logger()


class EmbeddingService:
    """Wraps sentence-transformers SentenceTransformer for async use with caching.

    The model is loaded in __init__ (blocking, called once in lifespan)
    and then used via asyncio.to_thread for all encode calls so the
    FastAPI event loop is never blocked during inference.

    Args:
        model_name: HuggingFace model identifier.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small") -> None:
        logger.info("embedding_model_loading", model=model_name)
        self.model = SentenceTransformer(model_name)
        self._model_name = model_name
        # Cache: SHA-256(text) → 384-dim vector
        self._cache: dict[str, list[float]] = {}
        logger.info("embedding_model_ready", model=model_name)

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into 384-dimensional vectors.

        Checks the in-memory cache first. Only texts not already cached
        are sent to the model. Results are stored back in the cache.
        Runs encoding in a thread pool to avoid blocking the event loop.

        Args:
            texts: List of strings to embed.

        Returns:
            List of 384-dim float vectors, one per input text.
        """
        if not texts:
            return []

        keys = [self._cache_key(t) for t in texts]

        # Separate already-cached from uncached
        uncached_indices = [i for i, k in enumerate(keys) if k not in self._cache]
        uncached_texts = [texts[i] for i in uncached_indices]

        cache_hits = len(texts) - len(uncached_texts)
        if cache_hits:
            logger.debug("embedding_cache_hit", hits=cache_hits, misses=len(uncached_texts))

        # Encode only the uncached texts
        if uncached_texts:
            new_vectors: list[list[float]] = await self._encode_batch(uncached_texts)

            for i, vec in zip(uncached_indices, new_vectors, strict=True):
                self._cache[keys[i]] = vec

            logger.debug("embedding_done", encoded=len(uncached_texts), cache_size=len(self._cache))

        # Reconstruct result in original order (all from cache now)
        return [self._cache[k] for k in keys]

    async def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts — Triton GPU path when enabled, local CPU otherwise."""
        if settings.use_triton:
            try:
                from services.triton_client import TritonClient
                client = TritonClient(host=settings.triton_host, grpc_port=settings.triton_grpc_port)
                vecs = await client.embed(texts)
                logger.debug("embedding_via_triton", count=len(texts))
                return vecs
            except Exception as exc:
                logger.warning("triton_embed_fallback", error=str(exc), fallback="local")

        def _local() -> list[list[float]]:
            return self.model.encode(texts, normalize_embeddings=True).tolist()

        return await asyncio.to_thread(_local)

    @property
    def cache_size(self) -> int:
        """Number of unique texts currently cached."""
        return len(self._cache)
