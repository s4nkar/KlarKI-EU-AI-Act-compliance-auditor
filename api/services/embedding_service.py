"""Local sentence-transformers embedding service.

Model: intfloat/multilingual-e5-small (384-dim, supports DE + EN).
Loaded once at FastAPI startup via the lifespan event — never reloaded per request.
No external API calls; all inference is local CPU.
"""

import asyncio

import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()


class EmbeddingService:
    """Wraps sentence-transformers SentenceTransformer for async use.

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
        logger.info("embedding_model_ready", model=model_name)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into 384-dimensional vectors.

        Runs in a thread pool to avoid blocking the event loop.
        Vectors are L2-normalised (cosine similarity = dot product).

        Args:
            texts: List of strings to embed.

        Returns:
            List of 384-dim float vectors, one per input text.
        """
        if not texts:
            return []

        def _encode() -> list[list[float]]:
            vectors = self.model.encode(texts, normalize_embeddings=True)
            return vectors.tolist()

        result = await asyncio.to_thread(_encode)
        logger.debug("embedding_done", count=len(texts))
        return result
