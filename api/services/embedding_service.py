"""Local sentence-transformers embedding service. (Phase 2)

Model: intfloat/multilingual-e5-small (384-dim, supports DE + EN).
Loaded once at FastAPI startup via the lifespan event — never reloaded per request.
No external API calls; all inference is local.
"""

import structlog

logger = structlog.get_logger()


class EmbeddingService:
    """Wraps sentence-transformers SentenceTransformer for async use.

    The model is loaded in __init__ (blocking) and then used via
    asyncio.to_thread for all encode calls.

    Args:
        model_name: HuggingFace model identifier.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small") -> None:
        raise NotImplementedError("EmbeddingService.__init__ — implemented in Phase 2")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into 384-dimensional vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of 384-dim float vectors, one per input text.
        """
        raise NotImplementedError("EmbeddingService.embed — implemented in Phase 2")
