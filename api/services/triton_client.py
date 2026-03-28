"""Triton Inference Server gRPC client. (Phase 5)

Provides an async interface to the Triton ensemble model:
BERT clause classifier + e5 embeddings + spaCy NER.
Only used when USE_TRITON=true in config.
"""

import structlog

logger = structlog.get_logger()


class TritonClient:
    """Async gRPC client for Triton Inference Server.

    Args:
        host: Triton server hostname.
        grpc_port: Triton gRPC port (default 8001).
    """

    def __init__(self, host: str, grpc_port: int = 8001) -> None:
        self._address = f"{host}:{grpc_port}"

    async def classify(self, texts: list[str]) -> list[str]:
        """Classify texts using the BERT clause classifier ensemble.

        Args:
            texts: List of text strings to classify.

        Returns:
            List of ArticleDomain value strings.
        """
        raise NotImplementedError("TritonClient.classify — implemented in Phase 5")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via Triton e5 model.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of 384-dim float vectors.
        """
        raise NotImplementedError("TritonClient.embed — implemented in Phase 5")

    async def health_check(self) -> bool:
        """Check Triton server readiness.

        Returns:
            True if server is ready, False otherwise.
        """
        raise NotImplementedError("TritonClient.health_check — implemented in Phase 5")
