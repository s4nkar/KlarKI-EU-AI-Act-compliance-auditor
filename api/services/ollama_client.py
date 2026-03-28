"""Async httpx wrapper for the Ollama REST API. (Phase 2)

Handles generate, generate_json (with retry), and health check.
All calls use a shared AsyncClient with keep-alive for efficiency.
"""

import json

import httpx
import structlog

logger = structlog.get_logger()


class OllamaClient:
    """Async client for Ollama's /api/generate endpoint.

    Args:
        host: Base URL of Ollama service, e.g. 'http://klarki-ollama:11434'.
        model: Model tag to use for all requests.
        timeout: HTTP timeout in seconds.
    """

    def __init__(self, host: str, model: str, timeout: float = 120.0) -> None:
        self._host = host
        self._model = model
        self._timeout = timeout

    async def generate(self, prompt: str, system: str = "") -> str:
        """Send a prompt and return the raw text response.

        Args:
            prompt: User prompt text.
            system: Optional system message.

        Returns:
            Model response string.

        Raises:
            httpx.HTTPError: On network or server errors.
        """
        raise NotImplementedError("OllamaClient.generate — implemented in Phase 2")

    async def generate_json(self, prompt: str, system: str = "") -> dict:
        """Send a prompt requesting JSON output, retry once on parse failure.

        Args:
            prompt: User prompt text.
            system: Optional system message.

        Returns:
            Parsed JSON dict from the model.

        Raises:
            ValueError: If JSON cannot be parsed after retry.
        """
        raise NotImplementedError("OllamaClient.generate_json — implemented in Phase 2")

    async def health_check(self) -> bool:
        """Check if Ollama is reachable and the target model is available.

        Returns:
            True if healthy, False otherwise.
        """
        raise NotImplementedError("OllamaClient.health_check — implemented in Phase 2")
