"""Async httpx wrapper for the Ollama REST API.

Handles generate, generate_json (with retry on parse failure), and health check.
Sequential by design — Ollama processes one request at a time.
"""

import json

import httpx
import structlog

logger = structlog.get_logger()

_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0)


class OllamaClient:
    """Async client for Ollama's /api/generate endpoint.

    Args:
        host: Base URL of Ollama service, e.g. 'http://klarki-ollama:11434'.
        model: Model tag to use for all requests.
    """

    def __init__(self, host: str, model: str) -> None:
        self._host = host.rstrip("/")
        self._model = model

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
        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(f"{self._host}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()

    async def generate_json(self, prompt: str, system: str = "") -> dict:
        """Send a prompt requesting JSON output, retry once on parse failure.

        Uses Ollama's format='json' mode to constrain output.

        Args:
            prompt: User prompt text.
            system: Optional system message.

        Returns:
            Parsed JSON dict from the model.

        Raises:
            ValueError: If JSON cannot be parsed after retry.
        """
        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        if system:
            payload["system"] = system

        for attempt in range(2):
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(f"{self._host}/api/generate", json=payload)
                resp.raise_for_status()
                raw = resp.json().get("response", "").strip()

            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Try extracting the first JSON object from the response
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start != -1 and end > start:
                    try:
                        return json.loads(raw[start:end])
                    except json.JSONDecodeError:
                        pass
                if attempt == 0:
                    logger.warning("ollama_json_parse_retry", raw=raw[:200])
                    continue
                raise ValueError(f"Ollama returned invalid JSON after retry: {raw[:200]}")

        raise ValueError("ollama_generate_json: unreachable")

    async def health_check(self) -> bool:
        """Check if Ollama is reachable and the model is available.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
                resp = await client.get(f"{self._host}/api/tags")
                return resp.status_code == 200
        except Exception as exc:
            logger.warning("ollama_health_fail", error=str(exc))
            return False
