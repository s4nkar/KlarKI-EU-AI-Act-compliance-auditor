"""Shared pytest fixtures for KlarKI test suite.

Provides:
    - test_client: async FastAPI TestClient
    - mock_ollama: httpx mock for Ollama API calls
    - seeded_chroma: in-memory ChromaDB with test data
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# ── App fixture ────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def test_client():
    """Async HTTPX client pointed at the FastAPI app.

    Yields:
        AsyncClient configured with the FastAPI ASGI app.
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


# ── ChromaDB fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def seeded_chroma(tmp_path):
    """In-memory ChromaDB client with minimal test data in all 3 collections.

    Yields:
        chromadb.Client instance with eu_ai_act, gdpr, compliance_checklist collections.
    """
    import chromadb
    client = chromadb.EphemeralClient()

    # Minimal seed — each collection has at least one document
    for name in ["eu_ai_act", "gdpr", "compliance_checklist"]:
        col = client.create_collection(name)
        col.upsert(
            ids=[f"{name}_test_001"],
            documents=[f"Test document for {name} collection"],
            embeddings=[[0.0] * 384],
            metadatas=[{"article_num": 9, "domain": "risk_management", "lang": "en"}],
        )
    yield client


# ── Ollama mock fixture ────────────────────────────────────────────────────────

@pytest.fixture
def mock_ollama_classify():
    """Return a callable that simulates Ollama classify response."""
    def _respond(prompt: str) -> str:
        return "risk_management"
    return _respond


@pytest.fixture
def mock_ollama_gap_analysis():
    """Return a callable that simulates Ollama gap analysis JSON response."""
    def _respond(prompt: str) -> dict:
        return {
            "score": 65,
            "gaps": [
                {
                    "title": "Missing risk documentation",
                    "description": "No formal risk register found in provided documentation.",
                    "severity": "major",
                }
            ],
            "recommendations": ["Create and maintain a formal risk register per Article 9(2)."],
        }
    return _respond
