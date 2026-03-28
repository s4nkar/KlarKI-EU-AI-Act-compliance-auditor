"""Shared pytest fixtures for KlarKI test suite."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


# ── App fixture ────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def test_client():
    """Async HTTPX client pointed at the FastAPI app.

    Mocks app.state so lifespan services (ChromaDB, EmbeddingService) don't
    need to be running during unit/integration tests.
    """
    from httpx import AsyncClient, ASGITransport
    from main import app
    from services.chroma_client import ChromaClient

    # Provide minimal mock state so endpoints can access app.state
    mock_chroma = AsyncMock(spec=ChromaClient)
    mock_chroma.health_check.return_value = True
    app.state.chroma = mock_chroma
    app.state.embeddings = MagicMock()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


# ── ChromaDB fixture ───────────────────────────────────────────────────────────

@pytest.fixture
def seeded_chroma():
    """In-memory ChromaDB client with minimal test data in all 3 collections.

    Uses get_or_create_collection to be safe across chromadb 1.x which may
    not fully reset EphemeralClient state between test runs.
    """
    import chromadb
    client = chromadb.EphemeralClient()

    for name in ["eu_ai_act", "gdpr", "compliance_checklist"]:
        col = client.get_or_create_collection(name)
        col.upsert(
            ids=[f"{name}_test_001"],
            documents=[f"Test document for {name} collection"],
            embeddings=[[0.0] * 384],
            metadatas=[{"article_num": 9, "domain": "risk_management", "lang": "en"}],
        )
    yield client


# ── Ollama mock fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def mock_ollama_classify():
    """Simulated Ollama classify response — returns domain label string."""
    return "risk_management"


@pytest.fixture
def mock_ollama_gap_analysis():
    """Simulated Ollama gap analysis JSON response — returns dict directly."""
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
