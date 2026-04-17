"""Shared pytest fixtures for KlarKI test suite."""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# ── Session-scoped spaCy NER model ────────────────────────────────────────────
# Loaded ONCE per pytest session and shared across test_ner.py and eval_ner.py.
# Prevents the ~750 MB de_core_news_lg from being loaded multiple times which
# would exhaust container memory and cause an OOM kill with no summary printed.

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def spacy_ner_nlp():
    """Return the loaded spaCy NER model, or None if unavailable."""
    model_path = next(
        (p for p in [
            _REPO_ROOT / "training" / "spacy_ner_model" / "model-final",
            Path("/training/spacy_ner_model/model-final"),
        ] if p.exists()),
        None,
    )
    if model_path is None:
        return None
    try:
        import spacy
        return spacy.load(str(model_path))
    except Exception:
        return None

# Support both local dev (repo/api) and Docker container (/app)
_local_api = os.path.join(os.path.dirname(__file__), "..", "api")
_container_api = "/app"
_api_path = _local_api if os.path.isdir(_local_api) else _container_api
sys.path.insert(0, _api_path)


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
