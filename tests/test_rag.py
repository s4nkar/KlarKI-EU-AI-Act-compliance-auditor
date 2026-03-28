"""Tests for RAG retrieval engine. (Phase 2)"""

import pytest


@pytest.mark.asyncio
async def test_retrieve_returns_top_k(seeded_chroma):
    """Retrieval returns at most top_k results."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_retrieve_prefers_same_language(seeded_chroma):
    """Same-language passages ranked above cross-language results."""
    pytest.skip("Implemented in Phase 2")
