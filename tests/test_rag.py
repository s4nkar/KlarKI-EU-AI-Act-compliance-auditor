"""Tests for the RAG retrieval engine."""

import pytest


@pytest.mark.asyncio
async def test_retrieve_returns_top_k():
    """Retrieval returns at most top_k results."""
    pytest.skip("Requires live ChromaDB — run with ./run.sh test")


@pytest.mark.asyncio
async def test_retrieve_prefers_same_language():
    """Same-language passages ranked above cross-language results."""
    pytest.skip("Requires live ChromaDB — run with ./run.sh test")