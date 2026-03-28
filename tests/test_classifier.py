"""Tests for LLM chunk classifier. (Phase 2)"""

import pytest


@pytest.mark.asyncio
async def test_classify_sets_domain(mock_ollama_classify):
    """Classifier sets .domain on each chunk."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_classify_unknown_label_maps_to_unrelated(mock_ollama_classify):
    """Unrecognised LLM output maps to ArticleDomain.UNRELATED."""
    pytest.skip("Implemented in Phase 2")
