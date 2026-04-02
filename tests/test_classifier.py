"""Tests for the chunk classifier service (Ollama and Triton backends)."""

import pytest


@pytest.mark.asyncio
async def test_classify_sets_domain():
    """Classifier sets .domain on each chunk."""
    pytest.skip("Requires live Ollama container — run with ./run.sh test")


@pytest.mark.asyncio
async def test_classify_unknown_label_maps_to_unrelated():
    """Unrecognised LLM output maps to ArticleDomain.UNRELATED."""
    pytest.skip("Requires live Ollama container — run with ./run.sh test")
