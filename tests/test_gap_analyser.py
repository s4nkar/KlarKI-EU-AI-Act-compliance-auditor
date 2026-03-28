"""Tests for gap analyser service. (Phase 2)"""

import pytest


@pytest.mark.asyncio
async def test_analyse_article_returns_article_score(mock_ollama_gap_analysis):
    """analyse_article returns a valid ArticleScore."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_analyse_article_score_in_range(mock_ollama_gap_analysis):
    """Score is between 0 and 100."""
    pytest.skip("Implemented in Phase 2")
