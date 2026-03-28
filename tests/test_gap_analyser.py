"""Tests for gap analyser service."""

import os
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.mark.asyncio
async def test_analyse_article_no_chunks():
    """analyse_article with no chunks returns zero score with critical gap."""
    from services.gap_analyser import analyse_article
    from models.schemas import ArticleDomain, Severity

    mock_ollama = AsyncMock()
    score = await analyse_article(
        article_num=9,
        domain=ArticleDomain.RISK_MANAGEMENT,
        user_chunks=[],
        regulatory_passages=[],
        ollama=mock_ollama,
    )
    assert score.score == 0.0
    assert score.chunk_count == 0
    assert len(score.gaps) == 1
    assert score.gaps[0].severity == Severity.CRITICAL
    mock_ollama.generate_json.assert_not_called()


@pytest.mark.asyncio
async def test_analyse_article_returns_valid_score(mock_ollama_gap_analysis):
    """analyse_article with mock LLM returns valid ArticleScore."""
    from services.gap_analyser import analyse_article
    from models.schemas import ArticleDomain, DocumentChunk

    mock_ollama = AsyncMock()
    mock_ollama.generate_json.return_value = mock_ollama_gap_analysis

    chunk = DocumentChunk(
        chunk_id="test-1",
        text="We maintain a risk register and conduct annual risk assessments.",
        source_file="policy.txt",
        chunk_index=0,
        domain=ArticleDomain.RISK_MANAGEMENT,
    )

    score = await analyse_article(
        article_num=9,
        domain=ArticleDomain.RISK_MANAGEMENT,
        user_chunks=[chunk],
        regulatory_passages=[],
        ollama=mock_ollama,
    )

    assert 0 <= score.score <= 100
    assert score.article_num == 9
    assert score.chunk_count == 1


@pytest.mark.asyncio
async def test_analyse_article_score_clamped():
    """LLM returning score > 100 or < 0 is clamped to valid range."""
    from services.gap_analyser import analyse_article
    from models.schemas import ArticleDomain, DocumentChunk

    mock_ollama = AsyncMock()
    mock_ollama.generate_json.return_value = {
        "score": 150,  # out of range
        "gaps": [],
        "recommendations": [],
    }

    chunk = DocumentChunk(
        chunk_id="test-1", text="Some text.", source_file="f.txt", chunk_index=0,
    )
    score = await analyse_article(9, ArticleDomain.RISK_MANAGEMENT, [chunk], [], mock_ollama)
    assert score.score == 100.0
