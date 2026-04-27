"""Tests for gap analyser service.

analyse_article() delegates to the 3-node LangGraph (legal → technical → synthesis).
Each test that exercises a LangGraph path must mock generate_json with side_effect
containing 3 sequential responses — one per node — not a single return_value.

Node call order and expected JSON shapes:
  1. legal_agent_node    → {"requirements": ["...", ...]}
  2. technical_agent_node → {"findings": {"req": "Found/Missing: ..."}}
  3. synthesis_agent_node → {"score": int, "reasoning": str,
                             "gaps": [{"title", "description", "severity"}],
                             "recommendations": ["..."]}
"""

import os
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


# ── shared response fixtures ────────────────────────────────────────────────────

def _legal_resp() -> dict:
    return {"requirements": ["Maintain a risk management system", "Document all hazards"]}


def _technical_resp() -> dict:
    return {"findings": {
        "Maintain a risk management system": "Found: risk register updated quarterly",
        "Document all hazards": "Missing: no hazard catalogue provided",
    }}


def _synthesis_resp(score: int = 65) -> dict:
    return {
        "score": score,
        "reasoning": "Risk management partially documented.",
        "gaps": [
            {
                "title": "Missing risk documentation",
                "description": "No formal risk register found in provided documentation.",
                "severity": "major",
            }
        ],
        "recommendations": ["Create and maintain a formal risk register per Article 9(2)."],
    }


def _make_chunk(text: str = "We maintain a risk register and conduct annual risk assessments."):
    from models.schemas import ArticleDomain, DocumentChunk
    return DocumentChunk(
        chunk_id="test-1",
        text=text,
        source_file="policy.txt",
        chunk_index=0,
        domain=ArticleDomain.RISK_MANAGEMENT,
    )


def _make_mock_ollama(*responses) -> AsyncMock:
    """Return a mock whose generate_json yields responses in sequence."""
    mock = AsyncMock()
    mock.generate_json = AsyncMock(side_effect=list(responses))
    return mock


# ── short-circuit paths (no LLM called) ────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyse_article_no_chunks():
    """analyse_article with no chunks returns zero score with a critical gap — no LLM call."""
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
async def test_analyse_article_not_applicable_skips_llm():
    """Phase 3: article outside applicable_articles returns score=100 with no LLM call."""
    from services.gap_analyser import analyse_article
    from models.schemas import ArticleDomain

    mock_ollama = AsyncMock()
    score = await analyse_article(
        article_num=9,
        domain=ArticleDomain.RISK_MANAGEMENT,
        user_chunks=[_make_chunk()],
        regulatory_passages=[],
        ollama=mock_ollama,
        applicable_articles=[10, 11, 12],  # 9 not in list
    )

    assert score.score == 100.0
    assert score.gaps == []
    assert "not applicable" in score.score_reasoning.lower()
    mock_ollama.generate_json.assert_not_called()


@pytest.mark.asyncio
async def test_analyse_article_applicable_articles_none_runs_normally():
    """applicable_articles=None means no gate applied — LangGraph runs for all articles."""
    from services.gap_analyser import analyse_article
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama(_legal_resp(), _technical_resp(), _synthesis_resp())
    score = await analyse_article(
        article_num=9,
        domain=ArticleDomain.RISK_MANAGEMENT,
        user_chunks=[_make_chunk()],
        regulatory_passages=[],
        ollama=mock_ollama,
        applicable_articles=None,
    )

    assert 0 <= score.score <= 100
    assert mock_ollama.generate_json.await_count == 3


@pytest.mark.asyncio
async def test_analyse_article_in_applicable_list_runs_normally():
    """Article inside applicable_articles still runs the full LangGraph path."""
    from services.gap_analyser import analyse_article
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama(_legal_resp(), _technical_resp(), _synthesis_resp())
    score = await analyse_article(
        article_num=9,
        domain=ArticleDomain.RISK_MANAGEMENT,
        user_chunks=[_make_chunk()],
        regulatory_passages=[],
        ollama=mock_ollama,
        applicable_articles=[9, 10, 11],  # 9 is in the list
    )

    assert 0 <= score.score <= 100
    assert mock_ollama.generate_json.await_count == 3


# ── LangGraph happy path ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyse_article_returns_valid_score():
    """analyse_article with 3-response mock returns a well-formed ArticleScore."""
    from services.gap_analyser import analyse_article
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama(_legal_resp(), _technical_resp(), _synthesis_resp())
    score = await analyse_article(
        article_num=9,
        domain=ArticleDomain.RISK_MANAGEMENT,
        user_chunks=[_make_chunk()],
        regulatory_passages=[],
        ollama=mock_ollama,
    )

    assert 0 <= score.score <= 100
    assert score.article_num == 9
    assert score.chunk_count == 1
    # synthesis produced 1 gap and 1 recommendation
    assert len(score.gaps) == 1
    assert score.gaps[0].title == "Missing risk documentation"
    assert len(score.recommendations) == 1
    assert mock_ollama.generate_json.await_count == 3


@pytest.mark.asyncio
async def test_analyse_article_uses_synthesis_score():
    """Score in the returned ArticleScore matches what the synthesis node returned."""
    from services.gap_analyser import analyse_article
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama(_legal_resp(), _technical_resp(), _synthesis_resp(score=82))
    score = await analyse_article(
        article_num=9,
        domain=ArticleDomain.RISK_MANAGEMENT,
        user_chunks=[_make_chunk()],
        regulatory_passages=[],
        ollama=mock_ollama,
    )

    assert score.score == 82.0


@pytest.mark.asyncio
async def test_analyse_article_score_clamped_high():
    """Synthesis returning score > 100 is clamped to 100 by gap_analyser."""
    from services.gap_analyser import analyse_article
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama(
        _legal_resp(),
        _technical_resp(),
        _synthesis_resp(score=150),
    )
    score = await analyse_article(
        article_num=9,
        domain=ArticleDomain.RISK_MANAGEMENT,
        user_chunks=[_make_chunk()],
        regulatory_passages=[],
        ollama=mock_ollama,
    )

    assert score.score == 100.0


@pytest.mark.asyncio
async def test_analyse_article_score_clamped_low():
    """Synthesis returning score < 0 is clamped to 0 by gap_analyser."""
    from services.gap_analyser import analyse_article
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama(
        _legal_resp(),
        _technical_resp(),
        _synthesis_resp(score=-10),
    )
    score = await analyse_article(
        article_num=9,
        domain=ArticleDomain.RISK_MANAGEMENT,
        user_chunks=[_make_chunk()],
        regulatory_passages=[],
        ollama=mock_ollama,
    )

    assert score.score == 0.0


@pytest.mark.asyncio
async def test_analyse_article_empty_gap_title_filtered():
    """Gaps with empty title or description are silently filtered out."""
    from services.gap_analyser import analyse_article
    from models.schemas import ArticleDomain

    synthesis_with_bad_gap = {
        "score": 60,
        "reasoning": "Some gaps found.",
        "gaps": [
            {"title": "", "description": "No description", "severity": "minor"},
            {"title": "Real gap", "description": "", "severity": "major"},
            {"title": "Valid gap", "description": "This one is fine.", "severity": "minor"},
        ],
        "recommendations": [],
    }
    mock_ollama = _make_mock_ollama(_legal_resp(), _technical_resp(), synthesis_with_bad_gap)
    score = await analyse_article(
        article_num=9,
        domain=ArticleDomain.RISK_MANAGEMENT,
        user_chunks=[_make_chunk()],
        regulatory_passages=[],
        ollama=mock_ollama,
    )

    # Only the gap with both title and description survives
    assert len(score.gaps) == 1
    assert score.gaps[0].title == "Valid gap"


@pytest.mark.asyncio
async def test_analyse_article_regulatory_passages_mapped():
    """Regulatory passages from ChromaDB are mapped to RegulatoryPassage objects."""
    from services.gap_analyser import analyse_article
    from models.schemas import ArticleDomain

    passages = [
        {
            "text": "Providers shall establish a risk management system.",
            "metadata": {
                "title": "Article 9 Risk Management",
                "article_num": 9,
                "regulation": "eu_ai_act",
            },
        }
    ]
    mock_ollama = _make_mock_ollama(_legal_resp(), _technical_resp(), _synthesis_resp())
    score = await analyse_article(
        article_num=9,
        domain=ArticleDomain.RISK_MANAGEMENT,
        user_chunks=[_make_chunk()],
        regulatory_passages=passages,
        ollama=mock_ollama,
    )

    assert len(score.regulatory_passages) == 1
    assert "risk management" in score.regulatory_passages[0].text.lower()


# ── LangGraph failure fallback ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyse_article_langgraph_failure_returns_fallback():
    """If the LangGraph graph.ainvoke() raises, gap_analyser returns score=30 fallback."""
    from services.gap_analyser import analyse_article
    from models.schemas import ArticleDomain, Severity

    mock_ollama = AsyncMock()
    # All LLM calls raise so the graph itself will propagate an error
    mock_ollama.generate_json = AsyncMock(side_effect=RuntimeError("model crashed"))

    score = await analyse_article(
        article_num=9,
        domain=ArticleDomain.RISK_MANAGEMENT,
        user_chunks=[_make_chunk()],
        regulatory_passages=[],
        ollama=mock_ollama,
    )

    # Fallback ArticleScore — gap_analyser catches graph.ainvoke() exception
    assert score.score == 30.0
    assert score.chunk_count == 1
    assert len(score.gaps) == 1
    assert score.gaps[0].severity == Severity.MAJOR
    assert "failed" in score.score_reasoning.lower()
