"""Tests for compliance scorer."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


def test_classify_risk_tier_prohibited():
    """Biometric real-time keyword yields PROHIBITED tier."""
    from services.compliance_scorer import classify_risk_tier
    from models.schemas import DocumentChunk, RiskTier

    chunks = [DocumentChunk(
        chunk_id="1", text="The system performs real-time biometric identification of employees.",
        source_file="test.txt", chunk_index=0,
    )]
    assert classify_risk_tier(chunks) == RiskTier.PROHIBITED


def test_classify_risk_tier_high():
    """Recruitment keyword yields HIGH tier."""
    from services.compliance_scorer import classify_risk_tier
    from models.schemas import DocumentChunk, RiskTier

    chunks = [DocumentChunk(
        chunk_id="1", text="This AI system is used for recruitment and candidate screening.",
        source_file="test.txt", chunk_index=0,
    )]
    assert classify_risk_tier(chunks) == RiskTier.HIGH


def test_classify_risk_tier_minimal():
    """No sensitive keywords yields MINIMAL tier."""
    from services.compliance_scorer import classify_risk_tier
    from models.schemas import DocumentChunk, RiskTier

    chunks = [DocumentChunk(
        chunk_id="1", text="The chatbot assists customers with product recommendations.",
        source_file="test.txt", chunk_index=0,
    )]
    assert classify_risk_tier(chunks) == RiskTier.MINIMAL


@pytest.mark.asyncio
async def test_overall_score_weighted_average():
    """Overall score is the equally-weighted average of article scores."""
    from services.compliance_scorer import score_audit
    from models.schemas import ArticleDomain, ArticleScore

    scores = [
        ArticleScore(article_num=9,  domain=ArticleDomain.RISK_MANAGEMENT, score=80.0),
        ArticleScore(article_num=10, domain=ArticleDomain.DATA_GOVERNANCE, score=60.0),
        ArticleScore(article_num=11, domain=ArticleDomain.TECHNICAL_DOCUMENTATION, score=70.0),
        ArticleScore(article_num=12, domain=ArticleDomain.RECORD_KEEPING, score=50.0),
        ArticleScore(article_num=13, domain=ArticleDomain.TRANSPARENCY, score=90.0),
        ArticleScore(article_num=14, domain=ArticleDomain.HUMAN_OVERSIGHT, score=40.0),
        ArticleScore(article_num=15, domain=ArticleDomain.SECURITY, score=100.0),
    ]
    report = await score_audit(scores, chunks=[], audit_id="test-001")
    expected = (80 + 60 + 70 + 50 + 90 + 40 + 100) / 7
    assert abs(report.overall_score - round(expected, 1)) < 0.1


@pytest.mark.asyncio
async def test_score_audit_fills_missing_articles():
    """score_audit adds zero-score entries for articles not in input."""
    from services.compliance_scorer import score_audit
    from models.schemas import ArticleDomain, ArticleScore

    # Only provide 2 articles — remaining 5 should be added with score=0
    scores = [
        ArticleScore(article_num=9, domain=ArticleDomain.RISK_MANAGEMENT, score=80.0),
        ArticleScore(article_num=13, domain=ArticleDomain.TRANSPARENCY, score=60.0),
    ]
    report = await score_audit(scores, chunks=[])
    assert len(report.article_scores) == 7
