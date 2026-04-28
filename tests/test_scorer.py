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


# ── Phase 3 — applicability-aware scoring ────────────────────────────────────


def _make_applicability(
    is_high_risk: bool = True,
    is_prohibited: bool = False,
    applicable_articles: list | None = None,
    gdpr_applicable_articles: list | None = None,
):
    from models.schemas import ApplicabilityResult
    return ApplicabilityResult(
        is_high_risk=is_high_risk,
        is_prohibited=is_prohibited,
        annex_iii_matches=[],
        annex_i_triggered=False,
        applicable_articles=applicable_articles or [],
        gdpr_applicable_articles=gdpr_applicable_articles or [],
        reasoning="test",
    )


@pytest.mark.asyncio
async def test_phase3_overall_score_averages_only_applicable_articles():
    """With applicability set, only applicable articles count in overall score."""
    from services.compliance_scorer import score_audit
    from models.schemas import ArticleDomain, ArticleScore

    scores = [
        ArticleScore(article_num=9,  domain=ArticleDomain.RISK_MANAGEMENT, score=80.0),
        ArticleScore(article_num=10, domain=ArticleDomain.DATA_GOVERNANCE, score=40.0),
        ArticleScore(article_num=11, domain=ArticleDomain.TECHNICAL_DOCUMENTATION, score=60.0),
        ArticleScore(article_num=12, domain=ArticleDomain.RECORD_KEEPING, score=0.0),
        ArticleScore(article_num=13, domain=ArticleDomain.TRANSPARENCY, score=100.0),
        ArticleScore(article_num=14, domain=ArticleDomain.HUMAN_OVERSIGHT, score=0.0),
        ArticleScore(article_num=15, domain=ArticleDomain.SECURITY, score=0.0),
    ]
    applicability = _make_applicability(applicable_articles=[9, 10])
    report = await score_audit(scores, chunks=[], applicability=applicability)
    # Only Arts 9 (80) and 10 (40) should be averaged
    assert abs(report.overall_score - 60.0) < 0.1


@pytest.mark.asyncio
async def test_phase3_minimal_risk_no_applicable_articles_scores_100():
    """Minimal-risk system with no applicable articles → overall_score=100."""
    from services.compliance_scorer import score_audit
    from models.schemas import ArticleDomain, ArticleScore

    scores = [
        ArticleScore(article_num=9, domain=ArticleDomain.RISK_MANAGEMENT, score=50.0),
    ]
    applicability = _make_applicability(is_high_risk=False, applicable_articles=[])
    report = await score_audit(scores, chunks=[], applicability=applicability)
    assert report.overall_score == 100.0


@pytest.mark.asyncio
async def test_phase3_risk_tier_from_is_prohibited():
    """applicability.is_prohibited=True → ComplianceReport.risk_tier=PROHIBITED."""
    from services.compliance_scorer import score_audit
    from models.schemas import ArticleDomain, ArticleScore, RiskTier

    scores = [ArticleScore(article_num=9, domain=ArticleDomain.RISK_MANAGEMENT, score=0.0)]
    applicability = _make_applicability(is_high_risk=False, is_prohibited=True, applicable_articles=[5])
    report = await score_audit(scores, chunks=[], applicability=applicability)
    assert report.risk_tier == RiskTier.PROHIBITED


@pytest.mark.asyncio
async def test_phase3_risk_tier_from_is_high_risk():
    """applicability.is_high_risk=True → ComplianceReport.risk_tier=HIGH."""
    from services.compliance_scorer import score_audit
    from models.schemas import ArticleDomain, ArticleScore, RiskTier

    scores = [ArticleScore(article_num=9, domain=ArticleDomain.RISK_MANAGEMENT, score=70.0)]
    applicability = _make_applicability(is_high_risk=True, applicable_articles=[9, 10, 11, 12, 13, 14, 15])
    report = await score_audit(scores, chunks=[], applicability=applicability)
    assert report.risk_tier == RiskTier.HIGH


@pytest.mark.asyncio
async def test_phase3_risk_tier_minimal_when_not_high_risk():
    """applicability.is_high_risk=False, is_prohibited=False → risk_tier=MINIMAL."""
    from services.compliance_scorer import score_audit
    from models.schemas import ArticleDomain, ArticleScore, RiskTier

    scores = [ArticleScore(article_num=9, domain=ArticleDomain.RISK_MANAGEMENT, score=100.0)]
    applicability = _make_applicability(is_high_risk=False, applicable_articles=[])
    report = await score_audit(scores, chunks=[], applicability=applicability)
    assert report.risk_tier == RiskTier.MINIMAL


@pytest.mark.asyncio
async def test_phase3_gdpr_articles_do_not_affect_eu_aiact_score():
    """gdpr_applicable_articles on ApplicabilityResult must not change overall_score."""
    from services.compliance_scorer import score_audit
    from models.schemas import ArticleDomain, ArticleScore

    scores = [
        ArticleScore(article_num=9,  domain=ArticleDomain.RISK_MANAGEMENT, score=80.0),
        ArticleScore(article_num=10, domain=ArticleDomain.DATA_GOVERNANCE, score=60.0),
    ]
    # Same applicable_articles, different gdpr_applicable_articles
    applicability_no_gdpr  = _make_applicability(applicable_articles=[9, 10], gdpr_applicable_articles=[])
    applicability_with_gdpr = _make_applicability(applicable_articles=[9, 10], gdpr_applicable_articles=[5, 6, 24, 25, 30])

    report_no_gdpr   = await score_audit(scores, chunks=[], applicability=applicability_no_gdpr)
    report_with_gdpr = await score_audit(scores, chunks=[], applicability=applicability_with_gdpr)

    assert report_no_gdpr.overall_score == report_with_gdpr.overall_score


@pytest.mark.asyncio
async def test_phase3_confidence_score_with_all_signals():
    """confidence_score is mean of actor confidence, evidence coverage, chunk ratio."""
    from services.compliance_scorer import score_audit
    from models.schemas import (
        ActorClassification, ActorType, ArticleDomain, ArticleScore,
        DocumentChunk, EvidenceMap,
    )

    chunks = [
        DocumentChunk(chunk_id=f"c{i}", text="text", source_file="f.txt",
                      chunk_index=i, domain=ArticleDomain.RISK_MANAGEMENT)
        for i in range(4)
    ]
    actor = ActorClassification(
        actor_type=ActorType.PROVIDER,
        confidence=0.90,
        matched_signals=["provider keyword"],
        reasoning="test",
    )
    evidence_map = EvidenceMap(
        total_obligations=4,
        fully_satisfied=3,
        partially_satisfied=1,
        missing=0,
        overall_coverage=0.80,
        items=[],
    )
    scores = [ArticleScore(article_num=9, domain=ArticleDomain.RISK_MANAGEMENT, score=80.0)]
    applicability = _make_applicability(applicable_articles=[9])

    report = await score_audit(
        scores, chunks=chunks,
        actor=actor, applicability=applicability, evidence_map=evidence_map,
    )
    # actor=0.90, evidence=0.5+0.80/2=0.90, classified/total=4/4=1.0 → mean=0.933
    assert 0.0 <= report.confidence_score <= 1.0
    assert abs(report.confidence_score - round((0.90 + 0.90 + 1.0) / 3, 2)) < 0.02


@pytest.mark.asyncio
async def test_phase3_requires_human_review_when_low_confidence():
    """requires_human_review=True when confidence < 0.70."""
    from services.compliance_scorer import score_audit
    from models.schemas import ActorClassification, ActorType, ArticleDomain, ArticleScore

    actor = ActorClassification(
        actor_type=ActorType.DEPLOYER,
        confidence=0.30,  # very low
        matched_signals=[],
        reasoning="test",
    )
    scores = [ArticleScore(article_num=9, domain=ArticleDomain.RISK_MANAGEMENT, score=50.0)]
    report = await score_audit(scores, chunks=[], actor=actor)
    assert report.requires_human_review is True


@pytest.mark.asyncio
async def test_phase3_requires_human_review_when_actor_unknown():
    """requires_human_review=True whenever actor_type=UNKNOWN regardless of confidence."""
    from services.compliance_scorer import score_audit
    from models.schemas import ActorClassification, ActorType, ArticleDomain, ArticleScore

    actor = ActorClassification(
        actor_type=ActorType.UNKNOWN,
        confidence=0.95,
        matched_signals=[],
        reasoning="test",
    )
    scores = [ArticleScore(article_num=9, domain=ArticleDomain.RISK_MANAGEMENT, score=90.0)]
    report = await score_audit(scores, chunks=[], actor=actor)
    assert report.requires_human_review is True


@pytest.mark.asyncio
async def test_phase3_applicability_stored_in_report():
    """ComplianceReport.applicability field mirrors what was passed to score_audit."""
    from services.compliance_scorer import score_audit
    from models.schemas import ArticleDomain, ArticleScore

    applicability = _make_applicability(
        is_high_risk=True,
        applicable_articles=[9, 10],
        gdpr_applicable_articles=[5, 6],
    )
    scores = [ArticleScore(article_num=9, domain=ArticleDomain.RISK_MANAGEMENT, score=70.0)]
    report = await score_audit(scores, chunks=[], applicability=applicability)

    assert report.applicability is not None
    assert report.applicability.is_high_risk is True
    assert report.applicability.gdpr_applicable_articles == [5, 6]
