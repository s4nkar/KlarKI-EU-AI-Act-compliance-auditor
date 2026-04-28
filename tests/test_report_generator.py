"""Tests for the PDF report generator service.

All tests mock WeasyPrint and Jinja2 — no browser engine required.
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


def _make_report(**overrides):
    """Build a minimal valid ComplianceReport for testing."""
    from models.schemas import (
        ArticleDomain, ArticleScore, ComplianceReport,
        EmotionFlag, RiskTier,
    )

    defaults = dict(
        audit_id="test-audit-123",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source_files=["policy.pdf"],
        language="en",
        risk_tier=RiskTier.HIGH,
        overall_score=72.5,
        article_scores=[
            ArticleScore(
                article_num=9,
                domain=ArticleDomain.RISK_MANAGEMENT,
                score=72.5,
                chunk_count=3,
            )
        ],
        emotion_flag=EmotionFlag(),
        total_chunks=10,
        classified_chunks=8,
        classifier_backend="ollama/phi3:mini",
    )
    defaults.update(overrides)
    return ComplianceReport(**defaults)


def _make_phase3_report():
    """Build a ComplianceReport that includes all Phase 3 fields."""
    from models.schemas import (
        ActorClassification, ActorType,
        AnnexIIICategory, AnnexIIIMatch,
        ApplicabilityResult,
        ArticleDomain, ArticleScore, ComplianceReport,
        EmotionFlag, EvidenceItem, EvidenceMap, RiskTier,
    )
    actor = ActorClassification(
        actor_type=ActorType.PROVIDER,
        confidence=0.88,
        matched_signals=["develops and places on market"],
        reasoning="Provider signals found.",
    )
    applicability = ApplicabilityResult(
        is_high_risk=True,
        is_prohibited=False,
        annex_iii_matches=[
            AnnexIIIMatch(
                category=AnnexIIICategory.EMPLOYMENT,
                category_name="Employment and Workers Management",
                matched_keywords=["recruitment", "CV screening"],
                obligation_id="AIACT_ART6_ANNEXIII_CAT4",
                evidence_required=["technical documentation", "bias audit"],
            )
        ],
        annex_i_triggered=False,
        applicable_articles=[9, 10, 11, 12, 13, 14, 15],
        gdpr_applicable_articles=[5, 6, 24, 25, 30],
        reasoning="Annex III Employment category matched.",
    )
    evidence_map = EvidenceMap(
        total_obligations=5,
        fully_satisfied=2,
        partially_satisfied=2,
        missing=1,
        overall_coverage=0.70,
        items=[
            EvidenceItem(
                obligation_id="AIACT_ART9_001",
                regulation="eu_ai_act",
                article="Article 9",
                requirement="Establish a risk management system.",
                evidence_required=["risk management system"],
                satisfied_by_chunks=["c1"],
                satisfied_evidence=["risk management system"],
                missing_evidence=[],
                coverage=1.0,
            ),
            EvidenceItem(
                obligation_id="GDPR_ART5_001",
                regulation="gdpr",
                article="Article 5",
                requirement="Process personal data lawfully.",
                evidence_required=["data governance documentation", "transparency notice"],
                satisfied_by_chunks=[],
                satisfied_evidence=[],
                missing_evidence=["data governance documentation", "transparency notice"],
                coverage=0.0,
            ),
        ],
    )
    return ComplianceReport(
        audit_id="phase3-audit-456",
        created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        source_files=["policy.pdf"],
        language="en",
        risk_tier=RiskTier.HIGH,
        overall_score=68.0,
        article_scores=[
            ArticleScore(article_num=9, domain=ArticleDomain.RISK_MANAGEMENT, score=68.0, chunk_count=5)
        ],
        emotion_flag=EmotionFlag(),
        total_chunks=12,
        classified_chunks=10,
        classifier_backend="ollama/phi3:mini",
        actor=actor,
        applicability=applicability,
        evidence_map=evidence_map,
        confidence_score=0.85,
        requires_human_review=False,
    )


# ── _render_pdf (blocking render) ────────────────────────────────────────────

def test_render_pdf_returns_bytes():
    """_render_pdf returns bytes when WeasyPrint succeeds."""
    from services.report_generator import _render_pdf

    fake_pdf_bytes = b"%PDF-1.4 fake content"

    mock_html_instance = MagicMock()
    mock_html_instance.write_pdf.return_value = fake_pdf_bytes

    with patch("services.report_generator.Environment") as mock_env_cls:
        mock_template = MagicMock()
        mock_template.render.return_value = "<html><body>Report</body></html>"
        mock_env_cls.return_value.get_template.return_value = mock_template

        # HTML is imported locally inside _render_pdf, so patch via weasyprint
        with patch("weasyprint.HTML", return_value=mock_html_instance):
            result = _render_pdf(_make_report())

    assert isinstance(result, bytes)
    assert result == fake_pdf_bytes


def test_render_pdf_passes_report_to_template():
    """_render_pdf calls template.render(report=...) with the ComplianceReport."""
    from services.report_generator import _render_pdf

    report = _make_report()
    mock_html_instance = MagicMock()
    mock_html_instance.write_pdf.return_value = b"%PDF"

    with patch("services.report_generator.Environment") as mock_env_cls:
        mock_template = MagicMock()
        mock_template.render.return_value = "<html/>"
        mock_env_cls.return_value.get_template.return_value = mock_template

        with patch("weasyprint.HTML", return_value=mock_html_instance):
            _render_pdf(report)

    mock_template.render.assert_called_once_with(report=report)


def test_render_pdf_loads_report_html_template():
    """_render_pdf requests 'report.html' from the template environment."""
    from services.report_generator import _render_pdf

    mock_html_instance = MagicMock()
    mock_html_instance.write_pdf.return_value = b"%PDF"

    with patch("services.report_generator.Environment") as mock_env_cls:
        mock_env = mock_env_cls.return_value
        mock_template = MagicMock()
        mock_template.render.return_value = "<html/>"
        mock_env.get_template.return_value = mock_template

        with patch("weasyprint.HTML", return_value=mock_html_instance):
            _render_pdf(_make_report())

    mock_env.get_template.assert_called_once_with("report.html")


# ── generate_pdf (async wrapper) ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_pdf_returns_bytes():
    """generate_pdf (async) delegates to _render_pdf and returns bytes."""
    from services.report_generator import generate_pdf

    fake_bytes = b"%PDF-1.4 async content"

    with patch("services.report_generator._render_pdf", return_value=fake_bytes):
        result = await generate_pdf(_make_report())

    assert result == fake_bytes


@pytest.mark.asyncio
async def test_generate_pdf_propagates_render_error():
    """generate_pdf lets WeasyPrint errors propagate (no silent swallow)."""
    from services.report_generator import generate_pdf

    with patch("services.report_generator._render_pdf", side_effect=RuntimeError("WeasyPrint error")):
        with pytest.raises(RuntimeError, match="WeasyPrint error"):
            await generate_pdf(_make_report())


# ── Phase 3 fields in template ────────────────────────────────────────────────

def test_render_pdf_with_phase3_fields_passes_report_to_template():
    """_render_pdf with Phase 3 report still calls template.render(report=...) correctly."""
    from services.report_generator import _render_pdf

    report = _make_phase3_report()
    mock_html_instance = MagicMock()
    mock_html_instance.write_pdf.return_value = b"%PDF-phase3"

    with patch("services.report_generator.Environment") as mock_env_cls:
        mock_template = MagicMock()
        mock_template.render.return_value = "<html/>"
        mock_env_cls.return_value.get_template.return_value = mock_template

        with patch("weasyprint.HTML", return_value=mock_html_instance):
            result = _render_pdf(report)

    assert result == b"%PDF-phase3"
    mock_template.render.assert_called_once_with(report=report)


def test_phase3_report_actor_field_present():
    """ComplianceReport built with Phase 3 fields has non-null actor."""
    report = _make_phase3_report()
    assert report.actor is not None
    assert report.actor.actor_type.value == "provider"
    assert 0.0 <= report.actor.confidence <= 1.0


def test_phase3_report_applicability_has_gdpr_articles():
    """ComplianceReport.applicability carries gdpr_applicable_articles."""
    report = _make_phase3_report()
    assert report.applicability is not None
    assert report.applicability.gdpr_applicable_articles == [5, 6, 24, 25, 30]
    assert report.applicability.applicable_articles == [9, 10, 11, 12, 13, 14, 15]


def test_phase3_report_evidence_map_has_both_regulations():
    """EvidenceMap items include entries with regulation='eu_ai_act' and 'gdpr'."""
    report = _make_phase3_report()
    assert report.evidence_map is not None
    regulations = {item.regulation for item in report.evidence_map.items}
    assert "eu_ai_act" in regulations
    assert "gdpr" in regulations


def test_phase3_report_confidence_score_in_range():
    """Phase 3 ComplianceReport.confidence_score is within [0, 1]."""
    report = _make_phase3_report()
    assert 0.0 <= report.confidence_score <= 1.0


def test_phase3_report_serialises_to_dict():
    """Phase 3 ComplianceReport.model_dump() succeeds without raising."""
    report = _make_phase3_report()
    data = report.model_dump()
    assert data["applicability"]["gdpr_applicable_articles"] == [5, 6, 24, 25, 30]
    assert data["evidence_map"]["items"][0]["regulation"] == "eu_ai_act"
    assert data["evidence_map"]["items"][1]["regulation"] == "gdpr"
