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
