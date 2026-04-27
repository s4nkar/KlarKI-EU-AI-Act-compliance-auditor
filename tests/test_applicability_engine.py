"""Unit tests for Article 6 + Annex III applicability engine.

All ML calls are patched to return None so tests run deterministically
without requiring trained specialist models.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


def _no_ml(_text):
    return None


def _chunk(text: str, idx: int = 0):
    from models.schemas import DocumentChunk
    return DocumentChunk(
        chunk_id=f"chunk_{idx}", text=text, source_file="test.txt", chunk_index=idx
    )


def _run(text: str) -> object:
    """Helper: run applicability check on a single-chunk document."""
    from services.applicability_engine import check_applicability

    with patch("services.applicability_engine._ml_prohibited", _no_ml), \
         patch("services.applicability_engine._ml_high_risk", _no_ml):
        return check_applicability([_chunk(text)])


# ── Article 5 prohibited practice detection ───────────────────────────────────


def test_social_scoring_is_prohibited():
    """Social scoring keyword → is_prohibited=True, applicable_articles=[5]."""
    result = _run(
        "This system performs social scoring of citizens based on their public behaviour "
        "and assigns a trustworthiness score used in public-sector decisions."
    )
    assert result.is_prohibited is True
    assert result.is_high_risk is False
    assert result.applicable_articles == [5]


def test_realtime_biometric_identification_is_prohibited():
    """Real-time biometric identification in public spaces → is_prohibited=True."""
    result = _run(
        "The system performs real-time biometric identification of pedestrians "
        "using CCTV cameras installed in public squares."
    )
    assert result.is_prohibited is True
    assert 5 in result.applicable_articles


def test_subliminal_techniques_is_prohibited():
    """Subliminal technique mention → is_prohibited=True."""
    result = _run(
        "The AI uses subliminal techniques to alter purchasing behaviour "
        "below the threshold of conscious awareness."
    )
    assert result.is_prohibited is True


def test_emotion_recognition_in_workplace_is_prohibited():
    """Emotion recognition targeting workplace employees → is_prohibited=True."""
    result = _run(
        "We deploy an emotion recognition system to monitor employee stress levels "
        "and productivity at their workstations."
    )
    assert result.is_prohibited is True


def test_emotion_recognition_in_education_is_prohibited():
    """Emotion recognition in school context → is_prohibited=True."""
    result = _run(
        "The classroom AI uses emotion recognition to track student engagement "
        "and attention levels during lessons at our school."
    )
    assert result.is_prohibited is True


def test_german_social_scoring_is_prohibited():
    """German 'Social-Scoring' keyword → is_prohibited=True."""
    result = _run(
        "Das System führt Social-Scoring von Bürgerinnen und Bürgern durch "
        "und bewertet ihr Verhalten im öffentlichen Raum."
    )
    assert result.is_prohibited is True


def test_german_echtzeitbiometrie_is_prohibited():
    """German 'Echtzeitbiometrie' keyword → is_prohibited=True."""
    result = _run(
        "Durch Echtzeitbiometrie können Personen in öffentlichen Bereichen "
        "live identifiziert werden."
    )
    assert result.is_prohibited is True


def test_prohibited_takes_priority_over_high_risk():
    """When both prohibited and high-risk signals are present, prohibited wins."""
    result = _run(
        "We perform social scoring of our recruits and also conduct CV screening "
        "for hiring decisions using AI."
    )
    assert result.is_prohibited is True
    assert result.is_high_risk is False


# ── Annex III high-risk category matching ─────────────────────────────────────


def test_employment_category_detected():
    """Recruitment / CV screening → HIGH RISK, EMPLOYMENT category."""
    from models.schemas import AnnexIIICategory

    result = _run(
        "Our AI system performs CV screening and candidate ranking for all hiring decisions. "
        "The algorithm evaluates resumes and shortlists applicants automatically."
    )
    assert result.is_high_risk is True
    assert result.is_prohibited is False
    categories = [m.category for m in result.annex_iii_matches]
    assert AnnexIIICategory.EMPLOYMENT in categories


def test_education_category_detected():
    """Student admission / academic assessment AI → HIGH RISK, EDUCATION."""
    from models.schemas import AnnexIIICategory

    result = _run(
        "The platform performs student admission decisions for university entry "
        "and evaluates learning outcomes through AI-based educational assessment."
    )
    assert result.is_high_risk is True
    categories = [m.category for m in result.annex_iii_matches]
    assert AnnexIIICategory.EDUCATION in categories


def test_essential_services_credit_scoring():
    """Credit scoring / creditworthiness assessment → HIGH RISK, ESSENTIAL_SERVICES."""
    from models.schemas import AnnexIIICategory

    result = _run(
        "We use AI to perform credit scoring and creditworthiness assessment. "
        "The algorithm determines loan eligibility and interest rates automatically."
    )
    assert result.is_high_risk is True
    categories = [m.category for m in result.annex_iii_matches]
    assert AnnexIIICategory.ESSENTIAL_SERVICES in categories


def test_law_enforcement_category_detected():
    """Predictive policing AI → HIGH RISK, LAW_ENFORCEMENT."""
    from models.schemas import AnnexIIICategory

    result = _run(
        "This law enforcement AI tool performs predictive policing by analysing "
        "historical crime data and generating risk scores for patrol allocation."
    )
    assert result.is_high_risk is True
    categories = [m.category for m in result.annex_iii_matches]
    assert AnnexIIICategory.LAW_ENFORCEMENT in categories


def test_biometric_categorisation_category():
    """Facial recognition → HIGH RISK, BIOMETRIC."""
    from models.schemas import AnnexIIICategory

    result = _run(
        "The system uses facial recognition and biometric identification "
        "to grant physical access to secure facilities."
    )
    assert result.is_high_risk is True
    categories = [m.category for m in result.annex_iii_matches]
    assert AnnexIIICategory.BIOMETRIC in categories


def test_migration_category_detected():
    """Visa / asylum application processing AI → HIGH RISK, MIGRATION."""
    from models.schemas import AnnexIIICategory

    result = _run(
        "The system assists in asylum application assessment and visa processing decisions. "
        "It evaluates refugee claims using AI-based document analysis."
    )
    assert result.is_high_risk is True
    categories = [m.category for m in result.annex_iii_matches]
    assert AnnexIIICategory.MIGRATION in categories


def test_applicable_articles_set_for_high_risk():
    """High-risk → applicable_articles = [9, 10, 11, 12, 13, 14, 15]."""
    result = _run(
        "Our AI system performs hiring decisions and candidate scoring "
        "for all open positions in our organisation."
    )
    assert sorted(result.applicable_articles) == [9, 10, 11, 12, 13, 14, 15]


def test_annex_iii_match_includes_keywords():
    """Each AnnexIIIMatch must list the matched keywords."""
    result = _run(
        "The system performs CV screening and candidate ranking for job applications."
    )
    assert len(result.annex_iii_matches) > 0
    for match in result.annex_iii_matches:
        assert len(match.matched_keywords) > 0


# ── Article 6(1) Annex I safety-component check ───────────────────────────────


def test_two_annex_i_signals_triggers():
    """Two Annex I signals (CE marking + notified body) → annex_i_triggered=True."""
    result = _run(
        "The system requires CE marking under the Medical Device Regulation "
        "and has been assessed by a notified body for conformity."
    )
    assert result.annex_i_triggered is True
    assert result.is_high_risk is True


def test_single_annex_i_signal_does_not_trigger():
    """A single CE marking signal alone does not trigger annex_i_triggered."""
    result = _run(
        "The product carries CE marking for general consumer use but is not "
        "subject to any specific sectoral safety regulation."
    )
    assert result.annex_i_triggered is False


# ── Minimal-risk systems ──────────────────────────────────────────────────────


def test_customer_chatbot_is_minimal():
    """Generic FAQ chatbot → minimal risk, no applicable articles."""
    result = _run(
        "Our chatbot helps customers find products, answer billing questions, "
        "and provides general customer service support."
    )
    assert result.is_high_risk is False
    assert result.is_prohibited is False
    assert result.applicable_articles == []
    assert result.annex_iii_matches == []


def test_invoice_processing_is_minimal():
    """Invoice OCR / extraction → minimal risk."""
    result = _run(
        "This AI system reads supplier invoices and automatically extracts "
        "payment amounts, due dates, and vendor details into our ERP system."
    )
    assert result.is_high_risk is False
    assert result.is_prohibited is False


def test_translation_tool_is_minimal():
    """AI translation service → minimal risk."""
    result = _run(
        "The translation AI converts documents from English to German for "
        "internal communication purposes. No decisions are automated."
    )
    assert result.is_high_risk is False
    assert result.is_prohibited is False
    assert result.applicable_articles == []


def test_result_always_has_reasoning():
    """ApplicabilityResult always includes a non-empty reasoning string."""
    result = _run("Our AI assists with product recommendations on our e-commerce website.")
    assert isinstance(result.reasoning, str)
    assert len(result.reasoning) > 10
