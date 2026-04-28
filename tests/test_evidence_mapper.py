"""Unit tests for the evidence mapping service.

Tests the deterministic regex-based evidence detection layer.
NLI model is patched out so tests run offline without downloading models.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


def _chunk(text: str, chunk_id: str = "c1"):
    from models.schemas import DocumentChunk
    return DocumentChunk(
        chunk_id=chunk_id, text=text, source_file="test.txt", chunk_index=0
    )


# ── _evidence_present: regex fast path ───────────────────────────────────────


def test_exact_term_match():
    """Exact evidence term found in chunk → chunk_id returned."""
    from services.evidence_mapper import _evidence_present

    chunks = [_chunk("We maintain a risk register that logs all identified system risks.")]
    matched = _evidence_present("risk register", chunks)
    assert "c1" in matched


def test_synonym_risk_log():
    """Synonym 'risk log' → counts as 'risk register'."""
    from services.evidence_mapper import _evidence_present

    chunks = [_chunk("Our risk log is reviewed quarterly by the safety committee.")]
    matched = _evidence_present("risk register", chunks)
    assert "c1" in matched


def test_synonym_risk_inventory():
    """Synonym 'risk inventory' → counts as 'risk register'."""
    from services.evidence_mapper import _evidence_present

    chunks = [_chunk("A formal risk inventory is maintained and updated after each audit cycle.")]
    matched = _evidence_present("risk register", chunks)
    assert "c1" in matched


def test_german_synonym_risikokatalog():
    """German 'Risikokatalog' → matched as 'risk register'."""
    from services.evidence_mapper import _evidence_present

    chunks = [_chunk("Der Risikokatalog wird monatlich aktualisiert und vom Vorstand freigegeben.")]
    matched = _evidence_present("risk register", chunks)
    assert "c1" in matched


def test_technical_file_matches_technical_documentation():
    """'technical file' → matched as 'technical documentation'."""
    from services.evidence_mapper import _evidence_present

    chunks = [_chunk("The technical file contains system architecture diagrams and test reports.")]
    matched = _evidence_present("technical documentation", chunks)
    assert "c1" in matched


def test_human_in_the_loop_matches_oversight():
    """'human-in-the-loop' → matched as 'human oversight procedure'."""
    from services.evidence_mapper import _evidence_present

    chunks = [_chunk("All AI-generated decisions pass through a human-in-the-loop review step.")]
    matched = _evidence_present("human oversight procedure", chunks)
    assert "c1" in matched


def test_audit_log_matches_logging_trail():
    """'audit log' → matched as 'logging and audit trail'."""
    from services.evidence_mapper import _evidence_present

    chunks = [_chunk("The system maintains a detailed audit log of every automated decision.")]
    matched = _evidence_present("logging and audit trail", chunks)
    assert "c1" in matched


def test_bias_testing_matches_bias_audit():
    """'bias testing' → matched as 'bias audit'."""
    from services.evidence_mapper import _evidence_present

    chunks = [_chunk("We conducted bias testing across protected demographic groups before deployment.")]
    matched = _evidence_present("bias audit", chunks)
    assert "c1" in matched


def test_menschliche_aufsicht_matches_oversight():
    """German 'menschliche Aufsicht' → matched as 'human oversight procedure'."""
    from services.evidence_mapper import _evidence_present

    chunks = [_chunk("Eine menschliche Aufsicht ist für alle Entscheidungen mit hohem Risiko vorgesehen.")]
    matched = _evidence_present("human oversight procedure", chunks)
    assert "c1" in matched


def test_no_match_returns_empty_list():
    """Chunk with no relevant content → empty list."""
    from services.evidence_mapper import _evidence_present

    chunks = [_chunk("This document covers our marketing strategy for the upcoming fiscal year.")]
    matched = _evidence_present("risk register", chunks)
    assert matched == []


def test_multiple_chunks_partial_match():
    """Three chunks: two match, one does not → only matching ids returned."""
    from services.evidence_mapper import _evidence_present

    chunks = [
        _chunk("We maintain a comprehensive risk register for all system components.", "c1"),
        _chunk("The marketing team manages our social media channels daily.", "c2"),
        _chunk("A risk log entry is created for every identified vulnerability.", "c3"),
    ]
    matched = _evidence_present("risk register", chunks)
    assert "c1" in matched
    assert "c2" not in matched
    assert "c3" in matched


def test_unknown_term_with_no_nli_returns_empty():
    """Unknown term not in synonym dict, NLI unavailable → no match."""
    from services.evidence_mapper import _evidence_present

    with patch("services.evidence_mapper._get_nli_model", return_value=None):
        chunks = [_chunk("We have a comprehensive compliance programme in place.")]
        matched = _evidence_present("nonexistent_evidence_term_xyz", chunks)
    assert matched == []


def test_case_insensitive_matching():
    """Pattern matching is case-insensitive."""
    from services.evidence_mapper import _evidence_present

    chunks = [_chunk("A RISK REGISTER is maintained according to ISO 31000 guidelines.")]
    matched = _evidence_present("risk register", chunks)
    assert "c1" in matched


# ── map_evidence: full pipeline ───────────────────────────────────────────────


def test_empty_applicable_articles_returns_zero_map():
    """No applicable articles → EvidenceMap with zero obligations."""
    from models.schemas import ActorType
    from services.evidence_mapper import map_evidence

    chunks = [_chunk("We have a risk register and technical documentation.")]
    result = map_evidence(chunks, ActorType.PROVIDER, applicable_articles=[])
    assert result.total_obligations == 0
    assert result.overall_coverage == 0.0
    assert result.items == []
    assert result.fully_satisfied == 0
    assert result.missing == 0


def test_evidence_map_fields_complete():
    """EvidenceMap always exposes all aggregate fields."""
    from models.schemas import ActorType
    from services.evidence_mapper import map_evidence

    chunks = [_chunk("No relevant evidence content here.")]
    result = map_evidence(chunks, ActorType.DEPLOYER, applicable_articles=[])
    assert hasattr(result, "total_obligations")
    assert hasattr(result, "fully_satisfied")
    assert hasattr(result, "partially_satisfied")
    assert hasattr(result, "missing")
    assert hasattr(result, "overall_coverage")
    assert hasattr(result, "items")


def test_evidence_item_coverage_bounded():
    """EvidenceItem coverage is always in [0.0, 1.0]."""
    from models.schemas import ActorType
    from services.evidence_mapper import map_evidence, _load_all_obligations

    obligations = _load_all_obligations()
    if not obligations:
        pytest.skip("No obligation JSONL files found — run ./run.sh setup first.")

    chunks = [
        _chunk("We maintain a risk register, technical documentation, and audit logs.", "c1"),
        _chunk("A conformity assessment was completed by a notified body.", "c2"),
    ]
    result = map_evidence(chunks, ActorType.PROVIDER, applicable_articles=[9, 10, 11, 12])
    for item in result.items:
        assert 0.0 <= item.coverage <= 1.0


def test_aggregate_counts_are_consistent():
    """fully_satisfied + partially_satisfied + missing == total_obligations."""
    from models.schemas import ActorType
    from services.evidence_mapper import map_evidence, _load_all_obligations

    obligations = _load_all_obligations()
    if not obligations:
        pytest.skip("No obligation JSONL files found — run ./run.sh setup first.")

    chunks = [_chunk("We have risk documentation and human oversight procedures.", "c1")]
    result = map_evidence(chunks, ActorType.PROVIDER, applicable_articles=[9, 10, 11, 12, 13, 14, 15])
    assert result.fully_satisfied + result.partially_satisfied + result.missing == result.total_obligations


def test_overall_coverage_bounded():
    """overall_coverage is always in [0.0, 1.0]."""
    from models.schemas import ActorType
    from services.evidence_mapper import map_evidence

    chunks = [_chunk("General compliance framework text without specific evidence artefacts.")]
    result = map_evidence(chunks, ActorType.DEPLOYER, applicable_articles=[9])
    assert 0.0 <= result.overall_coverage <= 1.0


# ── GDPR regulation-aware evidence mapping ────────────────────────────────────


def test_gdpr_obligations_included_when_gdpr_articles_provided():
    """map_evidence returns GDPR items when gdpr_applicable_articles is non-empty."""
    from models.schemas import ActorType
    from services.evidence_mapper import map_evidence, _load_all_obligations

    if not _load_all_obligations():
        pytest.skip("No obligation JSONL files found — run ./run.sh setup first.")

    chunks = [_chunk("We collect personal data and process it lawfully. Data governance documentation is maintained.")]
    result = map_evidence(
        chunks,
        ActorType.DEPLOYER,
        applicable_articles=[],
        gdpr_applicable_articles=[5, 6, 24, 25, 30],
    )
    gdpr_items = [i for i in result.items if i.regulation == "gdpr"]
    assert len(gdpr_items) > 0


def test_no_gdpr_items_when_gdpr_articles_empty():
    """map_evidence returns no GDPR items when gdpr_applicable_articles is empty."""
    from models.schemas import ActorType
    from services.evidence_mapper import map_evidence, _load_all_obligations

    if not _load_all_obligations():
        pytest.skip("No obligation JSONL files found — run ./run.sh setup first.")

    chunks = [_chunk("We have risk documentation and technical files.")]
    result = map_evidence(
        chunks,
        ActorType.PROVIDER,
        applicable_articles=[9, 10],
        gdpr_applicable_articles=[],
    )
    gdpr_items = [i for i in result.items if i.regulation == "gdpr"]
    assert gdpr_items == []


def test_evidence_item_regulation_field_is_set():
    """Every EvidenceItem has a non-empty regulation field."""
    from models.schemas import ActorType
    from services.evidence_mapper import map_evidence, _load_all_obligations

    if not _load_all_obligations():
        pytest.skip("No obligation JSONL files found — run ./run.sh setup first.")

    chunks = [_chunk("Risk register and technical documentation are available.")]
    result = map_evidence(
        chunks,
        ActorType.PROVIDER,
        applicable_articles=[9, 11],
        gdpr_applicable_articles=[5, 6],
    )
    for item in result.items:
        assert item.regulation in ("eu_ai_act", "gdpr"), (
            f"Unexpected regulation value '{item.regulation}' on {item.obligation_id}"
        )


def test_no_article_number_collision_gdpr_vs_aiact_article_5():
    """GDPR Art. 5 obligations must not appear in EU AI Act applicable_articles context.

    A document with only GDPR Art. 5 applicable (no EU AI Act Art. 5 / prohibited
    practices) must return no EU AI Act obligation items for article 5.
    """
    from models.schemas import ActorType
    from services.evidence_mapper import map_evidence, _load_all_obligations

    if not _load_all_obligations():
        pytest.skip("No obligation JSONL files found — run ./run.sh setup first.")

    chunks = [_chunk("We process personal data in accordance with GDPR principles.")]
    # Provide ONLY GDPR articles — no EU AI Act Art. 5 in applicable_articles
    result = map_evidence(
        chunks,
        ActorType.DEPLOYER,
        applicable_articles=[],          # EU AI Act: nothing
        gdpr_applicable_articles=[5, 6], # GDPR: Arts 5 and 6
    )
    # All items should be gdpr, none should be eu_ai_act for article "Article 5"
    eu_act_art5_items = [
        i for i in result.items
        if i.regulation == "eu_ai_act" and i.article == "Article 5"
    ]
    gdpr_art5_items = [
        i for i in result.items
        if i.regulation == "gdpr" and i.article == "Article 5"
    ]
    assert eu_act_art5_items == [], "EU AI Act Art. 5 obligations leaked into GDPR-only result"
    assert len(gdpr_art5_items) > 0, "Expected GDPR Art. 5 items but found none"


def test_both_regulations_appear_in_combined_map():
    """When both EU AI Act and GDPR articles are provided, items from both appear."""
    from models.schemas import ActorType
    from services.evidence_mapper import map_evidence, _load_all_obligations

    if not _load_all_obligations():
        pytest.skip("No obligation JSONL files found — run ./run.sh setup first.")

    chunks = [_chunk("Risk register, technical documentation, data governance documentation.")]
    result = map_evidence(
        chunks,
        ActorType.DEPLOYER,
        applicable_articles=[9, 12, 14],
        gdpr_applicable_articles=[5, 6, 24, 30],
    )
    regulations = {i.regulation for i in result.items}
    assert "eu_ai_act" in regulations
    assert "gdpr" in regulations


def test_empty_both_articles_returns_zero_map():
    """Both applicable_articles=[] and gdpr_applicable_articles=[] → empty map."""
    from models.schemas import ActorType
    from services.evidence_mapper import map_evidence

    chunks = [_chunk("We have all required documentation.")]
    result = map_evidence(
        chunks,
        ActorType.PROVIDER,
        applicable_articles=[],
        gdpr_applicable_articles=[],
    )
    assert result.total_obligations == 0
    assert result.items == []
