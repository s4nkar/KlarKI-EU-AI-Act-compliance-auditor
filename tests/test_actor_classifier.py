"""Unit tests for Article 3 actor role classifier.

Covers pattern-based classification (no ML) and the ML ensemble path.
All tests patch the ML predict function so they run offline without
requiring trained model artifacts.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


def _no_ml(_text):
    """Simulate ML model not available — always returns None."""
    return None


# ── Pattern-based classification ─────────────────────────────────────────────


def test_provider_developed_in_house():
    """'we developed' trigger → PROVIDER."""
    from models.schemas import ActorType
    from services.actor_classifier import classify_actor

    with patch("services.actor_classifier._ml_predict_actor", _no_ml):
        result = classify_actor(
            "We developed and trained our AI model entirely in-house "
            "using proprietary datasets collected over three years."
        )
    assert result.actor_type == ActorType.PROVIDER


def test_provider_placed_on_market():
    """'placed on the market' trigger → PROVIDER."""
    from models.schemas import ActorType
    from services.actor_classifier import classify_actor

    with patch("services.actor_classifier._ml_predict_actor", _no_ml):
        result = classify_actor(
            "TechCorp has placed this AI system on the market under its own brand name "
            "and maintains full intellectual property rights."
        )
    assert result.actor_type == ActorType.PROVIDER


def test_provider_proprietary_model():
    """'proprietary AI model' trigger → PROVIDER."""
    from models.schemas import ActorType
    from services.actor_classifier import classify_actor

    with patch("services.actor_classifier._ml_predict_actor", _no_ml):
        result = classify_actor(
            "Our proprietary AI model was designed by our research division. "
            "We offer it as a SaaS solution to enterprise customers."
        )
    assert result.actor_type == ActorType.PROVIDER


def test_deployer_third_party_ai():
    """'third-party AI system' trigger → DEPLOYER."""
    from models.schemas import ActorType
    from services.actor_classifier import classify_actor

    with patch("services.actor_classifier._ml_predict_actor", _no_ml):
        result = classify_actor(
            "We implemented a third-party AI system from Vendor X to screen "
            "job applications. The vendor retains all IP rights."
        )
    assert result.actor_type == ActorType.DEPLOYER


def test_deployer_licensed_software():
    """'licensed AI software' trigger → DEPLOYER."""
    from models.schemas import ActorType
    from services.actor_classifier import classify_actor

    with patch("services.actor_classifier._ml_predict_actor", _no_ml):
        result = classify_actor(
            "Our hospital deployed licensed AI software from MedTech Solutions "
            "to assist radiologists in interpreting scan images."
        )
    assert result.actor_type == ActorType.DEPLOYER


def test_deployer_vendor_system():
    """'vendor's AI system' trigger → DEPLOYER."""
    from models.schemas import ActorType
    from services.actor_classifier import classify_actor

    with patch("services.actor_classifier._ml_predict_actor", _no_ml):
        result = classify_actor(
            "We operate a vendor's AI system for fraud detection that was integrated "
            "into our banking infrastructure by an external consultancy."
        )
    assert result.actor_type == ActorType.DEPLOYER


def test_importer_outside_eu():
    """'established outside the European Union' + 'importer' → IMPORTER."""
    from models.schemas import ActorType
    from services.actor_classifier import classify_actor

    with patch("services.actor_classifier._ml_predict_actor", _no_ml):
        result = classify_actor(
            "We import this AI-powered security screening system from a manufacturer "
            "established outside the European Union. As importer, we verify EU compliance."
        )
    assert result.actor_type == ActorType.IMPORTER


def test_distributor_resell():
    """'we resell' + 'distribute' → DISTRIBUTOR."""
    from models.schemas import ActorType
    from services.actor_classifier import classify_actor

    with patch("services.actor_classifier._ml_predict_actor", _no_ml):
        result = classify_actor(
            "We distribute and resell the SmartHire AI platform to SMEs across Germany. "
            "We do not modify the underlying AI model or training data."
        )
    assert result.actor_type == ActorType.DISTRIBUTOR


def test_german_provider_pattern():
    """German 'wir haben entwickelt' + 'Anbieter' → PROVIDER."""
    from models.schemas import ActorType
    from services.actor_classifier import classify_actor

    with patch("services.actor_classifier._ml_predict_actor", _no_ml):
        result = classify_actor(
            "Wir haben unser KI-System selbst entwickelt und bringen es als Anbieter "
            "auf den europäischen Markt. Unser proprietäres Modell wurde intern trainiert."
        )
    assert result.actor_type == ActorType.PROVIDER


def test_german_deployer_pattern():
    """German 'wir nutzen' + 'Betreiber' → DEPLOYER."""
    from models.schemas import ActorType
    from services.actor_classifier import classify_actor

    with patch("services.actor_classifier._ml_predict_actor", _no_ml):
        result = classify_actor(
            "Wir nutzen eine lizenzierte KI-Software eines externen Anbieters. "
            "Als Betreiber sind wir verantwortlich für den ordnungsgemäßen Einsatz."
        )
    assert result.actor_type == ActorType.DEPLOYER


def test_german_einführer_pattern():
    """German 'Einführer' → IMPORTER."""
    from models.schemas import ActorType
    from services.actor_classifier import classify_actor

    with patch("services.actor_classifier._ml_predict_actor", _no_ml):
        result = classify_actor(
            "Als Einführer dieses KI-Systems, das von einem außerhalb der EU "
            "ansässigen Unternehmen entwickelt wurde, tragen wir die Verantwortung "
            "für die Einhaltung europäischer Vorschriften."
        )
    assert result.actor_type == ActorType.IMPORTER


def test_no_signals_defaults_to_deployer():
    """Zero actor signals → defaults to DEPLOYER with confidence < 0.5."""
    from models.schemas import ActorType
    from services.actor_classifier import classify_actor

    with patch("services.actor_classifier._ml_predict_actor", _no_ml):
        result = classify_actor(
            "This document describes the general compliance policy of our organisation "
            "for artificial intelligence governance."
        )
    assert result.actor_type == ActorType.DEPLOYER
    assert result.confidence < 0.5


def test_result_has_required_fields():
    """ActorClassification always has confidence, reasoning, matched_signals."""
    from services.actor_classifier import classify_actor

    with patch("services.actor_classifier._ml_predict_actor", _no_ml):
        result = classify_actor("We deploy a third-party AI solution from our vendor.")
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.reasoning, str) and len(result.reasoning) > 0
    assert isinstance(result.matched_signals, list)


# ── ML ensemble path ──────────────────────────────────────────────────────────


def test_ml_high_confidence_overrides_patterns():
    """ML prediction ≥ 0.80 confidence should be used directly, skipping patterns."""
    from services.actor_classifier import classify_actor
    from services.ml_classifiers import MLPrediction
    from models.schemas import ActorType

    mock_ml = MagicMock(return_value=MLPrediction(label="provider", confidence=0.95))
    with patch("services.actor_classifier._ml_predict_actor", mock_ml):
        result = classify_actor("Some generic compliance text without clear actor signals.")
    assert result.actor_type == ActorType.PROVIDER
    assert result.confidence == 0.95


def test_ml_below_threshold_falls_back_to_patterns():
    """ML prediction below 0.80 threshold falls back to pattern matching."""
    from services.actor_classifier import classify_actor
    from services.ml_classifiers import MLPrediction
    from models.schemas import ActorType

    # ML says provider with low confidence; pattern evidence clearly says deployer
    mock_ml = MagicMock(return_value=MLPrediction(label="provider", confidence=0.55))
    with patch("services.actor_classifier._ml_predict_actor", mock_ml):
        result = classify_actor(
            "We use a third-party AI system licensed from a vendor. "
            "We are the deployer and operator of this external AI system."
        )
    assert result.actor_type == ActorType.DEPLOYER


def test_ml_invalid_label_falls_back_to_unknown():
    """ML returning an invalid label → ActorType.UNKNOWN."""
    from services.actor_classifier import classify_actor
    from services.ml_classifiers import MLPrediction
    from models.schemas import ActorType

    mock_ml = MagicMock(return_value=MLPrediction(label="invalid_label_xyz", confidence=0.99))
    with patch("services.actor_classifier._ml_predict_actor", mock_ml):
        result = classify_actor("Some text.")
    assert result.actor_type == ActorType.UNKNOWN
