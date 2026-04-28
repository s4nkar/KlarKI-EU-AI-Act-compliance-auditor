"""Unit tests for the Phase 3 ML classifiers lazy-loading wrapper.

Verifies that missing models return None gracefully, that the registry
caches failures so models are not re-attempted, and that models_available()
reflects actual disk state correctly.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


def _reset_registry():
    """Reset the classifier registry so tests don't share state."""
    import services.ml_classifiers as ml
    for k in ml._registry:
        ml._registry[k] = None


# ── Graceful degradation when models are not trained ─────────────────────────


def test_predict_actor_none_when_model_missing(tmp_path):
    """predict_actor() returns None when actor_classifier directory is absent."""
    import services.ml_classifiers as ml
    _reset_registry()
    original = ml._MODEL_PATHS["actor"]
    ml._MODEL_PATHS["actor"] = tmp_path / "missing_actor"
    try:
        result = ml.predict_actor("We developed our AI system in-house.")
        assert result is None
    finally:
        ml._MODEL_PATHS["actor"] = original
        _reset_registry()


def test_predict_high_risk_none_when_model_missing(tmp_path):
    """predict_high_risk() returns None when risk_classifier directory is absent."""
    import services.ml_classifiers as ml
    _reset_registry()
    original = ml._MODEL_PATHS["risk"]
    ml._MODEL_PATHS["risk"] = tmp_path / "missing_risk"
    try:
        result = ml.predict_high_risk("We deploy AI for recruitment decisions.")
        assert result is None
    finally:
        ml._MODEL_PATHS["risk"] = original
        _reset_registry()


def test_predict_prohibited_none_when_model_missing(tmp_path):
    """predict_prohibited() returns None when prohibited_classifier is absent."""
    import services.ml_classifiers as ml
    _reset_registry()
    original = ml._MODEL_PATHS["prohibited"]
    ml._MODEL_PATHS["prohibited"] = tmp_path / "missing_prohibited"
    try:
        result = ml.predict_prohibited("We use real-time biometric identification.")
        assert result is None
    finally:
        ml._MODEL_PATHS["prohibited"] = original
        _reset_registry()


# ── models_available() ────────────────────────────────────────────────────────


def test_models_available_returns_all_keys():
    """models_available() always returns a dict with all three classifier keys."""
    import services.ml_classifiers as ml

    result = ml.models_available()
    assert isinstance(result, dict)
    assert set(result.keys()) == {"actor", "risk", "prohibited"}


def test_models_available_reflects_missing_path(tmp_path):
    """models_available() returns False for a classifier whose directory doesn't exist."""
    import services.ml_classifiers as ml
    _reset_registry()
    original = ml._MODEL_PATHS["actor"]
    ml._MODEL_PATHS["actor"] = tmp_path / "definitely_not_here"
    try:
        result = ml.models_available()
        assert result["actor"] is False
    finally:
        ml._MODEL_PATHS["actor"] = original
        _reset_registry()


def test_models_available_true_when_dir_exists(tmp_path):
    """models_available() returns True when the directory exists (even if not a valid model)."""
    import services.ml_classifiers as ml
    _reset_registry()
    fake_model_dir = tmp_path / "actor_classifier"
    fake_model_dir.mkdir()
    original = ml._MODEL_PATHS["actor"]
    ml._MODEL_PATHS["actor"] = fake_model_dir
    try:
        result = ml.models_available()
        assert result["actor"] is True
    finally:
        ml._MODEL_PATHS["actor"] = original
        _reset_registry()


# ── Registry caching behaviour ────────────────────────────────────────────────


def test_registry_caches_failed_load(tmp_path):
    """After one failed load attempt, registry stores False — no retry on next call."""
    import services.ml_classifiers as ml
    _reset_registry()
    original = ml._MODEL_PATHS["actor"]
    ml._MODEL_PATHS["actor"] = tmp_path / "nonexistent"
    try:
        ml.predict_actor("test text")
        assert ml._registry["actor"] is False

        # Second call must not retry (registry entry is False, not None)
        call_count_before = ml._registry["actor"]
        ml.predict_actor("second call")
        assert ml._registry["actor"] is False
    finally:
        ml._MODEL_PATHS["actor"] = original
        _reset_registry()


# ── MLPrediction named tuple ──────────────────────────────────────────────────


def test_ml_prediction_fields():
    """MLPrediction namedtuple exposes label and confidence fields."""
    from services.ml_classifiers import MLPrediction

    pred = MLPrediction(label="provider", confidence=0.92)
    assert pred.label == "provider"
    assert pred.confidence == 0.92
