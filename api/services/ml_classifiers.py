"""Phase 3 ML inference wrapper for specialist classifiers.

Provides lazy-loaded inference for three trained GBERT models:
  - actor      → training/actor_classifier/
  - risk       → training/risk_classifier/
  - prohibited → training/prohibited_classifier/

Each model is loaded once on first call and cached. If a model directory
does not exist (not yet trained), the function returns None — callers fall
back to the pattern-based approach automatically.

Inference is synchronous (blocking). Callers should wrap in asyncio.to_thread.

Usage:
    result = predict_actor("We developed our AI system in-house.")
    # Returns {"label": "provider", "confidence": 0.97} or None
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import NamedTuple

import structlog

logger = structlog.get_logger()

_ROOT = Path(__file__).parent.parent.parent

_MODEL_PATHS = {
    "actor":      _ROOT / "training" / "actor_classifier",
    "risk":       _ROOT / "training" / "risk_classifier",
    "prohibited": _ROOT / "training" / "prohibited_classifier",
}


class MLPrediction(NamedTuple):
    label: str
    confidence: float


# ── Lazy model registry ───────────────────────────────────────────────────────
# Each entry: None = not yet attempted; False = failed/unavailable; dict = loaded
_registry: dict[str, object] = {k: None for k in _MODEL_PATHS}


def _load_model(classifier_type: str) -> dict | None:
    """Load tokenizer + model for the given classifier type. Cached."""
    entry = _registry[classifier_type]
    if entry is False:
        return None
    if entry is not None:
        return entry  # type: ignore[return-value]

    model_path = _MODEL_PATHS[classifier_type]
    if not model_path.exists():
        logger.info(
            "ml_classifier_not_found",
            classifier=classifier_type,
            path=str(model_path),
            action="falling_back_to_patterns",
        )
        _registry[classifier_type] = False
        return None

    try:
        import torch
        from transformers import BertForSequenceClassification, BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(str(model_path))
        model = BertForSequenceClassification.from_pretrained(str(model_path))
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        entry = {
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
            "id2label": model.config.id2label,
        }
        _registry[classifier_type] = entry
        logger.info("ml_classifier_loaded", classifier=classifier_type, device=device)
        return entry  # type: ignore[return-value]

    except Exception as exc:
        logger.warning("ml_classifier_load_failed", classifier=classifier_type, error=str(exc))
        _registry[classifier_type] = False
        return None


def _predict(classifier_type: str, text: str, max_length: int = 256) -> MLPrediction | None:
    """Run inference for one text sample. Returns None if model unavailable."""
    entry = _load_model(classifier_type)
    if entry is None:
        return None

    try:
        import torch
        import torch.nn.functional as F

        tokenizer = entry["tokenizer"]  # type: ignore[index]
        model     = entry["model"]      # type: ignore[index]
        device    = entry["device"]     # type: ignore[index]
        id2label  = entry["id2label"]   # type: ignore[index]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = F.softmax(logits, dim=-1)[0]

        best_id   = int(probs.argmax().item())
        confidence = float(probs[best_id].item())
        label      = id2label[best_id]

        return MLPrediction(label=label, confidence=round(confidence, 4))

    except Exception as exc:
        logger.warning("ml_inference_failed", classifier=classifier_type, error=str(exc))
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def predict_actor(text: str) -> MLPrediction | None:
    """Predict Article 3 actor role: provider / deployer / importer / distributor.

    Returns None if the actor classifier model has not been trained yet.
    """
    return _predict("actor", text)


def predict_high_risk(text: str) -> MLPrediction | None:
    """Predict whether text describes a high-risk AI system (Article 6 + Annex III).

    Labels: high_risk / not_high_risk.
    Returns None if the risk classifier model has not been trained yet.
    """
    return _predict("risk", text)


def predict_prohibited(text: str) -> MLPrediction | None:
    """Predict whether text describes a prohibited AI practice (Article 5).

    Labels: prohibited / not_prohibited.
    Returns None if the prohibited classifier model has not been trained yet.
    """
    return _predict("prohibited", text)


def models_available() -> dict[str, bool]:
    """Return which specialist models are trained and loadable."""
    return {k: _MODEL_PATHS[k].exists() for k in _MODEL_PATHS}
