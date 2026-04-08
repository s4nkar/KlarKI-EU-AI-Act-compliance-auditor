"""Metrics router — expose BERT classifier and spaCy NER evaluation metrics.

Endpoints:
    GET /api/v1/metrics/classifier  — BERT per-class precision/recall/F1 + confusion matrix
    GET /api/v1/metrics/ner         — spaCy NER per-entity-label precision/recall/F1
    GET /api/v1/metrics/evaluation  — Latest results from evaluation suite (eval_*.py)
"""

import json
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException

from models.schemas import APIResponse

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])

# Paths work for both local dev (repo root) and Docker container (/app → /training, /tests)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TRAINING_ROOT = next(
    (p for p in [_REPO_ROOT / "training", Path("/training")] if p.is_dir()),
    _REPO_ROOT / "training",
)
_BERT_METRICS_PATH = _TRAINING_ROOT / "bert_classifier" / "metrics.json"
_NER_METRICS_PATH  = _TRAINING_ROOT / "spacy_ner_model" / "metrics.json"
_EVAL_RESULTS_DIR  = next(
    (p for p in [_REPO_ROOT / "tests" / "evaluation" / "results", Path("/tests/evaluation/results")] if p.is_dir()),
    Path("/tests/evaluation/results"),
)


def _load_metrics(path: Path, label: str) -> dict:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{label} metrics not found. Run './run.sh setup' first.",
        )
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("metrics_read_failed", path=str(path), error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to read {label} metrics.") from exc


@router.get("/classifier", response_model=APIResponse)
async def get_classifier_metrics() -> APIResponse:
    """Return BERT classifier evaluation metrics from the most recent training run.

    Returns per-class precision, recall, F1 and the full confusion matrix.
    Written by training/train_classifier.py on the held-out validation set (15% split).

    Raises:
        404: If training/bert_classifier/metrics.json is missing.
    """
    data = _load_metrics(_BERT_METRICS_PATH, "BERT classifier")
    logger.info("bert_metrics_served", macro_f1=data.get("macro_f1"))
    return APIResponse(status="success", data=data)


@router.get("/ner", response_model=APIResponse)
async def get_ner_metrics() -> APIResponse:
    """Return spaCy NER evaluation metrics from the most recent training run.

    Returns overall and per-entity-label precision, recall, F1.
    Written by training/train_ner.py on the held-out dev set (20% split).

    Raises:
        404: If training/spacy_ner_model/metrics.json is missing.
    """
    data = _load_metrics(_NER_METRICS_PATH, "NER")
    logger.info("ner_metrics_served", overall_f1=data.get("overall_f1"))
    return APIResponse(status="success", data=data)


@router.get("/evaluation", response_model=APIResponse)
async def get_evaluation_results() -> APIResponse:
    """Return the latest results from the evaluation suite (eval_*.py).

    Reads all *.json files from tests/evaluation/results/ and returns them
    keyed by eval name.  Returns an empty dict (not 404) when no results exist
    so the frontend can display a "not yet run" state gracefully.
    """
    if not _EVAL_RESULTS_DIR.exists():
        return APIResponse(status="success", data={})

    results: dict = {}
    for path in sorted(_EVAL_RESULTS_DIR.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            results[path.stem] = payload
        except Exception as exc:
            logger.warning("eval_result_read_failed", path=str(path), error=str(exc))

    logger.info("eval_results_served", count=len(results))
    return APIResponse(status="success", data=results)
