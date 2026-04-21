"""Monitoring router — observability dashboard endpoint.

Endpoints:
    GET /api/v1/monitoring  — Full observability snapshot
"""

import asyncio
import json
import time
from pathlib import Path

import structlog
from fastapi import APIRouter, Request

from models.schemas import APIResponse
from services.monitoring_stats import stats as pipeline_stats

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TRAINING_ROOT = next(
    (p for p in [_REPO_ROOT / "training", Path("/training")] if p.is_dir()),
    _REPO_ROOT / "training",
)
_ARTIFACTS_ROOT = _TRAINING_ROOT / "artifacts"
_REGISTRY_PATH  = _ARTIFACTS_ROOT / "registry.json"


def _load_registry() -> dict:
    if not _REGISTRY_PATH.exists():
        return {}
    try:
        with open(_REGISTRY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _model_versions(registry: dict) -> list[dict]:
    """Return a flat list of model version records for the UI table."""
    _types = {
        "bert":       ("bert_classifier",       "macro_f1"),
        "ner":        ("spacy_ner_model",        "overall_f1"),
        "actor":      ("actor_classifier",       "macro_f1"),
        "risk":       ("risk_classifier",        "macro_f1"),
        "prohibited": ("prohibited_classifier",  "macro_f1"),
    }
    rows = []
    for model_type, (dir_name, metric_key) in _types.items():
        section = registry.get(model_type, {})
        active_ver = section.get("active")
        active_dir = _ARTIFACTS_ROOT / dir_name
        on_disk = active_dir.is_dir() and any(active_dir.iterdir())
        versions = section.get("versions", {})

        if versions:
            for ver, info in versions.items():
                score = info.get("metrics", {}).get(metric_key)
                rows.append({
                    "model_type": model_type,
                    "version": ver,
                    "is_active": ver == active_ver,
                    "created_at": info.get("created_at"),
                    "metric_key": metric_key,
                    "score": round(score, 4) if score is not None else None,
                    "data_version": info.get("data_version"),
                    "on_disk": on_disk,
                })
        else:
            rows.append({
                "model_type": model_type,
                "version": None,
                "is_active": False,
                "created_at": None,
                "metric_key": metric_key,
                "score": None,
                "data_version": None,
                "on_disk": on_disk,
            })
    return rows


def _data_versions(registry: dict) -> list[dict]:
    """Return data version records for the UI table."""
    _types = {
        "bert":       "data/clause_labels.jsonl",
        "ner":        "data/ner_annotations.jsonl",
        "actor":      "data/actor_labels.jsonl",
        "risk":       "data/risk_labels.jsonl",
        "prohibited": "data/prohibited_labels.jsonl",
    }
    rows = []
    for data_type, rel_path in _types.items():
        key = f"data_{data_type}"
        section = registry.get(key, {})
        active_path = _TRAINING_ROOT / rel_path
        record_count = None
        if active_path.exists():
            try:
                record_count = sum(1 for ln in open(active_path, encoding="utf-8") if ln.strip())
            except Exception:
                pass
        rows.append({
            "data_type": data_type,
            "active_version": section.get("active"),
            "total_versions": len(section.get("versions", {})),
            "current_records": record_count,
            "file_exists": active_path.exists(),
        })
    return rows


async def _chroma_stats(request: Request) -> dict:
    """Query ChromaDB for collection counts."""
    try:
        chroma = request.app.state.chroma
        results = {}
        for name in ("eu_ai_act", "gdpr", "compliance_checklist"):
            try:
                col = await asyncio.to_thread(chroma._client.get_collection, name)
                results[name] = await asyncio.to_thread(col.count)
            except Exception:
                results[name] = None
        return results
    except Exception:
        return {}


async def _service_health(request: Request) -> dict:
    """Check health of API-dependent services."""
    from config import settings
    from services.chroma_client import ChromaClient
    from services.ollama_client import OllamaClient

    chroma = request.app.state.chroma
    chroma_ok = await chroma.health_check()

    ollama = OllamaClient(host=settings.ollama_host, model=settings.ollama_model)
    ollama_ok = await ollama.health_check()

    return {
        "chromadb": chroma_ok,
        "ollama": ollama_ok,
        "api": True,
    }


def _system_resources() -> dict:
    """Return CPU/memory stats via psutil (gracefully disabled if not installed)."""
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_used_mb": round(vm.used / 1024 / 1024),
            "memory_total_mb": round(vm.total / 1024 / 1024),
            "memory_percent": vm.percent,
        }
    except ImportError:
        return {"available": False}


@router.get("/", response_model=APIResponse)
async def get_monitoring_snapshot(request: Request) -> APIResponse:
    """Return a full observability snapshot of the KlarKI application.

    Includes:
    - Service health (ChromaDB, Ollama)
    - Audit pipeline metrics (totals, success rate, avg duration)
    - Per-stage timing (avg, p95)
    - LangGraph node invocation counts and timing
    - ChromaDB collection sizes
    - Model version registry
    - Data version registry
    - System resources (CPU, memory)
    """
    registry = _load_registry()

    chroma_stats, service_health = await asyncio.gather(
        _chroma_stats(request),
        _service_health(request),
    )

    sys_resources = await asyncio.to_thread(_system_resources)
    pipeline_snapshot = pipeline_stats.snapshot()

    data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "services": service_health,
        "pipeline": pipeline_snapshot["pipeline"],
        "stages": pipeline_snapshot["stages"],
        "graph_nodes": pipeline_snapshot["graph_nodes"],
        "uptime_s": pipeline_snapshot["uptime_s"],
        "chromadb": {
            "collections": chroma_stats,
        },
        "models": _model_versions(registry),
        "data": _data_versions(registry),
        "system": sys_resources,
    }

    logger.info("monitoring_snapshot_served", total_audits=pipeline_snapshot["pipeline"]["total"])
    return APIResponse(status="success", data=data)
