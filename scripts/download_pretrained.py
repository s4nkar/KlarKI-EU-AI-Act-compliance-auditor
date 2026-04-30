"""Download all 5 pretrained KlarKI models from HuggingFace Hub.

Run this instead of `./run.sh setup` when you want to skip local training
and use the published models directly.

Usage:
    python scripts/download_pretrained.py                      # download all
    python scripts/download_pretrained.py --model bert         # single model
    python scripts/download_pretrained.py --force              # re-download even if present
    python scripts/download_pretrained.py --token hf_xxx       # private repos

    # From Python (e.g. custom setup scripts):
    from scripts.download_pretrained import download_pretrained_models
    download_pretrained_models(models=["bert", "ner"])

Models are placed at:
    training/artifacts/bert_classifier/
    training/artifacts/actor_classifier/
    training/artifacts/risk_classifier/
    training/artifacts/prohibited_classifier/
    training/artifacts/spacy_ner_model/model-final/   <- fine-tuned de_core_news_lg + KlarKI NER data

After downloading, run:
    ./run.sh up          # production
    ./run.sh dev         # hot-reload dev mode

Requires:  pip install huggingface-hub>=0.26.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "training"))

# ── Repo map ──────────────────────────────────────────────────────────────────
# Edit repo_id values here if you fork KlarKI and publish under a different HF username.
HF_MODELS: dict[str, dict] = {
    "bert": {
        "repo_id": "s4nkar/klarki-bert-classifier",
        # snapshot_download places all repo files directly into target/
        "target": _ROOT / "training" / "artifacts" / "bert_classifier",
        "is_spacy": False,
        "label": "Article-domain BERT classifier (8-class, Articles 9-15 + unrelated)",
    },
    "actor": {
        "repo_id": "s4nkar/klarki-actor-classifier",
        "target": _ROOT / "training" / "artifacts" / "actor_classifier",
        "is_spacy": False,
        "label": "Actor classifier (provider / deployer / importer / distributor)",
    },
    "risk": {
        "repo_id": "s4nkar/klarki-risk-classifier",
        "target": _ROOT / "training" / "artifacts" / "risk_classifier",
        "is_spacy": False,
        "label": "High-risk binary classifier (Article 6 + Annex III)",
    },
    "prohibited": {
        "repo_id": "s4nkar/klarki-prohibited-classifier",
        "target": _ROOT / "training" / "artifacts" / "prohibited_classifier",
        "is_spacy": False,
        "label": "Prohibited-practice binary classifier (Article 5)",
    },
    "ner": {
        "repo_id": "s4nkar/klarki-ner-spacy",
        # Files in the HF repo live under model-final/; snapshot_download places them
        # at target/model-final/ — which is exactly where ner_service.py expects them.
        "target": _ROOT / "training" / "artifacts" / "spacy_ner_model",
        "is_spacy": True,
        "label": "spaCy NER model (de_core_news_lg fine-tuned, 8 EU AI Act entity types)",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_present(info: dict) -> bool:
    """True if the model appears to already be downloaded."""
    target: Path = info["target"]
    if not target.exists():
        return False
    if info["is_spacy"]:
        return (target / "model-final").is_dir() and any(
            (target / "model-final").iterdir()
        )
    # BERT: must have config.json + at least one weight file
    return (target / "config.json").exists() and any(
        p.suffix in {".safetensors", ".bin"} for p in target.iterdir()
    )


def _repo_commit(repo_id: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi
        info = HfApi(token=token).repo_info(repo_id=repo_id, repo_type="model")
        return getattr(info, "sha", None) or "unknown"
    except Exception:
        return "unknown"


def _read_metrics(info: dict) -> dict:
    """Read metrics.json (BERT) or meta.json (spaCy) after download."""
    if info["is_spacy"]:
        path = info["target"] / "model-final" / "meta.json"
    else:
        path = info["target"] / "metrics.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ── Core download ─────────────────────────────────────────────────────────────

def _download_one(
    model_type: str,
    info: dict,
    token: str | None,
    force: bool,
) -> bool:
    target: Path = info["target"]

    if not force and _is_present(info):
        print(f"  [skip]  {model_type:12s}  already at {target.relative_to(_ROOT)}")
        return True

    print(f"  [fetch] {model_type:12s}  ← {info['repo_id']}")
    print(f"           {info['label']}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface-hub not installed. Run: pip install huggingface-hub>=0.26.0")
        return False

    target.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=info["repo_id"],
        local_dir=str(target),
        ignore_patterns=["*.gitattributes", ".cache/*"],
        token=token,
    )

    commit = _repo_commit(info["repo_id"], token)
    metrics = _read_metrics(info)

    try:
        from version_manager import VersionManager
        VersionManager().record_hub_download(model_type, info["repo_id"], commit, metrics)
    except Exception as exc:
        print(f"           warning: registry update skipped ({exc})")

    model_path = target / "model-final" if info["is_spacy"] else target
    print(f"           ✓ {model_path.relative_to(_ROOT)}")
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def download_pretrained_models(
    models: list[str] | None = None,
    force: bool = False,
    token: str | None = None,
) -> None:
    """Download pretrained KlarKI models from HuggingFace Hub.

    Args:
        models: List of model keys to download, e.g. ["bert", "ner"].
                Defaults to all 5 models.
        force:  Re-download even if files are already present.
        token:  HuggingFace token for private repos (or set HF_TOKEN env var).
    """
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    targets = {k: v for k, v in HF_MODELS.items() if not models or k in models}

    print(f"\nKlarKI — downloading {len(targets)} pretrained model(s) from HuggingFace Hub")
    print(f"Artifacts dir: {(_ROOT / 'training' / 'artifacts').relative_to(_ROOT)}\n")

    ok, failed = 0, []
    for model_type, info in targets.items():
        try:
            if _download_one(model_type, info, token, force):
                ok += 1
            else:
                failed.append(model_type)
        except Exception as exc:
            print(f"  [ERROR] {model_type}: {exc}")
            failed.append(model_type)

    print(f"\nResult: {ok}/{len(targets)} downloaded", end="")
    if failed:
        print(f"  —  failed: {failed}")
    else:
        print()

    if ok == len(targets):
        print("\nAll models ready.  Next step:")
        print("  ./run.sh up        # start the stack")
        print("  ./run.sh dev       # start with hot reload")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pretrained KlarKI models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=list(HF_MODELS.keys()) + ["all"],
        default="all",
        help="Which model to download (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the model is already present",
    )
    parser.add_argument(
        "--token",
        help="HuggingFace token (default: reads HF_TOKEN env var)",
    )
    args = parser.parse_args()

    models = None if args.model == "all" else [args.model]
    download_pretrained_models(models=models, force=args.force, token=args.token)


if __name__ == "__main__":
    main()
