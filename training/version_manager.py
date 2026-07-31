"""Model and data versioning manager for KlarKI.

Registry file:            training/artifacts/registry.json
Versioned data snapshots: training/artifacts/data_versions/<type>_<version>.jsonl
Versioned model dirs:     training/artifacts/<type>_<version>/
Active model dirs:        training/artifacts/<type>/

Source training data (committed) stays in training/data/.

Usage:
    from training.version_manager import VersionManager
    vm = VersionManager()
    if not vm.data_exists("bert"):
        generate_data()
        vm.snapshot_data("bert", Path("training/data/clause_labels.jsonl"))
    if not vm.model_exists("bert"):
        train()
        vm.save_and_promote("bert", Path("training/artifacts/bert_classifier"), metrics)
"""

import hashlib
import json
import shutil
import time
from pathlib import Path


_TRAINING_DIR  = Path(__file__).parent
_ARTIFACTS_DIR = _TRAINING_DIR / "artifacts"
_REGISTRY_PATH = _ARTIFACTS_DIR / "registry.json"
_VERSIONS_DIR  = _ARTIFACTS_DIR / "data_versions"

# Primary metric key per model type used for best-model comparison
_METRIC_KEYS: dict[str, str] = {
    "bert":       "macro_f1",
    "ner":        "overall_f1",
    "actor":      "macro_f1",
    "risk":       "macro_f1",
    "prohibited": "macro_f1",
}

# Active output directory names under _ARTIFACTS_DIR (where pipeline reads from)
_ACTIVE_DIRS: dict[str, str] = {
    "bert":       "bert_classifier",
    "ner":        "spacy_ner_model",
    "actor":      "actor_classifier",
    "risk":       "risk_classifier",
    "prohibited": "prohibited_classifier",
}

# Active data file paths relative to _TRAINING_DIR (committed source data)
_ACTIVE_DATA: dict[str, str] = {
    "bert":       "data/clause_labels.jsonl",
    "ner":        "data/ner_annotations.jsonl",
    "actor":      "data/actor_labels.jsonl",
    "risk":       "data/risk_labels.jsonl",
    "prohibited": "data/prohibited_labels.jsonl",
}


class VersionManager:
    """Read/write the training registry and manage versioned artefacts."""

    def __init__(self, registry_path: Path = _REGISTRY_PATH) -> None:
        self._path = registry_path
        _VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Registry I/O ─────────────────────────────────────────────────────────

    def load(self) -> dict:
        if not self._path.exists():
            return {}
        with open(self._path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ── Version helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _next_version(self, section: dict) -> str:
        existing = list(section.get("versions", {}).keys())
        if not existing:
            return "v1"
        nums = []
        for v in existing:
            try:
                nums.append(int(v.lstrip("v")))
            except ValueError:
                pass
        return f"v{max(nums) + 1}" if nums else "v1"

    def get_active_version(self, model_type: str) -> str | None:
        """Return the currently active version string, or None if no model recorded."""
        return self.load().get(model_type, {}).get("active")

    def get_model_data_version(self, model_type: str) -> str | None:
        """Return the data_version the currently active model was trained on."""
        registry = self.load()
        section = registry.get(model_type, {})
        active_ver = section.get("active")
        if not active_ver:
            return None
        return section.get("versions", {}).get(active_ver, {}).get("data_version")

    # ── Existence checks (used by setup.py to decide skip) ───────────────────

    def data_exists(self, data_type: str) -> bool:
        """True if the active data file for data_type is present on disk."""
        rel = _ACTIVE_DATA.get(data_type)
        if not rel:
            return False
        return (_TRAINING_DIR / rel).exists()

    def model_exists(self, model_type: str) -> bool:
        """True if the active model directory exists and has at least one file."""
        dir_name = _ACTIVE_DIRS.get(model_type)
        if not dir_name:
            return False
        d = _ARTIFACTS_DIR / dir_name
        return d.is_dir() and any(d.iterdir())

    # ── Data versioning ───────────────────────────────────────────────────────

    def snapshot_data(self, data_type: str, source_path: Path) -> tuple[str, bool]:
        """Copy the active data file into versions/ only if content changed.

        Returns (version, changed) where changed=False means the file matched
        the last snapshot hash and no new snapshot was created.
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Data file not found: {source_path}")

        current_hash = self._file_hash(source_path)
        registry = self.load()
        key = f"data_{data_type}"
        section = registry.setdefault(key, {"active": None, "versions": {}})

        # Skip if last snapshot has identical content
        active_ver = section.get("active")
        if active_ver and active_ver in section["versions"]:
            if section["versions"][active_ver].get("hash") == current_hash:
                print(f"  [version] Data {key}@{active_ver} unchanged (same hash) — no new snapshot")
                return active_ver, False

        version = self._next_version(section)
        ext = source_path.suffix
        dest = _VERSIONS_DIR / f"{data_type}_{version}{ext}"
        shutil.copy2(source_path, dest)

        record_count = sum(1 for line in open(source_path, encoding="utf-8") if line.strip())
        section["versions"][version] = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "record_count": record_count,
            "snapshot_path": str(dest),
            "hash": current_hash,
        }
        section["active"] = version
        self._save(registry)
        print(f"  [version] Data snapshot {key}@{version} → {dest} ({record_count} records)")
        return version, True

    # ── Model versioning ──────────────────────────────────────────────────────

    def _point_active(self, model_type: str, target_versioned_dir: Path) -> None:
        """Point training/artifacts/<type>/ at target_versioned_dir via a relative
        symlink, replacing whatever is currently there.

        Relative (not absolute) so the link resolves the same whether read from
        the host (/home/.../training/artifacts/...) or a container
        (/workspace/training/artifacts/...) — both see target_versioned_dir as a
        sibling in the same directory.
        """
        active_dir = _ARTIFACTS_DIR / _ACTIVE_DIRS[model_type]
        if active_dir.is_symlink() or active_dir.is_file():
            active_dir.unlink()
        elif active_dir.is_dir():
            # Legacy physical directory from before symlink-based promotion —
            # safe to remove, its content is byte-identical to some versioned_dir.
            shutil.rmtree(active_dir)
        active_dir.symlink_to(target_versioned_dir.name)

    def save_and_promote(
        self,
        model_type: str,
        source_dir: Path,
        metrics: dict,
        data_version: str | None = None,
    ) -> str:
        """Move source_dir (disposable training output) into a versioned
        directory, compare with the previous best, and point the active
        symlink at whichever version wins. Never stores the same model twice.

        Returns the new version string.
        """
        if not source_dir.is_dir():
            raise NotADirectoryError(f"Model dir not found: {source_dir}")

        registry = self.load()
        section = registry.setdefault(model_type, {"active": None, "versions": {}})
        version = self._next_version(section)

        versioned_dir = _ARTIFACTS_DIR / f"{_ACTIVE_DIRS[model_type]}_{version}"
        if versioned_dir.exists():
            shutil.rmtree(versioned_dir)
        shutil.move(str(source_dir), str(versioned_dir))

        primary_key = _METRIC_KEYS.get(model_type, "macro_f1")
        # Prefer the honest gold-set score when the new model reports it: gold is
        # measured on real out-of-distribution text, while the primary key
        # (macro_f1 / overall_f1) is computed on the same synthetic distribution
        # the model trained on and routinely reads ~99% even when gold fails.
        metric_key = "gold_macro_f1" if "gold_macro_f1" in metrics else primary_key
        new_score = float(metrics.get(metric_key, 0.0))

        section["versions"][version] = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "versioned_dir": str(versioned_dir),
            "metrics": metrics,
            "data_version": data_version,
        }

        # Compare with currently active version — promote only if better
        active_ver = section.get("active")
        if active_ver and active_ver in section["versions"] and active_ver != version:
            prev_metrics = section["versions"][active_ver].get("metrics", {})
            prev_dir = Path(section["versions"][active_ver]["versioned_dir"])
            # Compound gold gate as a promotion criterion: never replace a model
            # that PASSES the honest gold gate with one that FAILS it, even if the
            # failing model's macro_f1 is nominally higher (e.g. a prohibited model
            # that trades away TNR). If no active model passes, fall through to the
            # score comparison so the system always keeps its best available model.
            new_passed = bool(metrics.get("gold_passed", True))
            prev_passed = bool(prev_metrics.get("gold_passed", True))
            if prev_passed and not new_passed:
                print(
                    f"  [version] {model_type}@{version} FAILS gold gate while "
                    f"{model_type}@{active_ver} passes → keeping {active_ver}"
                )
                self._point_active(model_type, prev_dir)
                self._save(registry)
                return version
            # If the previous version predates gold gating, fall back to the
            # primary key for a fair like-for-like comparison.
            if metric_key == "gold_macro_f1" and "gold_macro_f1" not in prev_metrics:
                metric_key = primary_key
                new_score = float(metrics.get(metric_key, 0.0))
            prev_score = float(prev_metrics.get(metric_key, 0.0))
            if new_score > prev_score:
                print(
                    f"  [version] {model_type}@{version} {metric_key}={new_score:.4f} "
                    f"> {model_type}@{active_ver} {metric_key}={prev_score:.4f} → PROMOTED"
                )
                section["active"] = version
                self._point_active(model_type, versioned_dir)
            else:
                print(
                    f"  [version] {model_type}@{version} {metric_key}={new_score:.4f} "
                    f"<= {model_type}@{active_ver} {metric_key}={prev_score:.4f} → keeping {active_ver}"
                )
                self._point_active(model_type, prev_dir)
        else:
            # First version — always promote
            section["active"] = version
            print(f"  [version] {model_type}@{version} {metric_key}={new_score:.4f} → ACTIVE (first version)")
            self._point_active(model_type, versioned_dir)

        self._save(registry)
        return version

    # ── HuggingFace Hub tracking ──────────────────────────────────────────────

    def record_hub_push(
        self,
        model_type: str,
        version: str,
        repo_id: str,
        commit_hash: str,
    ) -> None:
        """Attach hub push metadata to an existing version entry."""
        registry = self.load()
        ver_entry = (
            registry.setdefault(model_type, {"active": None, "versions": {}})
            .setdefault("versions", {})
            .setdefault(version, {})
        )
        ver_entry["hub"] = {
            "repo_id": repo_id,
            "commit": commit_hash,
            "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save(registry)
        print(f"  [hub] {model_type}@{version} → {repo_id} ({commit_hash[:8]}…)")

    def record_hub_download(
        self,
        model_type: str,
        repo_id: str,
        commit_hash: str,
        metrics: dict | None = None,
    ) -> str:
        """Create a version entry for a hub-downloaded model and set it active.

        The model files are assumed to already be in the active directory
        (placed there by snapshot_download). Returns the new version string.
        """
        registry = self.load()
        section = registry.setdefault(model_type, {"active": None, "versions": {}})
        version = self._next_version(section)
        active_dir = _ARTIFACTS_DIR / _ACTIVE_DIRS[model_type]

        section["versions"][version] = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "versioned_dir": str(active_dir),
            "metrics": metrics or {},
            "data_version": None,
            "hub": {
                "repo_id": repo_id,
                "commit": commit_hash,
                "source": "downloaded",
                "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        }
        section["active"] = version
        self._save(registry)
        print(
            f"  [hub] {model_type}@{version} ← {repo_id} "
            f"({commit_hash[:8]}…) → ACTIVE"
        )
        return version

    def get_hub_info(self, model_type: str) -> dict | None:
        """Return hub metadata for the active version, or None if not hub-sourced."""
        registry = self.load()
        section = registry.get(model_type, {})
        active_ver = section.get("active")
        if not active_ver:
            return None
        return section.get("versions", {}).get(active_ver, {}).get("hub")

    def should_retrain(self, model_type: str, data_type: str) -> bool:
        """True if the active data version differs from what the active model was trained on.

        Returns True (must retrain) when:
        - No model exists yet
        - data_version on the model entry is null (legacy, pre-tracking)
        - The data version that trained the model != the current active data version

        Hub-downloaded models are treated as authoritative and never trigger retrain
        unless the user explicitly passes --retrain.
        """
        if not self.model_exists(model_type):
            return True
        hub_info = self.get_hub_info(model_type)
        if hub_info and hub_info.get("source") == "downloaded":
            return False
        registry = self.load()
        model_data_ver = self.get_model_data_version(model_type)
        current_data_ver = registry.get(f"data_{data_type}", {}).get("active")
        if model_data_ver is None or current_data_ver is None:
            return True
        return model_data_ver != current_data_ver

    # ── Summary helpers (used by monitoring) ─────────────────────────────────

    def get_all_model_versions(self) -> dict:
        """Return a dict of {model_type: {active, versions: [...]}} for monitoring."""
        registry = self.load()
        result = {}
        for model_type in _ACTIVE_DIRS:
            if model_type not in registry:
                result[model_type] = {"active": None, "versions": []}
                continue
            section = registry[model_type]
            versions = []
            for ver, info in section.get("versions", {}).items():
                metric_key = _METRIC_KEYS.get(model_type, "macro_f1")
                score = info.get("metrics", {}).get(metric_key)
                versions.append({
                    "version": ver,
                    "created_at": info.get("created_at"),
                    "score": score,
                    "data_version": info.get("data_version"),
                    "is_active": ver == section.get("active"),
                    "hub": info.get("hub"),
                })
            versions.sort(key=lambda x: x["version"])
            result[model_type] = {
                "active": section.get("active"),
                "metric_key": _METRIC_KEYS.get(model_type, "macro_f1"),
                "active_dir": str(_ARTIFACTS_DIR / _ACTIVE_DIRS[model_type]),
                "versions": versions,
            }
        return result

    def get_all_data_versions(self) -> dict:
        """Return data version summary for monitoring."""
        registry = self.load()
        result = {}
        for data_type in _ACTIVE_DATA:
            key = f"data_{data_type}"
            if key not in registry:
                active_path = _TRAINING_DIR / _ACTIVE_DATA[data_type]
                record_count = None
                if active_path.exists():
                    record_count = sum(1 for ln in open(active_path, encoding="utf-8") if ln.strip())
                result[data_type] = {"active": None, "versions": [], "current_records": record_count}
                continue
            section = registry[key]
            versions = []
            for ver, info in section.get("versions", {}).items():
                versions.append({
                    "version": ver,
                    "created_at": info.get("created_at"),
                    "record_count": info.get("record_count"),
                    "is_active": ver == section.get("active"),
                })
            versions.sort(key=lambda x: x["version"])
            result[data_type] = {"active": section.get("active"), "versions": versions}
        return result
