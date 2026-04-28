#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KlarKI one-shot setup pipeline.

Runs every initialisation step in order:

  Stage 1    seed-ollama              - pull phi3:mini into the Ollama container
  Stage 1.5  seed-nli                 - download NLI cross-encoder to cache
  Stage 2    knowledge-base           - chunk + embed EU AI Act / GDPR -> ChromaDB
  Stage 2.5  build-graph              - generate obligation JSONL from regulatory text
  Stage 3    generate-data            - generate synthetic BERT training data via Ollama
  Stage 4    train-bert               - fine-tune deepset/gbert-base (GPU recommended)
  Stage 3.5  generate-specialist-data - generate actor/risk/prohibited training data
  Stage 4.5  train-specialist         - fine-tune actor/risk/prohibited classifiers
  Stage 5    generate-ner-data        - generate NER training data via deterministic templates
  Stage 6    train-ner                - train spaCy NER model
  Stage 7    export-bert              - export fine-tuned BERT -> ONNX
  Stage 8    export-e5                - export multilingual-e5-small -> ONNX
  Stage 9    benchmark                - latency comparison Ollama vs Triton

Re-running is safe: each long-running stage auto-skips if its outputs exist.
Use --retrain to force full regeneration of data + models + exports.

Usage:
    # Full pipeline (Ollama + ChromaDB must be running):
    python scripts/setup.py

    # Skip BERT/NER training and ONNX stages:
    python scripts/setup.py --skip-phase5

    # ChromaDB only (re-seed knowledge base):
    python scripts/setup.py --only knowledge-base

    # Triton pipeline only (assumes BERT already trained):
    python scripts/setup.py --only export-bert --only export-e5 --only benchmark

    # Dry run (print what would run, no execution):
    python scripts/setup.py --dry-run

Environment (defaults match .env.example):
    OLLAMA_HOST      http://localhost:11434
    CHROMA_HOST      http://localhost:8001
    OLLAMA_MODEL     phi3:mini
    TRITON_HOST      localhost
    TRITON_GRPC_PORT 8003
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# VersionManager — optional; gracefully disabled if training deps not installed
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
    from version_manager import VersionManager as _VersionManager  # type: ignore
    _VM: "_VersionManager | None" = _VersionManager()
except Exception:
    _VM = None


# Populated by generate stages; read by train stages to wire data_version into the model registry.
_DATA_VERSIONS: dict[str, str] = {}


def _min_per_class_lang(filepath: Path) -> int:
    """Return the minimum record count for any (label, lang) pair in a JSONL file.

    Used to decide whether specialist data files have enough records to skip
    regeneration (file-existence alone is not sufficient).
    """
    if not filepath.exists():
        return 0
    counts: dict[tuple[str, str], int] = {}
    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec.get("label", "?"), rec.get("lang", "?"))
                counts[key] = counts.get(key, 0) + 1
            except json.JSONDecodeError:
                pass
    return min(counts.values(), default=0) if counts else 0


def _vm_snapshot_data(data_type: str, path: Path) -> str | None:
    """Snapshot data if content changed. Returns version string (no-op returns None)."""
    if _VM and path.exists():
        try:
            version, changed = _VM.snapshot_data(data_type, path)
            _DATA_VERSIONS[data_type] = version
            if not changed:
                print(_c(DIM, f"     [version] {data_type} data unchanged — training skip eligible"))
            return version
        except Exception as exc:
            print(_c(DIM, f"     [version] snapshot skipped: {exc}"))
    return None


def _vm_save_model(model_type: str, model_dir: Path, metrics_path: Path, data_type: str | None = None) -> None:
    """Save versioned model copy and promote if best (no-op if VM unavailable)."""
    if _VM and model_dir.is_dir():
        try:
            metrics: dict = {}
            if metrics_path.exists():
                with open(metrics_path, encoding="utf-8") as f:
                    metrics = json.load(f)
            data_version = _DATA_VERSIONS.get(data_type) if data_type else None
            _VM.save_and_promote(model_type, model_dir, metrics, data_version=data_version)
        except Exception as exc:
            print(_c(DIM, f"     [version] model save skipped: {exc}"))


def _should_retrain(model_type: str, data_type: str) -> bool:
    """Return True if VersionManager says data changed since last model training."""
    if _VM:
        try:
            return _VM.should_retrain(model_type, data_type)
        except Exception:
            pass
    return True  # can't determine — safe default

# Windows cp1252 stdout can't encode UTF-8 output from subprocesses (German
# characters, progress bars). Reconfigure to UTF-8 so streamed output renders.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")



RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
DIM    = "\033[2m"


def _c(colour: str, text: str) -> str:
    return f"{colour}{text}{RESET}"


def banner(text: str) -> None:
    width = 62
    print()
    print(_c(CYAN, "-" * width))
    print(_c(BOLD + CYAN, f"  {text}"))
    print(_c(CYAN, "-" * width))


def step(msg: str) -> None:
    print(_c(BOLD, f"\n  >>  {msg}"))


def _progress_bar(current: int, total: int, width: int = 24) -> str:
    """Return a filled ASCII bar: [########....] 4/8."""
    filled = int(width * current / max(total, 1))
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}] {current}/{total}"


def stage_header(idx: int, total: int, desc: str, elapsed_so_far: float) -> None:
    """Print a prominent stage header with overall pipeline progress."""
    bar = _progress_bar(idx - 1, total)
    eta_str = ""
    if idx > 1 and elapsed_so_far > 0:
        avg_per_stage = elapsed_so_far / (idx - 1)
        remaining = avg_per_stage * (total - (idx - 1))
        eta_str = _c(DIM, f"  ~{remaining / 60:.0f} min remaining")
    print()
    print(_c(CYAN, f"  +-- Stage {idx}/{total}  {bar}{eta_str}"))
    print(_c(BOLD + CYAN, f"  |  {desc}"))
    print(_c(CYAN,  "  +" + "-" * 54))


def ok(msg: str, elapsed: float) -> None:
    print(_c(GREEN, f"  OK  {msg}") + _c(DIM, f"  ({elapsed:.1f}s)"))


def skip_msg(msg: str) -> None:
    print(_c(YELLOW, f"  --  {msg}  (skipped)"))


def fail(msg: str, elapsed: float) -> None:
    print(_c(RED, f"  !!  {msg}") + _c(DIM, f"  ({elapsed:.1f}s)"))


def run(cmd: list[str], dry_run: bool = False) -> int:
    """Run a subprocess, streaming stdout+stderr in real-time. Returns exit code."""
    import os
    print(_c(DIM, "     $ " + " ".join(str(c) for c in cmd)))
    sys.stdout.flush()
    if dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Merge stderr into stdout so all output is streamed live in order.
    proc = subprocess.Popen(
        cmd, text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    return proc.returncode


ROOT = Path(__file__).parent.parent  # project root


def stage_seed_ollama(args: argparse.Namespace) -> bool:
    """Stage 1 - pull Ollama model (pure Python, no bash dependency)."""
    step("Seeding Ollama model")
    host = args.ollama_host.rstrip("/")
    model = args.ollama_model
    print(_c(DIM, f"     ollama_host={host}  model={model}"))

    if args.dry_run:
        return True

    # Wait for Ollama to be ready
    max_wait, interval = 60, 3
    elapsed = 0
    while True:
        try:
            urllib.request.urlopen(f"{host}/api/tags", timeout=5)
            break
        except (urllib.error.URLError, OSError):
            if elapsed >= max_wait:
                print(_c(RED, f"     ERROR: Ollama not ready after {max_wait}s"))
                return False
            print(_c(DIM, f"     Waiting for Ollama... ({elapsed}s)"))
            time.sleep(interval)
            elapsed += interval

    # Check if model already present
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as resp:
            tags = json.loads(resp.read())
        if any(m.get("name", "").startswith(model) for m in tags.get("models", [])):
            print(_c(DIM, f"     Model '{model}' already present, skipping pull."))
            return True
    except Exception:
        pass  # proceed to pull

    # Pull the model
    print(_c(DIM, f"     Pulling '{model}'... (may take several minutes)"))
    payload = json.dumps({"name": model, "stream": False}).encode()
    req = urllib.request.Request(
        f"{host}/api/pull",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read())
        print(_c(DIM, f"     {result.get('status', 'done')}"))
        return True
    except Exception as exc:
        print(_c(RED, f"     ERROR pulling model: {exc}"))
        return False


def stage_seed_nli(args: argparse.Namespace) -> bool:
    """Stage 1.5 - download NLI model to cache."""
    step("Seeding NLI model (cross-encoder/nli-deberta-v3-small)")
    if args.dry_run:
        return True
    
    cmd = [
        sys.executable,
        "-c",
        "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/nli-deberta-v3-small')"
    ]
    return run(cmd, args.dry_run) == 0


def stage_knowledge_base(args: argparse.Namespace) -> bool:
    """Stage 2 - build ChromaDB knowledge base."""
    step("Building ChromaDB knowledge base")
    # build_knowledge_base.py expects --host <hostname> --port <int>, not a full URL
    host = args.chroma_host
    port = "8001"
    if "://" in host:
        # Strip scheme: "http://localhost:8001" -> host="localhost", port="8001"
        host = host.split("://", 1)[1]
    if ":" in host:
        host, port = host.rsplit(":", 1)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_knowledge_base.py"),
        "--host", host,
        "--port", port,
    ]
    if args.rebuild_kb:
        cmd.append("--rebuild")
    return run(cmd, args.dry_run) == 0


def stage_build_graph(args: argparse.Namespace) -> bool:
    """Stage 2.5 - Write deterministic obligation JSONL files from hardcoded legal text."""
    step("Building structured obligation JSONL (deterministic, no Ollama)")
    graph_dir = ROOT / "data" / "obligations"
    
    if not args.gen_overwrite and not args.dry_run and graph_dir.exists() and any(graph_dir.glob("*/*.jsonl")):
        print(_c(YELLOW, "  --  data/obligations/ already contains .jsonl files -- skipping graph generation."))
        print(_c(DIM, "      Pass --gen-overwrite to force regeneration."))
        return True

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_structured_obligations.py")
    ]
    return run(cmd, args.dry_run) == 0


def stage_generate_data(args: argparse.Namespace) -> bool:
    """Stage 3 - generate synthetic BERT training data via Ollama.

    Auto-skips if clause_labels.jsonl already contains enough examples
    (>= gen_per_class * 8 classes). Pass --gen-overwrite to force regeneration.
    """
    step("Generating synthetic BERT training data via Ollama")
    data_path = ROOT / "training" / "data" / "clause_labels.jsonl"

    # Auto-skip if the corpus already has sufficient data
    if not args.gen_overwrite and not args.dry_run and data_path.exists():
        line_count = sum(1 for ln in data_path.open(encoding="utf-8") if ln.strip())
        min_needed = args.gen_per_class * 8 * 2  # 8 classes × 2 languages
        if line_count >= min_needed:
            print(_c(YELLOW,
                f"  --  clause_labels.jsonl already has {line_count} examples "
                f"(>= {min_needed} target) -- skipping generation."))
            print(_c(DIM, "      Pass --gen-overwrite to force regeneration."))
            return True

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_bert_training_data.py"),
        "--output",        str(data_path),
        "--ollama-host",   args.ollama_host,
        "--ollama-model",  args.ollama_model,
        "--n-per-class",   str(args.gen_per_class),
        "--batch-size",    str(args.gen_batch),
    ]
    if args.gen_overwrite:
        cmd.append("--overwrite")
    ok = run(cmd, args.dry_run) == 0
    if ok:
        _vm_snapshot_data("bert", ROOT / "training" / "data" / "clause_labels.jsonl")
    return ok


def stage_train_bert(args: argparse.Namespace) -> bool:
    """Stage 4 - fine-tune BERT classifier.

    Auto-skips if bert_classifier/ already exists with a trained model.
    Pass --retrain to force retraining.
    """
    step("Training BERT classifier (deepset/gbert-base)")
    bert_dir = ROOT / "training" / "artifacts" / "bert_classifier"
    model_exists = (bert_dir / "config.json").exists() and (
        (bert_dir / "model.safetensors").exists() or (bert_dir / "pytorch_model.bin").exists()
    )
    if not args.dry_run and model_exists:
        if args.force_retrain:
            pass  # always train a new version
        elif args.retrain:
            if not _should_retrain("bert", "bert"):
                print(_c(YELLOW, "  --  bert data unchanged since last training — skipping BERT retrain."))
                return True
        else:
            print(_c(YELLOW, "  --  artifacts/bert_classifier/ already exists -- skipping BERT training."))
            print(_c(DIM, "      Run with --retrain (data-change-aware) or --force-retrain."))
            return True
    cmd = [
        sys.executable,
        str(ROOT / "training" / "scripts" / "train_classifier.py"),
        "--data",       str(ROOT / "training" / "data" / "clause_labels.jsonl"),
        "--output",     str(ROOT / "training" / "artifacts" / "bert_classifier"),
        "--epochs",     str(args.bert_epochs),
        "--batch-size", str(args.bert_batch),
    ]
    ok = run(cmd, args.dry_run) == 0
    if ok:
        _vm_save_model("bert", ROOT / "training" / "artifacts" / "bert_classifier",
                       ROOT / "training" / "artifacts" / "bert_classifier" / "metrics.json",
                       data_type="bert")
    return ok


def stage_generate_ner_data(args: argparse.Namespace) -> bool:
    """Stage - generate NER training data via deterministic template expansion.

    No LLM required. Auto-skips if ner_annotations.jsonl already has enough records.
    Pass --gen-overwrite to force regeneration.
    """
    step("Generating NER training data (deterministic templates — no Ollama required)")
    data_path = ROOT / "training" / "data" / "ner_annotations.jsonl"

    min_needed = args.ner_templates
    if not args.gen_overwrite and not args.dry_run and data_path.exists():
        line_count = sum(1 for ln in data_path.open(encoding="utf-8") if ln.strip())
        if line_count >= min_needed:
            print(_c(YELLOW,
                f"  --  ner_annotations.jsonl already has {line_count} records "
                f"(>= {min_needed} target) -- skipping generation."))
            print(_c(DIM, "      Pass --gen-overwrite to force regeneration."))
            return True

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_ner_data.py"),
        "--output",       str(data_path),
        "--n-templates",  str(args.ner_templates),
    ]
    if args.gen_overwrite:
        cmd.append("--overwrite")
    ok = run(cmd, args.dry_run) == 0
    if ok:
        _vm_snapshot_data("ner", ROOT / "training" / "data" / "ner_annotations.jsonl")
    return ok


def stage_train_ner(args: argparse.Namespace) -> bool:
    """Stage - train spaCy NER model.

    Auto-skips if spacy_ner_model/model-final already exists.
    Pass --retrain to force retraining.
    """
    step("Training spaCy NER model")
    ner_model = ROOT / "training" / "artifacts" / "spacy_ner_model" / "model-final"
    if not args.dry_run and ner_model.exists():
        if args.force_retrain:
            pass
        elif args.retrain:
            if not _should_retrain("ner", "ner"):
                print(_c(YELLOW, "  --  NER data unchanged since last training — skipping NER retrain."))
                return True
        else:
            print(_c(YELLOW, "  --  artifacts/spacy_ner_model/model-final already exists -- skipping NER training."))
            print(_c(DIM, "      Run with --retrain (data-change-aware) or --force-retrain."))
            return True
    cmd = [
        sys.executable,
        str(ROOT / "training" / "scripts" / "train_ner.py"),
        "--data",       str(ROOT / "training" / "data" / "ner_annotations.jsonl"),
        "--output",     str(ROOT / "training" / "artifacts" / "spacy_ner_model"),
        "--epochs",     str(args.ner_epochs),
        "--batch-size", str(args.ner_batch),
        "--patience",   str(args.ner_patience),
    ]
    ok = run(cmd, args.dry_run) == 0
    if ok:
        _vm_save_model("ner", ROOT / "training" / "artifacts" / "spacy_ner_model",
                       ROOT / "training" / "artifacts" / "spacy_ner_model" / "metrics.json",
                       data_type="ner")
    return ok


def stage_generate_specialist_data(args: argparse.Namespace) -> bool:
    """Stage 3.5 - generate specialist classifier training data (actor / risk / prohibited).

    Two-step pipeline:
      1. Weak supervision (deterministic, no Ollama) — extracts labeled sentences
         from data/regulatory/**/*.txt using regex pattern matching.
      2. LLM augmentation (Ollama) — generates additional synthetic examples to
         reach --gen-per-class target per class per language.

    Auto-skips the LLM step if every (label, lang) combination already has at least
    gen_per_class // 4 examples. Pass --gen-overwrite to force both steps.
    """
    step("Generating specialist classifier training data")
    data_dir = ROOT / "training" / "data"
    needed_files = ["actor_labels.jsonl", "risk_labels.jsonl", "prohibited_labels.jsonl"]

    # ── Step 1: weak supervision (always runs unless --gen-overwrite skips via its own logic) ──
    ws_script = ROOT / "scripts" / "build_weak_supervision_labels.py"
    if ws_script.exists():
        print(_c(DIM, "  Pre-step: extracting labeled sentences from regulatory text (no Ollama)..."))
        ws_cmd = [sys.executable, str(ws_script), "--type", "all"]
        if args.gen_overwrite:
            ws_cmd.append("--overwrite")
        run(ws_cmd, args.dry_run)
    else:
        print(_c(YELLOW, f"  [warn] {ws_script} not found — skipping weak supervision pre-step"))

    # ── Step 2: LLM augmentation — skip if already sufficient ────────────────────
    min_threshold = max(30, args.gen_per_class // 4)
    if not args.gen_overwrite and not args.dry_run:
        all_sufficient = all(
            _min_per_class_lang(data_dir / f) >= min_threshold
            for f in needed_files
        )
        if all_sufficient:
            print(_c(YELLOW, f"  --  all specialist label files have ≥{min_threshold} records per "
                              "class/lang -- skipping LLM augmentation."))
            print(_c(DIM, "      Pass --gen-overwrite to force regeneration."))
            return True
        elif any((data_dir / f).exists() for f in needed_files):
            mins = {f: _min_per_class_lang(data_dir / f) for f in needed_files}
            print(_c(YELLOW, f"  --  insufficient records (min per class/lang: {mins}) — "
                              f"threshold={min_threshold}, running LLM augmentation."))

    script = ROOT / "scripts" / "generate_specialist_training_data.py"
    if not args.dry_run and not script.exists():
        print(_c(RED, f"     ERROR: {script} not found"))
        return False

    cmd = [
        sys.executable,
        str(script),
        "--type",        "all",
        "--ollama-host", args.ollama_host,
        "--model",       args.ollama_model,
        "--batch-size",  "10",
        "--n-per-class", str(args.gen_per_class),
    ]
    if args.gen_overwrite:
        cmd.append("--overwrite")
    ok = run(cmd, args.dry_run) == 0
    if ok:
        for dt in ("actor", "risk", "prohibited"):
            _vm_snapshot_data(dt, ROOT / "training" / "data" / f"{dt}_labels.jsonl")
    return ok


def stage_train_specialist(args: argparse.Namespace) -> bool:
    """Stage 4.5 - fine-tune all three specialist classifiers (actor / risk / prohibited).

    Auto-skips if all three model directories already exist.
    Pass --retrain to force retraining.
    """
    step("Training specialist classifiers (actor / risk / prohibited)")
    artifacts_dir = ROOT / "training" / "artifacts"
    model_dirs = ["actor_classifier", "risk_classifier", "prohibited_classifier"]

    script = ROOT / "training" / "scripts" / "train_specialist_classifiers.py"
    if not args.dry_run and not script.exists():
        print(_c(RED, f"     ERROR: {script} not found"))
        return False

    all_ok = True
    for classifier_type in ("actor", "risk", "prohibited"):
        model_dir = artifacts_dir / f"{classifier_type}_classifier"
        model_exists = (model_dir / "config.json").exists()

        if not args.dry_run and model_exists:
            if args.force_retrain:
                pass
            elif args.retrain:
                if not _should_retrain(classifier_type, classifier_type):
                    print(_c(YELLOW, f"  --  {classifier_type} data unchanged — skipping {classifier_type} retrain."))
                    continue
            else:
                print(_c(YELLOW, f"  --  {classifier_type}_classifier already trained -- skipping."))
                continue

        cmd = [
            sys.executable,
            str(script),
            "--type", classifier_type,
            "--epochs", str(args.bert_epochs),
            "--batch-size", str(args.bert_batch),
        ]
        if run(cmd, args.dry_run) != 0:
            print(_c(RED, f"     WARNING: {classifier_type} classifier training failed, continuing"))
            all_ok = False
        else:
            _vm_save_model(classifier_type, model_dir, model_dir / "metrics.json",
                           data_type=classifier_type)

    if not args.dry_run and all(
        (artifacts_dir / d / "config.json").exists() for d in model_dirs
    ) and all_ok:
        print(_c(DIM, "      Run with --retrain (data-change-aware) or --force-retrain to train new versions."))
    return all_ok


def stage_export_bert(args: argparse.Namespace) -> bool:
    """Stage 5 - export fine-tuned BERT to ONNX.

    Auto-skips if the ONNX model already exists and --retrain was not passed.
    """
    step("Exporting BERT classifier to ONNX")
    bert_dir = ROOT / "training" / "artifacts" / "bert_classifier"
    onnx_path = ROOT / "model_repository" / "bert_clause_classifier" / "1" / "model.onnx"

    if not args.gen_overwrite and not args.dry_run and onnx_path.exists():
        print(_c(YELLOW, "  --  bert_clause_classifier ONNX already exists -- skipping export."))
        print(_c(DIM, "      Run './run.sh retrain' to force re-export."))
        return True

    if not args.dry_run and not bert_dir.exists():
        print(_c(RED, f"     ERROR: {bert_dir} not found - run train-bert first"))
        return False
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_onnx.py"),
        "--model-path",  str(bert_dir),
        "--output-path", str(onnx_path),
        "--model-type",  "classifier",
    ]
    return run(cmd, args.dry_run) == 0


def stage_export_e5(args: argparse.Namespace) -> bool:
    """Stage 6 - export multilingual-e5-small to ONNX.

    Auto-skips if the ONNX model already exists and --retrain was not passed.
    """
    step("Exporting multilingual-e5-small to ONNX")
    onnx_path = ROOT / "model_repository" / "e5_embeddings" / "1" / "model.onnx"

    if not args.gen_overwrite and not args.dry_run and onnx_path.exists():
        print(_c(YELLOW, "  --  e5_embeddings ONNX already exists -- skipping export."))
        print(_c(DIM, "      Run './run.sh retrain' to force re-export."))
        return True

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_onnx.py"),
        "--model-path",  "intfloat/multilingual-e5-small",
        "--output-path", str(onnx_path),
        "--model-type",  "embeddings",
    ]
    return run(cmd, args.dry_run) == 0


def stage_benchmark(args: argparse.Namespace) -> bool:
    """Stage 7 - run latency benchmark."""
    step("Running Ollama vs Triton benchmark")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_triton.py"),
        "--n-samples",   str(args.bench_samples),
        "--ollama-host", args.ollama_host,
        "--triton-host", args.triton_host,
        "--triton-port", str(args.triton_port),
    ]
    return run(cmd, args.dry_run) == 0


STAGES: list[tuple[str, str, bool]] = [
    # (id, description, skip_with_skip_phase5)
    ("seed-ollama",              "Seed Ollama model",                          False),
    ("seed-nli",                 "Seed NLI model",                             False),
    ("knowledge-base",           "Build ChromaDB knowledge base",              False),
    ("build-graph",              "Generate structured Knowledge Graph",        False),
    ("generate-data",            "Generate BERT training data",                True),
    ("train-bert",               "Train BERT classifier",                      True),
    ("generate-specialist-data", "Generate specialist classifier training data", True),
    ("train-specialist",         "Train specialist classifiers (actor/risk/prohibited)", True),
    ("generate-ner-data",        "Generate NER training data",                 True),
    ("train-ner",                "Train spaCy NER",                            True),
    ("export-bert",              "Export BERT to ONNX",                        True),
    ("export-e5",                "Export e5-small to ONNX",                    True),
    ("benchmark",                "Benchmark Triton vs Ollama",                 True),
]

STAGE_FNS = {
    "seed-ollama":              stage_seed_ollama,
    "seed-nli":                 stage_seed_nli,
    "knowledge-base":           stage_knowledge_base,
    "build-graph":              stage_build_graph,
    "generate-data":            stage_generate_data,
    "train-bert":               stage_train_bert,
    "generate-specialist-data": stage_generate_specialist_data,
    "train-specialist":         stage_train_specialist,
    "generate-ner-data":        stage_generate_ner_data,
    "train-ner":                stage_train_ner,
    "export-bert":              stage_export_bert,
    "export-e5":                stage_export_e5,
    "benchmark":                stage_benchmark,
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KlarKI one-shot setup pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Stage selection
    stage_ids = [s[0] for s in STAGES]
    p.add_argument(
        "--only", metavar="STAGE", action="append", dest="only_stages",
        choices=stage_ids,
        help="Run only this stage (repeat for multiple). Overrides --skip-*.",
    )
    p.add_argument("--skip-seed",      action="store_true", help="Skip Ollama model pull")
    p.add_argument("--skip-kb",        action="store_true", help="Skip ChromaDB knowledge base build")
    p.add_argument("--skip-generate",  action="store_true", help="Skip synthetic training data generation")
    p.add_argument("--skip-train",     action="store_true", help="Skip BERT + NER training")
    p.add_argument("--skip-export",    action="store_true", help="Skip ONNX export")
    p.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark")
    p.add_argument("--skip-phase5",    action="store_true",
                   help="Skip BERT/NER training and ONNX export stages (generate + train + export + benchmark)")

    # Behaviour
    p.add_argument("--dry-run",       action="store_true", help="Print commands without executing")
    p.add_argument("--rebuild-kb",    action="store_true", help="Force rebuild of ChromaDB collections")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Abort entire pipeline on first stage failure (default: continue)")

    # Service addresses
    p.add_argument("--ollama-host",  default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--chroma-host",  default=os.environ.get("CHROMADB_HOST", "http://localhost:8001"))
    p.add_argument("--triton-host",  default=os.environ.get("TRITON_HOST", "localhost"))
    p.add_argument("--triton-port",  type=int, default=8003)
    p.add_argument("--ollama-model", default="phi3:mini")

    # generate-data hyper-params
    p.add_argument("--gen-per-class", type=int, default=400,
                   help="Synthetic examples per class per language (default: 400)")
    p.add_argument("--gen-batch",     type=int, default=20,
                   help="Examples per Ollama call in generate-data stage (default: 20)")
    p.add_argument("--gen-overwrite", action="store_true",
                   help="Overwrite existing clause_labels.jsonl / ner_annotations.jsonl instead of appending")
    p.add_argument("--ner-templates", type=int, default=5000,
                   help="Template-generated NER records to produce (default: 5000)")
    p.add_argument("--retrain", action="store_true",
                   help="Regenerate data and retrain only if data content changed (hash-aware)")
    p.add_argument("--force-retrain", action="store_true",
                   help="Always train a new model version regardless of data changes")

    # Training hyper-params
    p.add_argument("--bert-epochs",   type=int, default=12)
    p.add_argument("--bert-batch",    type=int, default=16)
    p.add_argument("--ner-epochs",    type=int, default=60)
    p.add_argument("--ner-batch",     type=int, default=32)
    p.add_argument("--ner-patience",  type=int, default=10)
    p.add_argument("--bench-samples", type=int, default=20,
                   help="Number of samples per backend in benchmark")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    banner("KlarKI Setup Pipeline")

    if args.dry_run:
        print(_c(YELLOW, "  DRY RUN - commands printed, nothing executed\n"))

    # Build the list of stages to run
    skip_set: set[str] = set()
    if args.skip_seed:      skip_set.add("seed-ollama")
    if args.skip_kb:        skip_set.add("knowledge-base")
    if args.skip_generate:  skip_set.update({"generate-data", "generate-specialist-data", "generate-ner-data"})
    if args.skip_train:     skip_set.update({"train-bert", "train-specialist", "train-ner"})
    if args.skip_export:    skip_set.update({"export-bert", "export-e5"})
    if args.skip_benchmark: skip_set.add("benchmark")
    if args.skip_phase5:
        skip_set.update(s[0] for s in STAGES if s[2])  # phase5_only=True

    if args.force_retrain:
        # force-retrain: regenerate data unconditionally + always train new version
        args.gen_overwrite = True
        run_ids = [
            "generate-data", "train-bert",
            "generate-specialist-data", "train-specialist",
            "generate-ner-data", "train-ner",
            "export-bert", "export-e5",
        ]
    elif args.retrain:
        # retrain: regenerate data, then train only if data content actually changed
        args.gen_overwrite = True
        run_ids = [
            "generate-data", "train-bert",
            "generate-specialist-data", "train-specialist",
            "generate-ner-data", "train-ner",
            "export-bert", "export-e5",
        ]
    elif args.only_stages:
        run_ids = list(args.only_stages)
    else:
        run_ids = [s[0] for s in STAGES if s[0] not in skip_set]

    # Print plan
    print(_c(BOLD, "  Stages to run:"))
    for sid, desc, _ in STAGES:
        if sid in run_ids:
            print(f"    {_c(GREEN, '[x]')} {desc}")
        else:
            print(f"    {_c(DIM, '[ ]')} {desc}  {_c(DIM, '(skipped)')}")

    # Execute
    results: dict[str, tuple[bool | None, float]] = {}
    pipeline_start = time.time()

    active_stages = [(sid, desc) for sid, desc, _ in STAGES if sid in run_ids]
    total_active = len(active_stages)
    active_idx = 0  # counts only stages that will actually run

    for sid, desc, _ in STAGES:
        if sid not in run_ids:
            skip_msg(desc)
            results[sid] = (None, 0.0)
            continue

        active_idx += 1
        elapsed_so_far = time.time() - pipeline_start
        stage_header(active_idx, total_active, desc, elapsed_so_far)

        t0 = time.time()
        success = STAGE_FNS[sid](args)
        elapsed = time.time() - t0
        results[sid] = (success, elapsed)

        if success:
            ok(desc, elapsed)
        else:
            fail(desc, elapsed)
            if args.stop_on_error:
                print(_c(RED, "\n  Pipeline aborted (--stop-on-error)."))
                _print_summary(results, time.time() - pipeline_start)
                sys.exit(1)

    _print_summary(results, time.time() - pipeline_start)

    failed = [sid for sid, (outcome, _) in results.items() if outcome is False]
    sys.exit(1 if failed else 0)


def _print_summary(results: dict[str, tuple[bool | None, float]], total: float) -> None:
    banner("Summary")
    for sid, desc, _ in STAGES:
        outcome, elapsed = results.get(sid, (None, 0.0))
        if outcome is True:
            status = _c(GREEN, "PASS")
        elif outcome is False:
            status = _c(RED,   "FAIL")
        else:
            status = _c(DIM,   "skip")
        time_str = _c(DIM, f"  {elapsed:.1f}s") if elapsed else ""
        print(f"  {status}  {desc}{time_str}")

    n_failed = sum(1 for outcome, _ in results.values() if outcome is False)
    n_passed = sum(1 for outcome, _ in results.values() if outcome is True)
    n_total  = n_passed + n_failed

    # Final progress bar — fills only with passed stages
    bar = _progress_bar(n_passed, n_total) if n_total else ""
    print()
    colour = RED if n_failed else GREEN
    status_label = _c(GREEN, "ALL PASSED") if not n_failed else _c(RED, f"{n_failed} FAILED")
    print(f"  {bar}  {status_label}")
    print(f"  {_c(BOLD, f'{n_passed} passed')}, {_c(colour, f'{n_failed} failed')}  "
          + _c(DIM, f"({total:.1f}s  ~  {total/60:.1f} min total)"))


if __name__ == "__main__":
    main()
