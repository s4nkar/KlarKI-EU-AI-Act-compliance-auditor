#!/usr/bin/env python3
"""KlarKI one-shot setup pipeline.

Runs every initialisation step in order:

  Stage 1  seed-ollama     - pull phi3:mini into the Ollama container
  Stage 2  knowledge-base  - chunk + embed EU AI Act / GDPR -> ChromaDB
  Stage 3  train-bert      - fine-tune deepset/gbert-base (Phase 5, GPU optional)
  Stage 4  train-ner       - train spaCy NER model (Phase 5)
  Stage 5  export-bert     - export fine-tuned BERT -> ONNX (Phase 5)
  Stage 6  export-e5       - export multilingual-e5-small -> ONNX (Phase 5)
  Stage 7  benchmark       - latency comparison Ollama vs Triton (Phase 5)

Usage:
    # Full pipeline (Ollama + ChromaDB must be running):
    python scripts/setup.py

    # Skip GPU-heavy Phase 5 stages (fast first-run setup):
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
import os
import subprocess
import sys
import time
from pathlib import Path

# -- Colour helpers ------------------------------------------------------------

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


def ok(msg: str, elapsed: float) -> None:
    print(_c(GREEN, f"  OK  {msg}") + _c(DIM, f"  ({elapsed:.1f}s)"))


def skip_msg(msg: str) -> None:
    print(_c(YELLOW, f"  --  {msg}  (skipped)"))


def fail(msg: str, elapsed: float) -> None:
    print(_c(RED, f"  !!  {msg}") + _c(DIM, f"  ({elapsed:.1f}s)"))


# -- Stage runner --------------------------------------------------------------

def run(cmd: list[str], dry_run: bool = False) -> int:
    """Run a subprocess, streaming output. Returns exit code."""
    print(_c(DIM, "     $ " + " ".join(str(c) for c in cmd)))
    if dry_run:
        return 0
    result = subprocess.run(cmd, text=True)
    return result.returncode


# -- Stage definitions ---------------------------------------------------------

ROOT = Path(__file__).parent.parent  # project root


def stage_seed_ollama(args: argparse.Namespace) -> bool:
    """Stage 1 - pull Ollama model via seed_ollama.sh."""
    step("Seeding Ollama model")
    script = ROOT / "scripts" / "seed_ollama.sh"
    env_model = args.ollama_model
    cmd = ["bash", str(script), env_model]
    env = {**os.environ, "OLLAMA_HOST": args.ollama_host, "OLLAMA_MODEL": env_model}
    print(_c(DIM, f"     $ OLLAMA_HOST={args.ollama_host} bash {script} {env_model}"))
    if args.dry_run:
        return True
    result = subprocess.run(cmd, text=True, env=env)
    return result.returncode == 0


def stage_knowledge_base(args: argparse.Namespace) -> bool:
    """Stage 2 - build ChromaDB knowledge base."""
    step("Building ChromaDB knowledge base")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_knowledge_base.py"),
        "--host", args.chroma_host,
    ]
    if args.rebuild_kb:
        cmd.append("--rebuild")
    return run(cmd, args.dry_run) == 0


def stage_train_bert(args: argparse.Namespace) -> bool:
    """Stage 3 - fine-tune BERT classifier."""
    step("Training BERT classifier (deepset/gbert-base)")
    cmd = [
        sys.executable,
        str(ROOT / "training" / "train_classifier.py"),
        "--data",       str(ROOT / "training" / "data" / "clause_labels.jsonl"),
        "--output",     str(ROOT / "training" / "bert_classifier"),
        "--epochs",     str(args.bert_epochs),
        "--batch-size", str(args.bert_batch),
    ]
    return run(cmd, args.dry_run) == 0


def stage_train_ner(args: argparse.Namespace) -> bool:
    """Stage 4 - train spaCy NER model."""
    step("Training spaCy NER model")
    cmd = [
        sys.executable,
        str(ROOT / "training" / "train_ner.py"),
        "--data",   str(ROOT / "training" / "data" / "ner_annotations.jsonl"),
        "--output", str(ROOT / "training" / "spacy_ner_model"),
        "--epochs", "30",
    ]
    return run(cmd, args.dry_run) == 0


def stage_export_bert(args: argparse.Namespace) -> bool:
    """Stage 5 - export fine-tuned BERT to ONNX."""
    step("Exporting BERT classifier to ONNX")
    bert_dir = ROOT / "training" / "bert_classifier"
    if not args.dry_run and not bert_dir.exists():
        print(_c(RED, f"     ERROR: {bert_dir} not found - run train-bert first"))
        return False
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_onnx.py"),
        "--model-path",  str(bert_dir),
        "--output-path", str(ROOT / "model_repository" / "bert_clause_classifier" / "1" / "model.onnx"),
        "--model-type",  "classifier",
    ]
    return run(cmd, args.dry_run) == 0


def stage_export_e5(args: argparse.Namespace) -> bool:
    """Stage 6 - export multilingual-e5-small to ONNX."""
    step("Exporting multilingual-e5-small to ONNX")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_onnx.py"),
        "--model-path",  "intfloat/multilingual-e5-small",
        "--output-path", str(ROOT / "model_repository" / "e5_embeddings" / "1" / "model.onnx"),
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


# -- Stage registry (ordered) --------------------------------------------------

STAGES: list[tuple[str, str, bool]] = [
    # (id, description, phase5_only)
    ("seed-ollama",    "Seed Ollama model",             False),
    ("knowledge-base", "Build ChromaDB knowledge base", False),
    ("train-bert",     "Train BERT classifier",         True),
    ("train-ner",      "Train spaCy NER",               True),
    ("export-bert",    "Export BERT to ONNX",           True),
    ("export-e5",      "Export e5-small to ONNX",       True),
    ("benchmark",      "Benchmark Triton vs Ollama",    True),
]

STAGE_FNS = {
    "seed-ollama":    stage_seed_ollama,
    "knowledge-base": stage_knowledge_base,
    "train-bert":     stage_train_bert,
    "train-ner":      stage_train_ner,
    "export-bert":    stage_export_bert,
    "export-e5":      stage_export_e5,
    "benchmark":      stage_benchmark,
}

# -- CLI -----------------------------------------------------------------------

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
    p.add_argument("--skip-train",     action="store_true", help="Skip BERT + NER training")
    p.add_argument("--skip-export",    action="store_true", help="Skip ONNX export")
    p.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark")
    p.add_argument("--skip-phase5",    action="store_true",
                   help="Skip all Phase 5 stages (train + export + benchmark)")

    # Behaviour
    p.add_argument("--dry-run",       action="store_true", help="Print commands without executing")
    p.add_argument("--rebuild-kb",    action="store_true", help="Force rebuild of ChromaDB collections")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Abort entire pipeline on first stage failure (default: continue)")

    # Service addresses
    p.add_argument("--ollama-host",  default="http://localhost:11434")
    p.add_argument("--chroma-host",  default="http://localhost:8001")
    p.add_argument("--triton-host",  default="localhost")
    p.add_argument("--triton-port",  type=int, default=8003)
    p.add_argument("--ollama-model", default="phi3:mini")

    # Training hyper-params
    p.add_argument("--bert-epochs",   type=int, default=5)
    p.add_argument("--bert-batch",    type=int, default=16)
    p.add_argument("--bench-samples", type=int, default=20,
                   help="Number of samples per backend in benchmark")

    return p.parse_args()


# -- Main ----------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    banner("KlarKI Setup Pipeline")

    if args.dry_run:
        print(_c(YELLOW, "  DRY RUN - commands printed, nothing executed\n"))

    # Build the list of stages to run
    skip_set: set[str] = set()
    if args.skip_seed:      skip_set.add("seed-ollama")
    if args.skip_kb:        skip_set.add("knowledge-base")
    if args.skip_train:     skip_set.update({"train-bert", "train-ner"})
    if args.skip_export:    skip_set.update({"export-bert", "export-e5"})
    if args.skip_benchmark: skip_set.add("benchmark")
    if args.skip_phase5:
        skip_set.update(s[0] for s in STAGES if s[2])  # phase5_only=True

    if args.only_stages:
        run_ids = list(args.only_stages)
    else:
        run_ids = [s[0] for s in STAGES if s[0] not in skip_set]

    # Print plan
    print(_c(BOLD, "  Stages to run:"))
    for sid, desc, p5 in STAGES:
        tag = _c(DIM, " [Phase 5]") if p5 else ""
        if sid in run_ids:
            print(f"    {_c(GREEN, '[x]')} {desc}{tag}")
        else:
            print(f"    {_c(DIM, '[ ]')} {desc}{tag}  {_c(DIM, '(skipped)')}")

    # Execute
    results: dict[str, tuple[bool | None, float]] = {}
    pipeline_start = time.time()

    for sid, desc, _ in STAGES:
        if sid not in run_ids:
            skip_msg(desc)
            results[sid] = (None, 0.0)
            continue

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
    print()
    colour = RED if n_failed else DIM
    print(f"  {_c(BOLD, f'{n_passed} passed')}, {_c(colour, f'{n_failed} failed')}  "
          + _c(DIM, f"({total:.1f}s total)"))


if __name__ == "__main__":
    main()
