#!/usr/bin/env bash
# KlarKI runner — three commands to rule them all
#
#   ./run.sh setup    first-time init: start containers + seed Ollama + build ChromaDB
#   ./run.sh up       start (or restart) all containers
#   ./run.sh test     run the full test suite inside the API container
#
# Phase 5 extras (requires NVIDIA GPU):
#   ./run.sh triton   train BERT + export ONNX + start Triton
#   ./run.sh bench    latency benchmark Ollama vs Triton
#
# Other:
#   ./run.sh down     stop all containers
#   ./run.sh logs     tail API logs
#   ./run.sh clean    full wipe (containers + volumes + data)

set -e
CMD="${1:-help}"

# ── Helpers ───────────────────────────────────────────────────────────────────

info()    { echo ""; echo "==> $*"; }
success() { echo ""; echo "    OK: $*"; }
abort()   { echo ""; echo "ERROR: $*" >&2; exit 1; }

# ── Commands ──────────────────────────────────────────────────────────────────

cmd_setup() {
  info "Starting containers..."
  docker compose up -d

  info "Running setup pipeline (Ollama + ChromaDB)..."
  python scripts/setup.py --skip-phase5 --stop-on-error

  success "Done. Open http://localhost to use KlarKI."
}

cmd_up() {
  docker compose up -d
}

cmd_test() {
  docker exec klarki-api python -m pytest /tests/ -v --tb=short --asyncio-mode=auto
}

cmd_triton() {
  info "Running Phase 5 pipeline (train + export)..."
  python scripts/setup.py \
    --only train-bert \
    --only train-ner \
    --only export-bert \
    --only export-e5 \
    --stop-on-error

  info "Starting Triton container..."
  docker compose --profile triton up -d klarki-triton

  info "Enabling Triton backend..."
  sed -i 's/USE_TRITON=false/USE_TRITON=true/' .env
  docker compose up -d klarki-api

  success "Triton is live. Run './run.sh bench' to compare latency."
}

cmd_bench() {
  python scripts/benchmark_triton.py --n-samples 50
}

cmd_down() {
  docker compose --profile triton down
}

cmd_logs() {
  docker compose logs -f klarki-api
}

cmd_clean() {
  echo ""
  echo "WARNING: deletes all uploads, ChromaDB data, and Ollama models."
  read -r -p "Are you sure? [y/N] " confirm
  [ "$confirm" = "y" ] || abort "Cancelled."
  docker compose --profile triton down -v
  rm -rf chroma_data/
  success "Wiped."
}

cmd_help() {
  cat <<EOF

  KlarKI -- available commands

    ./run.sh setup    First-time init (containers + Ollama model + ChromaDB)
    ./run.sh up       Start (or restart) all containers
    ./run.sh test     Run full test suite inside the API container

    ./run.sh triton   Phase 5: train BERT, export ONNX, start Triton  [GPU]
    ./run.sh bench    Benchmark Ollama vs Triton latency               [GPU]

    ./run.sh down     Stop all containers
    ./run.sh logs     Tail API logs
    ./run.sh clean    Full wipe (containers + volumes + data)
    ./run.sh help     Show this message

  Phase 5 commands require an NVIDIA GPU and Docker nvidia runtime.

EOF
}

# ── Dispatch ──────────────────────────────────────────────────────────────────

case "$CMD" in
  setup)  cmd_setup  ;;
  up)     cmd_up     ;;
  test)   cmd_test   ;;
  triton) cmd_triton ;;
  bench)  cmd_bench  ;;
  down)   cmd_down   ;;
  logs)   cmd_logs   ;;
  clean)  cmd_clean  ;;
  help|*) cmd_help   ;;
esac
