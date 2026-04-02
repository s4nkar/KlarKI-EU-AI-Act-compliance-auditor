#!/usr/bin/env bash
# KlarKI runner — all commands in one place.
#
# FIRST TIME
#   ./run.sh setup      Full init: containers + Ollama + ChromaDB + training pipeline.
#                       Smart skip conditions mean re-running is safe:
#                         - Ollama model already present?          skip pull
#                         - clause_labels.jsonl has enough rows?   skip BERT data gen
#                         - ner_annotations.jsonl has enough rows? skip NER data gen
#                         - bert_classifier/ weights exist?        skip BERT training
#                         - spacy_ner_model/model-final exists?    skip NER training
#                         - ONNX files exist?                      skip ONNX export
#
# DAY-TO-DAY
#   ./run.sh up         Start containers in Ollama mode  (USE_TRITON=false)
#   ./run.sh triton     Switch to Triton/BERT mode       (USE_TRITON=true, GPU required)
#
# RETRAINING
#   ./run.sh retrain    Force-regenerate training data + retrain BERT + NER + re-export ONNX.
#                       Skips infra stages (Ollama pull, ChromaDB rebuild).
#                       Equivalent to: python scripts/setup.py --retrain
#
#   To retrain with more data:
#     python scripts/setup.py --retrain --gen-per-class 300 --ner-per-label 60
#
#   To regenerate data only (no retrain):
#     python scripts/setup.py --only generate-data --gen-overwrite
#     python scripts/setup.py --only generate-ner-data --gen-overwrite
#
# OTHER
#   ./run.sh bench      Latency benchmark Ollama vs Triton
#   ./run.sh test       Run full test suite inside the API container
#   ./run.sh down       Stop all containers
#   ./run.sh logs       Tail API logs
#   ./run.sh clean      Full wipe (containers + volumes + ChromaDB data)

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

  info "Installing training dependencies..."
  python -m pip install -r training/requirements-training.txt -q \
    || abort "pip install failed. Check training/requirements-training.txt."

  info "Downloading spaCy German model..."
  python -m spacy download de_core_news_sm -q 2>/dev/null || true

  info "Running full setup pipeline (Ollama + ChromaDB + data gen + BERT training)..."
  python scripts/setup.py --stop-on-error

  success "Setup complete!"
  echo ""
  echo "    ./run.sh up      — start in Ollama mode (default)"
  echo "    ./run.sh triton  — switch to Triton BERT mode"
}

cmd_up() {
  # Ensure USE_TRITON is off for plain Ollama mode
  if grep -q "USE_TRITON=true" .env 2>/dev/null; then
    sed -i 's/USE_TRITON=true/USE_TRITON=false/' .env
    echo "Switched to Ollama mode (USE_TRITON=false)"
  fi
  docker compose up -d
}

cmd_retrain() {
  info "Installing training dependencies..."
  python -m pip install -r training/requirements-training.txt -q \
    || abort "pip install failed."

  info "Retraining: force-regenerate data + retrain BERT + NER + re-export ONNX..."
  info "(Skips: Ollama pull, ChromaDB rebuild — pass --only <stage> for finer control)"
  python scripts/setup.py --retrain --stop-on-error

  success "Retrain complete. Run './run.sh triton' to switch to the new model."
}

cmd_triton() {
  # Guard: ONNX model must exist (produced by setup)
  if [ ! -f "model_repository/bert_clause_classifier/1/model.onnx" ]; then
    abort "BERT ONNX model not found. Run './run.sh setup' first to train and export."
  fi

  info "Starting Triton inference server..."
  docker compose --profile triton up -d --build klarki-triton

  info "Enabling Triton backend in API..."
  sed -i 's/USE_TRITON=false/USE_TRITON=true/' .env
  docker compose up -d klarki-api

  success "Triton is live. Run './run.sh bench' to compare latency."
}

cmd_test() {
  MSYS_NO_PATHCONV=1 docker exec klarki-api python -m pytest /tests/ -v --tb=short --asyncio-mode=auto
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

    ./run.sh setup    Complete init: containers + models + training + ONNX export.
                      Re-running is safe: long stages are skipped if outputs exist.

    ./run.sh up       Start all containers in Ollama mode (day-to-day)
    ./run.sh triton   Switch to Triton BERT mode (requires prior setup + NVIDIA GPU)
    ./run.sh test     Run full test suite inside the API container

    ./run.sh retrain  Force-regenerate data + retrain BERT + NER + re-export ONNX.
                      Use when you want fresh training (ignores existing outputs).
    ./run.sh bench    Benchmark Ollama vs Triton latency
    ./run.sh down     Stop all containers
    ./run.sh logs     Tail API logs
    ./run.sh clean    Full wipe (containers + volumes + ChromaDB data)
    ./run.sh help     Show this message

  First time:   ./run.sh setup    (does everything; GPU recommended for training)
  Day-to-day:   ./run.sh up       (Ollama)  or  ./run.sh triton  (BERT/Triton)
  Retrain only: ./run.sh retrain  (skips Ollama pull and ChromaDB rebuild)

  Fine-grained control (see python scripts/setup.py --help):
    python scripts/setup.py --only train-bert          # retrain BERT only
    python scripts/setup.py --only generate-data --gen-overwrite  # regen BERT data
    python scripts/setup.py --retrain --gen-per-class 300         # more data

EOF
}

# ── Dispatch ──────────────────────────────────────────────────────────────────

case "$CMD" in
  setup)   cmd_setup   ;;
  up)      cmd_up      ;;
  triton)  cmd_triton  ;;
  retrain) cmd_retrain ;;
  test)    cmd_test    ;;
  bench)   cmd_bench   ;;
  down)    cmd_down    ;;
  logs)    cmd_logs    ;;
  clean)   cmd_clean   ;;
  help|*)  cmd_help    ;;
esac
