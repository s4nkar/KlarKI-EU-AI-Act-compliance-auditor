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

# Portable sed -i: macOS (BSD sed) requires an explicit backup extension with -i,
# GNU sed and Git Bash on Windows do not.
_sed_inplace() {
  if sed --version 2>/dev/null | grep -q GNU; then
    sed -i "$@"
  else
    sed -i '' "$@"
  fi
}

# Returns 0 if an NVIDIA GPU is reachable via nvidia-smi, 1 otherwise.
_has_nvidia_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1
}

# Run a command inside the training container (Python 3.11, isolated from host).
# Automatically uses the GPU overlay + CUDA torch when nvidia-smi is detected;
# falls back to CPU-only torch on machines without an NVIDIA GPU.
# Build args (USE_GPU) come from the compose file, not the run command, so we
# build explicitly first and then run without --build.
_run_training() {
  local compose_files="-f docker-compose.yml"
  if _has_nvidia_gpu && [ -f docker-compose.gpu.yml ]; then
    compose_files="-f docker-compose.yml -f docker-compose.gpu.yml"
  fi
  docker compose $compose_files --profile training build klarki-training
  docker compose $compose_files --profile training run --rm klarki-training "$@"
}

# Build the docker compose file list: base + GPU overlay when GPU is present.
_compose_files() {
  if _has_nvidia_gpu && [ -f docker-compose.gpu.yml ]; then
    echo "-f docker-compose.yml -f docker-compose.gpu.yml"
  else
    echo "-f docker-compose.yml"
  fi
}

# ── Commands ──────────────────────────────────────────────────────────────────

cmd_setup() {
  COMPOSE=$(_compose_files)
  if echo "$COMPOSE" | grep -q gpu; then
    info "NVIDIA GPU detected — Ollama will use GPU acceleration."
  else
    info "No GPU detected — running in CPU-only mode (Ollama on CPU, slower but fully functional)."
  fi

  info "Building and starting infrastructure containers..."
  # shellcheck disable=SC2086
  docker compose $COMPOSE up --build -d

  info "Running setup pipeline inside training container (Python 3.11, isolated from host)..."
  info "ONNX export and Triton benchmark are skipped — run './run.sh triton' for GPU inference."
  _run_training python scripts/setup.py --stop-on-error --skip-export --skip-benchmark

  success "Setup complete!"
  echo ""
  echo "    ./run.sh up      — start in Ollama mode (default, works without GPU)"
  echo "    ./run.sh triton  — switch to Triton BERT mode (NVIDIA GPU required)"
}

cmd_up() {
  # Ensure USE_TRITON is off for plain Ollama mode
  if grep -q "USE_TRITON=true" .env 2>/dev/null; then
    _sed_inplace 's/USE_TRITON=true/USE_TRITON=false/' .env
    echo "Switched to Ollama mode (USE_TRITON=false)"
  fi

  COMPOSE=$(_compose_files)
  if echo "$COMPOSE" | grep -q gpu; then
    echo "NVIDIA GPU detected — Ollama will use GPU acceleration."
  else
    echo "No GPU detected — running in CPU-only mode."
  fi
  # shellcheck disable=SC2086
  docker compose $COMPOSE up --build -d
}

cmd_dev() {
  # Development mode: hot reload, source volumes, Dockerfile.dev, port 3000 for frontend.
  # Uses docker-compose.dev.yml overlay on top of the production compose file.
  if [ ! -f docker-compose.dev.yml ]; then
    abort "docker-compose.dev.yml not found."
  fi
  if grep -q "USE_TRITON=true" .env 2>/dev/null; then
    _sed_inplace 's/USE_TRITON=true/USE_TRITON=false/' .env
  fi

  COMPOSE=$(_compose_files)
  # shellcheck disable=SC2086
  docker compose $COMPOSE -f docker-compose.dev.yml up --build -d
  echo ""
  echo "Dev mode: API hot-reload on :8000, frontend on :3000"
}

cmd_retrain() {
  info "Retraining inside training container (Python 3.11)..."
  info "(Skips: Ollama pull, ChromaDB rebuild — pass --only <stage> for finer control)"
  _run_training python scripts/setup.py --retrain --stop-on-error

  success "Retrain complete. Run './run.sh triton' to switch to the new model."
}

cmd_triton() {
  # GPU is required for Triton — fail fast with a clear message.
  if ! _has_nvidia_gpu; then
    abort "No NVIDIA GPU detected. Triton requires a CUDA-capable GPU with the NVIDIA Container Toolkit installed.
       Use './run.sh up' for CPU/Ollama mode — it works on all hardware."
  fi

  # Export ONNX models if not already done (first-time Triton setup).
  if [ ! -f "model_repository/bert_clause_classifier/1/model.onnx" ]; then
    info "BERT ONNX model not found — exporting inside training container..."
    _run_training python scripts/setup.py --only export-bert --only export-e5 --stop-on-error \
      || abort "ONNX export failed. Run './run.sh setup' first to train the models."
  fi

  info "Starting Triton inference server..."
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile triton up -d --build klarki-triton

  info "Enabling Triton backend in API..."
  _sed_inplace 's/USE_TRITON=false/USE_TRITON=true/' .env
  docker compose $(_compose_files) up -d --build klarki-api

  success "Triton is live. Run './run.sh bench' to compare latency."
}

cmd_test() {
  # Ensure report plugins are present (no-op if already installed)
  MSYS_NO_PATHCONV=1 docker exec klarki-api pip install -q pytest-html==4.1.1 pytest-cov==5.0.0

  MSYS_NO_PATHCONV=1 docker exec klarki-api mkdir -p /tests/reports

  MSYS_NO_PATHCONV=1 docker exec klarki-api python -m pytest /tests/ -v --tb=short --asyncio-mode=auto \
    --html=/tests/reports/report.html --self-contained-html \
    --cov=. --cov-report=html:/tests/reports/coverage --cov-report=term-missing

  echo ""
  echo "Test report : tests/reports/report.html"
  echo "Coverage    : tests/reports/coverage/index.html"
}

cmd_bench() {
  _run_training python scripts/benchmark_triton.py --n-samples 50
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

    ./run.sh setup    Complete init: containers + models + knowledge graph + training.
                      Works on all OS and hardware — GPU accelerates Ollama if detected.
                      Re-running is safe: long stages are skipped if outputs exist.

    ./run.sh up       Start containers in production mode (nginx, compiled images, no reload)
    ./run.sh dev      Start containers in dev mode (hot reload, source volumes, port 3000)
    ./run.sh triton   Switch to Triton BERT mode (requires NVIDIA GPU + Container Toolkit)
    ./run.sh test     Run full test suite inside the API container

    ./run.sh retrain  Force-regenerate data + retrain BERT + NER + re-export ONNX.
                      Use when you want fresh training (ignores existing outputs).
    ./run.sh bench    Benchmark Ollama vs Triton latency
    ./run.sh down     Stop all containers
    ./run.sh logs     Tail API logs
    ./run.sh clean    Full wipe (containers + volumes + ChromaDB data)
    ./run.sh help     Show this message

  First time:    ./run.sh setup    (does everything; GPU auto-detected for Ollama)
  Day-to-day:    ./run.sh up       (production mode — nginx, compiled images)
  Development:   ./run.sh dev      (hot reload, source volumes, frontend on :3000)
  GPU inference: ./run.sh triton   (Triton BERT — NVIDIA GPU required)
  Retrain only:  ./run.sh retrain  (skips Ollama pull and ChromaDB rebuild)

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
  dev)     cmd_dev     ;;
  triton)  cmd_triton  ;;
  retrain) cmd_retrain ;;
  test)    cmd_test    ;;
  bench)   cmd_bench   ;;
  down)    cmd_down    ;;
  logs)    cmd_logs    ;;
  clean)   cmd_clean   ;;
  help|*)  cmd_help    ;;
esac
