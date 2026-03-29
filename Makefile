# KlarKI — three commands to rule them all
#
#   make setup    first-time init: start containers + seed Ollama + build ChromaDB
#   make up       start (or restart) all containers
#   make test     run the full test suite inside the API container
#
# Phase 5 extras (requires NVIDIA GPU):
#   make triton   full Phase 5 pipeline: train BERT + export ONNX + start Triton
#   make bench    latency benchmark Ollama vs Triton

.PHONY: setup up down test triton bench logs clean help

# ── Core three ────────────────────────────────────────────────────────────────

## First-time setup: start containers, pull Ollama model, seed ChromaDB
setup:
	@echo ""
	@echo "==> Starting containers..."
	docker compose up -d
	@echo ""
	@echo "==> Running setup pipeline (Ollama + ChromaDB)..."
	python scripts/setup.py --skip-phase5 --stop-on-error
	@echo ""
	@echo "==> Done. Open http://localhost to use KlarKI."

## Start (or restart) all containers
up:
	docker compose up -d

## Run the full test suite inside the API container
test:
	docker exec klarki-api python -m pytest /tests/ -v --tb=short --asyncio-mode=auto

# ── Phase 5 (GPU required) ────────────────────────────────────────────────────

## Train BERT, export ONNX models, start Triton, enable Triton backend
triton:
	@echo "==> Running Phase 5 pipeline (train + export + start Triton)..."
	python scripts/setup.py --only train-bert --only train-ner \
	                        --only export-bert --only export-e5 \
	                        --stop-on-error
	@echo ""
	@echo "==> Starting Triton container..."
	docker compose --profile triton up -d klarki-triton
	@echo ""
	@echo "==> Enabling Triton backend in .env..."
	sed -i 's/USE_TRITON=false/USE_TRITON=true/' .env
	docker compose up -d klarki-api
	@echo ""
	@echo "==> Triton is live. Run 'make bench' to compare latency."

## Benchmark Ollama vs Triton (both must be running)
bench:
	python scripts/benchmark_triton.py --n-samples 50

# ── Helpers ───────────────────────────────────────────────────────────────────

## Stop all containers
down:
	docker compose --profile triton down

## Tail API logs
logs:
	docker compose logs -f klarki-api

## Remove containers, volumes, and ChromaDB data (full wipe)
clean:
	@echo "WARNING: this deletes all uploads, ChromaDB data, and Ollama models."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	docker compose --profile triton down -v
	rm -rf chroma_data/

## Show this help
help:
	@echo ""
	@echo "  KlarKI — available commands"
	@echo ""
	@grep -E '^## ' Makefile | sed 's/^## /    /'
	@echo ""
	@echo "  Phase 5 extras require an NVIDIA GPU and Docker nvidia runtime."
	@echo ""
