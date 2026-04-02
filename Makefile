# KlarKI — convenience aliases for run.sh
# All targets delegate to run.sh — see that file for full documentation.
#
# First time:   make setup
# Day-to-day:   make up      (Ollama mode)  or  make triton  (Triton/BERT mode)
# Force retrain: make retrain

.PHONY: setup up triton retrain test test-local bench down logs clean help

help:
	@./run.sh help

setup:
	./run.sh setup

up:
	./run.sh up

triton:
	./run.sh triton

# Regenerate training data + retrain BERT + NER + re-export ONNX.
# Skips infra stages (Ollama pull, ChromaDB rebuild).
retrain:
	./run.sh retrain

test:
	./run.sh test

# Run tests locally (no Docker) — faster for development
test-local:
	cd api && python -m pytest ../tests/ -v --tb=short --asyncio-mode=auto \
		--html=../tests/reports/report.html --self-contained-html \
		--cov=. --cov-report=html:../tests/reports/coverage --cov-report=term-missing

bench:
	./run.sh bench

down:
	./run.sh down

logs:
	./run.sh logs

clean:
	./run.sh clean
