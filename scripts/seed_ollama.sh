#!/bin/bash
# seed_ollama.sh — Pull the Ollama model on first run.
#
# Called by docker-compose as a one-shot init step or run manually.
# Waits for Ollama to be ready before pulling the model.
#
# Usage:
#   ./scripts/seed_ollama.sh [model_tag]
#   model_tag defaults to $OLLAMA_MODEL or phi3:mini

set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MODEL="${1:-${OLLAMA_MODEL:-phi3:mini}}"
MAX_WAIT=60
WAIT_INTERVAL=3

echo "==> KlarKI Ollama seeder"
echo "    Host:  $OLLAMA_HOST"
echo "    Model: $MODEL"

# ── Wait for Ollama to be ready ───────────────────────────────────────────────
echo "==> Waiting for Ollama to be ready..."
elapsed=0
until curl -sf "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; do
  if [ "$elapsed" -ge "$MAX_WAIT" ]; then
    echo "ERROR: Ollama not ready after ${MAX_WAIT}s. Exiting."
    exit 1
  fi
  echo "    Waiting... (${elapsed}s elapsed)"
  sleep "$WAIT_INTERVAL"
  elapsed=$((elapsed + WAIT_INTERVAL))
done
echo "==> Ollama is ready."

# ── Check if model already pulled ────────────────────────────────────────────
if curl -sf "$OLLAMA_HOST/api/tags" | grep -q "\"$MODEL\""; then
  echo "==> Model '$MODEL' already present. Skipping pull."
  exit 0
fi

# ── Pull the model ────────────────────────────────────────────────────────────
echo "==> Pulling model '$MODEL'... (this may take several minutes)"
curl -sf -X POST "$OLLAMA_HOST/api/pull" \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"$MODEL\", \"stream\": false}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','done'))"

echo "==> Model '$MODEL' is ready."
