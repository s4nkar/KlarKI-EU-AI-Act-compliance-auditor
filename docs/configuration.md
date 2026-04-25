# Configuration Guide

Every knob in KlarKI — runtime settings, training parameters, optional features, and how to change them.

---

## Runtime configuration (`.env`)

Copy `.env.example` to `.env` before first run. All values have safe defaults.

```env
# ── API ──────────────────────────────────────────────────────────
API_PORT=8000
API_HOST=0.0.0.0
DEBUG=false                   # true enables FastAPI debug mode + verbose logging

# ── LLM ─────────────────────────────────────────────────────────
OLLAMA_HOST=http://klarki-ollama:11434
OLLAMA_MODEL=phi3:mini        # Change to use a different model (see below)

# ── Vector DB ────────────────────────────────────────────────────
CHROMADB_HOST=http://klarki-chromadb:8000

# ── Embeddings ───────────────────────────────────────────────────
EMBEDDING_MODEL=intfloat/multilingual-e5-small

# ── File uploads ─────────────────────────────────────────────────
UPLOAD_MAX_SIZE_MB=10
UPLOAD_DIR=/data/uploads

# ── Triton GPU inference (optional) ──────────────────────────────
USE_TRITON=false              # Set to true by ./run.sh triton automatically
TRITON_HOST=klarki-triton
TRITON_GRPC_PORT=8001

# ── OpenSearch BM25 (optional) ───────────────────────────────────
USE_OPENSEARCH=false
OPENSEARCH_HOST=klarki-opensearch
OPENSEARCH_PORT=9200

# ── Frontend ─────────────────────────────────────────────────────
VITE_API_URL=http://localhost:8000
```

---

## Switching inference backends

### Enable Triton (GPU BERT inference, ~50–100× faster than Ollama for classification)

Requirements: NVIDIA GPU + NVIDIA Container Toolkit installed.

```bash
./run.sh triton
# Automatically:
#   1. Checks for nvidia-smi
#   2. Exports BERT + e5 to ONNX if not already done
#   3. Starts klarki-triton container
#   4. Sets USE_TRITON=true in .env
#   5. Restarts klarki-api

# To switch back to Ollama:
./run.sh up
# Sets USE_TRITON=false and restarts API in Ollama mode
```

What changes when Triton is enabled:
- Chunk classification: goes from Ollama few-shot (~5–10s/chunk) to Triton ONNX batch (~50ms/batch-32)
- LangGraph gap analysis: still uses Ollama (LLM-based, Triton doesn't replace this)
- Embedding: e5-small ONNX served by Triton instead of CPU inference

### Enable OpenSearch (alternative BM25 backend)

Requirements: Docker running (no GPU needed).

```bash
# Start OpenSearch container
docker compose --profile opensearch up -d

# Index regulatory text into OpenSearch (keep ChromaDB for vector search)
python scripts/build_knowledge_base.py --opensearch

# Enable in .env
USE_OPENSEARCH=true

# Restart API to pick up the new setting
docker compose up -d klarki-api
```

What changes when OpenSearch is enabled:
- BM25 keyword search: goes from rank_bm25 in-memory to OpenSearch HTTP queries
- Vector search: stays in ChromaDB (unchanged)
- RRF merge, cross-encoder re-ranking: unchanged
- Metadata filtering: now server-side in OpenSearch (article_num + regulation + lang filters)

### Change the Ollama LLM model

```bash
# In .env:
OLLAMA_MODEL=llama3.2:1b       # smaller, faster on CPU
OLLAMA_MODEL=gemma2:2b          # good accuracy, moderate size
OLLAMA_MODEL=phi3:mini          # default — best balance for compliance tasks

# Pull the new model into Ollama:
docker exec klarki-ollama ollama pull llama3.2:1b

# Restart API:
docker compose up -d klarki-api
```

Note: `temperature=0, seed=42` is hardcoded in all Ollama calls for deterministic output. Do not change these without re-evaluating output quality.

---

## Training data size

All data generation uses Ollama and can be tuned with `--gen-per-class`.

### BERT classifier training data

Default: 400 examples per class × 8 classes × 2 languages = **6,400 total**

```bash
# Double the data (better accuracy, takes ~2× longer to generate)
python scripts/setup.py --only generate-data --gen-per-class 800 --gen-overwrite

# Minimal smoke test (fast, low quality — for development only)
python scripts/setup.py --only generate-data --gen-per-class 20 --gen-overwrite

# Then retrain BERT:
python scripts/setup.py --only train-bert
```

### Specialist classifiers training data (actor / risk / prohibited)

Default: same `--gen-per-class` value (defaults to 400).

```bash
# Change specialist data size independently
python scripts/setup.py --only generate-specialist-data --gen-per-class 300 --gen-overwrite
python scripts/setup.py --only train-specialist

# Or all together:
python scripts/setup.py --retrain --gen-per-class 600
```

### NER training data

Default: 5,000 template-generated records. No Ollama needed.

```bash
python scripts/setup.py --only generate-ner-data --ner-templates 10000 --gen-overwrite
python scripts/setup.py --only train-ner
```

---

## Training hyperparameters

### BERT and specialist classifiers

All use the same training script with the same defaults:

| Parameter | Default | Flag | Effect |
|---|---|---|---|
| Epochs | 12 | `--bert-epochs N` | More epochs = better fit but risk of overfitting; early stopping usually triggers at 5–8 |
| Batch size | 16 | `--bert-batch N` | Reduce to 8 if GPU OOM; increase to 32 for faster training |
| Learning rate | 2e-5 | (script only: `--lr`) | Start value; cosine decay applied |
| Max token length | 256 | (script only: `--max-length`) | Increase to 512 for longer documents; doubles memory |

```bash
# Via setup.py:
python scripts/setup.py --only train-bert --bert-epochs 20 --bert-batch 8

# Directly:
python training/scripts/train_classifier.py \
  --data training/data/clause_labels.jsonl \
  --output training/artifacts/bert_classifier \
  --epochs 20 --batch-size 8 --lr 3e-5 --max-length 512
```

### NER model

| Parameter | Default | Flag |
|---|---|---|
| Epochs | 60 | `--ner-epochs N` |
| Batch size | 32 | `--ner-batch N` |
| Early stop patience | 10 | `--ner-patience N` |

```bash
python scripts/setup.py --only train-ner --ner-epochs 80 --ner-patience 15
```

---

## Retrain strategies

```bash
# Smart retrain: only retrains if training data hash changed
./run.sh retrain

# Force retrain everything (always produces a new model version)
python scripts/setup.py --force-retrain

# Retrain with more data
python scripts/setup.py --retrain --gen-per-class 600 --ner-templates 8000

# Retrain only one model (skip data generation)
python scripts/setup.py --only train-bert
python scripts/setup.py --only train-specialist
python scripts/setup.py --only train-ner

# Regenerate data only (no retrain)
python scripts/setup.py --only generate-data --gen-overwrite
python scripts/setup.py --only generate-specialist-data --gen-overwrite

# Skip all ML stages (infrastructure-only setup)
python scripts/setup.py --skip-phase5
```

---

## Ollama GPU tuning

In `docker-compose.yml`, the Ollama service has these environment variables. GPU settings are only applied when the GPU overlay is active (`docker-compose.gpu.yml`).

```yaml
environment:
  - OLLAMA_GPU_LAYERS=20       # Layers offloaded to GPU. phi3:mini has 32 layers.
                                # 20 layers ≈ 1.8 GB VRAM (safe for 4 GB cards)
                                # Set to 32 for full GPU if you have ≥ 4 GB free
  - OLLAMA_MAX_LOADED_MODELS=1 # Keep 1 model in VRAM (memory pressure guard)
  - OLLAMA_NUM_CTX=2048        # Context window. 2048 is sufficient for compliance chunks.
```

To change these, edit `docker-compose.yml` and restart: `docker compose up -d klarki-ollama`

---

## Upload limits

```env
UPLOAD_MAX_SIZE_MB=10    # Maximum file size in MB (enforced in FastAPI route)
```

To change: edit `.env`, restart API (`docker compose up -d klarki-api`).

---

## Adding a new regulatory article

1. Add `data/regulatory/<regulation>/article_N.txt` with the standard header:
   ```
   ARTICLE: N
   REGULATION: eu_ai_act
   DOMAIN: your_new_domain

   === EN ===
   Article N text...

   === DE ===
   Artikel N text...
   ```

2. Add the domain in `scripts/generate_bert_training_data.py` → `DOMAIN_ARTICLE_MAP`

3. Add the new `ArticleDomain` enum value in `api/models/schemas.py`

4. Re-run: `./run.sh setup --retrain`

---

## Confidence thresholds

These are hardcoded in the services. Change them in source if needed:

| Threshold | File | Meaning |
|---|---|---|
| Actor ML confidence ≥ 0.80 | `actor_classifier.py` | Below this, use regex patterns |
| Risk ML confidence ≥ 0.85 | `applicability_engine.py` | Below this, ML doesn't augment patterns |
| Prohibited ML confidence ≥ 0.85 | `applicability_engine.py` | Below this, ML doesn't augment patterns |
| Annex I safety signals ≥ 2 | `applicability_engine.py` | Require 2+ signals to avoid false positives |
| Human review confidence < 0.70 | `compliance_scorer.py` | Below this, report flags for human review |

---

## Debug and logging

```env
DEBUG=true    # Enables FastAPI debug + structlog verbose output
```

View API logs in real time:
```bash
./run.sh logs
# or:
docker compose logs -f klarki-api
```

All structured logs use `structlog` and are JSON-formatted in production (`DEBUG=false`), human-readable in development (`DEBUG=true`).
