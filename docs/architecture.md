# Architecture

## Container topology

```
Browser
   │
   ▼
klarki-frontend  (port 80)
   Nginx reverse proxy
   Serves React 18 SPA
   Proxies /api/* → klarki-api:8000
   │
   ▼
klarki-api  (port 8000)
   FastAPI + Uvicorn
   Python 3.11, single worker
   Holds all business logic and ML models in-process
   │
   ├──► klarki-chromadb  (port 8001 → internal 8000)
   │      ChromaDB vector database
   │      3 collections: eu_ai_act, gdpr, compliance_checklist
   │      Persistent via Docker volume chroma_data
   │
   ├──► klarki-ollama  (port 11434)
   │      Ollama LLM server (phi3:mini 3.8B Q4 by default)
   │      Used for: chunk classification + LangGraph gap analysis
   │      GPU-accelerated when NVIDIA GPU present (auto-detected by run.sh)
   │
   └──► klarki-triton  (ports 8002/8003/8004)  [optional — profile: triton]
          NVIDIA Triton Inference Server
          Serves BERT ONNX + e5 ONNX + spaCy Python backend
          Requires NVIDIA GPU + Container Toolkit
          Started with: ./run.sh triton

klarki-training  [ephemeral — profile: training]
   Python 3.11 container for all training jobs
   Mounts project root as /workspace
   Talks to klarki-ollama (data generation) and klarki-chromadb (knowledge base)
   Started with: docker compose --profile training run --rm klarki-training ...

klarki-opensearch  (port 9200)  [optional — profile: opensearch]
   OpenSearch full-text search engine
   Optional drop-in for BM25 (default is rank_bm25 in-memory)
   Started with: docker compose --profile opensearch up -d
```

## Storage layout

```
Docker volumes (persistent across restarts):
  chroma_data    ChromaDB index + embeddings
  ollama_data    Ollama model cache (phi3:mini ~2.2 GB)
  upload_data    Temporary user uploads (deleted after pipeline completes)

Host filesystem (mounted into containers):
  training/artifacts/
    bert_classifier/          BERT gbert-base fine-tuned weights
    actor_classifier/         Actor role classifier weights
    risk_classifier/          High-risk detector weights
    prohibited_classifier/    Prohibition detector weights
    spacy_ner_model/          spaCy NER model-final
  model_repository/
    bert_clause_classifier/1/model.onnx    BERT exported for Triton
    e5_embeddings/1/model.onnx             e5-small exported for Triton
  training/data/
    clause_labels.jsonl       BERT training data
    actor_labels.jsonl        Actor classifier training data
    risk_labels.jsonl         Risk classifier training data
    prohibited_labels.jsonl   Prohibited classifier training data
    ner_annotations.jsonl     NER training data
  data/
    regulatory/               Raw EU AI Act + GDPR text (EN + DE)
    obligations/              Machine-readable obligation schemas (JSONL)
```

## Request lifecycle

```
POST /api/v1/audit/upload
  │
  ├─ Save file to upload_data volume
  ├─ Return {audit_id} immediately
  └─ Launch _run_pipeline() as FastAPI BackgroundTask
        │
        ├─ status: PARSING
        │    parse_document()         PyMuPDF / python-docx / OCR
        │    proposition_chunk_text() heading-aware + enumeration splitting
        │    detect_language()        langdetect → "en" | "de"
        │
        ├─ status: CLASSIFYING
        │    classify_actor()         ML model + 39 regex patterns
        │    check_applicability()    4-step legal gate (deterministic)
        │    classify_chunks()        BERT via Ollama or Triton
        │    enrich_chunks_with_ner() spaCy NER entity extraction + domain correction
        │
        ├─ status: ANALYSING
        │    For each of 7 articles (asyncio.gather — all concurrent):
        │      retrieve_requirements()  hybrid RAG
        │      analyse_article()        LangGraph 3-node graph
        │
        ├─ status: SCORING
        │    map_evidence()           regex + NLI cross-encoder
        │    check_emotion_recognition()
        │    score_audit()            aggregate scores + confidence
        │
        └─ status: COMPLETE
             ComplianceReport stored in-memory (keyed by audit_id)
             Uploaded file deleted

GET /api/v1/audit/{id}       → full ComplianceReport (polls until COMPLETE)
GET /api/v1/reports/{id}/pdf → stream PDF bytes (WeasyPrint)
GET /api/v1/reports/{id}/json → ComplianceReport as JSON
```

## Network and ports

| Port (host) | Service | Purpose |
|---|---|---|
| 80 | klarki-frontend | React SPA + Nginx reverse proxy |
| 8000 | klarki-api | FastAPI backend (direct access) |
| 8001 | klarki-chromadb | ChromaDB HTTP API |
| 11434 | klarki-ollama | Ollama HTTP API |
| 8002 | klarki-triton | Triton HTTP API (optional) |
| 8003 | klarki-triton | Triton gRPC API (optional) |
| 9200 | klarki-opensearch | OpenSearch HTTP API (optional) |
| 3000 | klarki-frontend (dev) | Vite dev server (./run.sh dev only) |

All inter-service communication uses Docker internal DNS (service names as hostnames) on the `klarki-net` bridge network.

## Compose file hierarchy

```
docker-compose.yml          Base: all services, production Dockerfiles
docker-compose.gpu.yml      GPU overlay: adds NVIDIA reservation to Ollama
docker-compose.dev.yml      Dev overlay: hot reload, source volumes, Dockerfile.dev
```

Run modes:
- `./run.sh up`  → `docker-compose.yml` [+ `docker-compose.gpu.yml` if GPU detected]
- `./run.sh dev` → same + `docker-compose.dev.yml`
- `./run.sh triton` → same + explicitly starts Triton with GPU
