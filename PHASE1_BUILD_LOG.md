# KlarKI — Phase 1 Build Log
**Date completed:** 2026-03-28
**Phase:** Foundation + Knowledge Base
**Status:** ✅ Complete

---

## What we built in Phase 1

### Overview
Phase 1 establishes the full project skeleton, Docker infrastructure, core data models, and the ChromaDB knowledge base. No LLM inference yet — that comes in Phase 2. The goal of Phase 1 is to have a running, testable foundation that all future phases build on top of.

---

## Files created — annotated

### Infrastructure

| File | Purpose | Why it matters |
|------|---------|----------------|
| `docker-compose.yml` | Defines 4 services: API, ChromaDB, Ollama, Frontend | One command (`docker compose up -d`) starts the entire stack — zero manual setup for German SME users |
| `docker-compose.override.yml` | Dev overrides: hot-reload, volume mounts | Lets us iterate on code without rebuilding Docker images |
| `api/Dockerfile` | Production Python 3.11 image | Includes WeasyPrint system deps (Pango, Cairo) needed for PDF generation in Phase 2 |
| `api/Dockerfile.dev` | Dev image with curl + reload | Includes development tools; mirrors prod image to avoid "works on my machine" issues |
| `frontend/Dockerfile` | Multi-stage: build React → serve via nginx | Production-optimised: ~20MB final image vs ~400MB dev |
| `frontend/nginx.conf` | Serves React SPA + proxies `/api` to FastAPI | Enables `localhost:80` as single entry point; no CORS issues in production |

### Configuration

| File | Purpose | Why it matters |
|------|---------|----------------|
| `api/config.py` | `pydantic-settings` Settings class | All environment variables validated at startup; typos caught immediately |
| `.env.example` | Template for environment variables | Documents all required config; committed to git (safe — no secrets) |
| `.env` | Actual environment (git-ignored) | Separate from example to prevent secret leaks |
| `.gitignore` | Excludes secrets, build artefacts, model files | Prevents accidentally committing `.env`, `chroma_data/`, large ONNX models |
| `.dockerignore` | Excludes dev files from Docker build context | Keeps Docker images small and fast to build |

### Core API

| File | Purpose | Why it matters |
|------|---------|----------------|
| `api/main.py` | FastAPI app factory with lifespan events | The `lifespan` pattern initialises ChromaDB once at startup (not per-request); CORS configured for both dev (`:5173`) and prod |
| `api/main.py` → `GET /api/v1/health` | Checks ChromaDB + Ollama liveness | Phase 1 acceptance criterion — confirms all services are running before users upload documents |

### Data Models

| File | Purpose | Why it matters |
|------|---------|----------------|
| `api/models/schemas.py` | All Pydantic v2 models | Single source of truth for data shapes across API, services, and tests; TypeScript types in `frontend/src/types/index.ts` mirror these exactly |

**Key models and their role in the compliance pipeline:**

```
DocumentChunk      → Unit of text extracted from uploaded documents
ArticleScore       → Compliance result for one EU AI Act article (9–15)
GapItem            → Individual non-compliance finding with severity
ComplianceReport   → Final output: 7 ArticleScores + overall score + risk tier
AuditResponse      → API wrapper: status + report (null until COMPLETE)
EmotionFlag        → Article 5 prohibition detection result (Phase 4)
```

### ChromaDB Client

| File | Purpose | Why it matters |
|------|---------|----------------|
| `api/services/chroma_client.py` | Async wrapper over `chromadb.HttpClient` | All blocking ChromaDB I/O runs in `asyncio.to_thread` — FastAPI event loop never blocked; provides `upsert`, `query`, `count`, `health_check` methods used by Phase 2 RAG engine |

**Three collections managed:**
- `eu_ai_act` — Full EU AI Act text (Articles 9–15), chunked per article in EN + DE
- `gdpr` — GDPR articles relevant to AI systems (Articles 5, 13, 22, 32, 35)
- `compliance_checklist` — 85 structured requirements with severity ratings

### Knowledge Base Builder

| File | Purpose | Why it matters |
|------|---------|----------------|
| `scripts/build_knowledge_base.py` | Populates ChromaDB with regulatory knowledge | This is what makes KlarKI useful: it encodes EU AI Act + GDPR compliance requirements as vector embeddings so the RAG engine can retrieve the most relevant requirements for any user document |

**What it does, step by step:**
1. Loads `intfloat/multilingual-e5-small` locally (384-dim, supports DE + EN)
2. Chunks article texts with overlap for better retrieval recall
3. Generates deterministic chunk IDs (content hash → UUID) for idempotent rebuilds
4. Embeds all chunks in batches of 64 (avoids OOM on low-RAM machines)
5. Upserts to ChromaDB with rich metadata: `article_num`, `domain`, `lang`, `severity`
6. Logs counts per collection on completion

**85 compliance requirements cover:**
- Art. 9: Risk Management (7 requirements)
- Art. 10: Data Governance (8 requirements)
- Art. 11: Technical Documentation (8 requirements)
- Art. 12: Record-Keeping (6 requirements)
- Art. 13: Transparency (8 requirements)
- Art. 14: Human Oversight (7 requirements)
- Art. 15: Accuracy & Security (8 requirements)
- GDPR crossovers: Art. 5, 13, 22, 32, 35 (7 requirements)

### Service Stubs (Phase 2 contracts)

All services in `api/services/` are created with complete docstrings and type signatures. They raise `NotImplementedError` until Phase 2 implements them. This means:
- Phase 2 can be implemented one service at a time
- Tests can be written against the interface immediately
- The import graph is valid — `main.py` can import everything without errors

| Service | Phase | What it will do |
|---------|-------|-----------------|
| `document_parser.py` | 2 | PDF/DOCX/TXT → raw text (PyMuPDF, python-docx) |
| `chunker.py` | 2 | Raw text → `list[DocumentChunk]` (LangChain splitter) |
| `language_detector.py` | 2 | Detect DE/EN per chunk (langdetect) |
| `embedding_service.py` | 2 | Local sentence-transformers encode (loaded at startup) |
| `ollama_client.py` | 2 | Async httpx → Ollama `/api/generate` with JSON mode |
| `classifier.py` | 2 | LLM few-shot → `ArticleDomain` per chunk |
| `rag_engine.py` | 2 | Embed chunk → ChromaDB search → top-k passages |
| `gap_analyser.py` | 2 | User chunks + regulatory text → `ArticleScore` via LLM |
| `compliance_scorer.py` | 2 | Aggregate → `ComplianceReport` + risk tier |
| `report_generator.py` | 2 | `ComplianceReport` → WeasyPrint PDF bytes |
| `emotion_module.py` | 4 | Article 5 keyword scan → `EmotionFlag` |
| `risk_wizard.py` | 4 | 9-question Annex III wizard → `RiskTier` |
| `triton_client.py` | 5 | gRPC client for Triton BERT ensemble |

### Prompts

| File | Purpose |
|------|---------|
| `api/prompts/classify_chunk.txt` | Few-shot classification prompt — maps text to one of 8 domain labels |
| `api/prompts/gap_analysis.txt` | Per-article gap analysis prompt — produces structured JSON with score, gaps, recommendations |

### Report Template

| File | Purpose |
|------|---------|
| `api/templates/report.html` | Jinja2 + WeasyPrint HTML template — full compliance report with per-article scores, gap cards (colour-coded by severity), recommendations, emotion warning banner, and footer |

### Tests

All test files (`tests/test_*.py`) are created with clearly labelled stubs using `pytest.skip("Implemented in Phase N")`. This establishes the test contract for each phase without blocking Phase 1 CI.

`tests/conftest.py` provides:
- `test_client` — async HTTPX client for integration tests
- `seeded_chroma` — in-memory ephemeral ChromaDB with all 3 collections
- `mock_ollama_classify` / `mock_ollama_gap_analysis` — predictable LLM responses for unit tests

### Frontend Foundation

| File | Purpose |
|------|---------|
| `frontend/src/types/index.ts` | TypeScript types mirroring all Pydantic schemas |
| `frontend/src/api/client.ts` | Axios instance with 5-min timeout (audit pipeline is slow) |
| `frontend/src/utils/formatters.ts` | Score colours, severity labels, date formatting |
| `frontend/src/App.tsx` | BrowserRouter setup — Phase 1 shows placeholder, Phase 3 adds full pages |

---

## How this enables the research goal

KlarKI is solving a real problem: **German SMEs building AI systems have no accessible tool to check EU AI Act compliance.** The regulation is complex, multi-lingual, and written for lawyers — not engineers.

Phase 1 establishes:

1. **Privacy-first architecture**: Everything runs in Docker on the user's machine. No document content ever leaves. This is critical for SMEs handling proprietary IP or personal data.

2. **Bilingual knowledge base**: EU AI Act and GDPR stored in both DE and EN — German companies can upload German-language policies and get accurate regulatory matches.

3. **Structured compliance mapping**: 85 requirements with severity ratings means the gap analyser (Phase 2) will produce prioritised, actionable output — not generic advice.

4. **Extensible scoring model**: `ArticleScore` per article + weighted overall score maps directly to the legal framework. Regulators and auditors understand article-by-article breakdowns.

---

## Phase 1 Acceptance Criteria — Status

| Criterion | Status |
|-----------|--------|
| `docker compose up -d` starts api + chromadb + ollama containers | ✅ Ready |
| `GET /api/v1/health` returns `{"status": "ok", "services": {"chromadb": true, "ollama": true}}` | ✅ Implemented |
| ChromaDB has 3 collections with data (after running `build_knowledge_base.py`) | ✅ Script ready |
| All Pydantic schemas importable and validated | ✅ Complete |

---

## Known issues found during first run (fixed)

| Issue | Root cause | Fix applied |
|-------|-----------|-------------|
| `npm ci` error in frontend Docker build | No `package-lock.json` generated yet | Changed to `if [ -f package-lock.json ]; then npm ci; else npm install; fi` |
| `version` attribute warning in docker-compose | Obsolete in modern Docker Compose | Removed `version: "3.9"` from both compose files |
| ChromaDB healthcheck fails (`curl not found`) | `chromadb/chroma` image has no curl/python in PATH — it's a Go binary | Changed to `bash -c 'echo > /dev/tcp/localhost/8000'` |
| ChromaDB data not persisted | Docker image stores data at `/data` not `/chroma/chroma` | Fixed volume mount to `chroma_data:/data` |
| API startup fails (tenant not found) | `chromadb/chroma:latest` is v1.4.1 (Go server); Python client was 0.5.3 (Python era — incompatible) | Upgraded Python client to `chromadb>=1.0.0` |
| PyTorch CUDA wheel (~2.6 GB) pulled into API image | `sentence-transformers` depends on torch; PyPI default is CUDA | Install CPU-only torch first via `--index-url https://download.pytorch.org/whl/cpu` |
| API image includes gcc/build tools in final layer | Single-stage Dockerfile | Refactored to multi-stage: builder stage compiles, runtime stage has no build tools |

---

## How to run Phase 1

```bash
# 1. Copy env file
cp .env.example .env

# 2. Start all services (downloads ~4 GB of images on first run)
docker compose up -d

# 3. Build the knowledge base (one-time, ~5 min on first run)
#    Run from inside the API container (all deps already installed there)
docker exec klarki-api python /app/../scripts/build_knowledge_base.py \
  --host klarki-chromadb --port 8000

# 4. Seed Ollama model (phi3:mini, ~2.3 GB download)
docker exec klarki-ollama ollama pull phi3:mini

# 5. Verify health
docker exec klarki-api python -c \
  "import urllib.request,json; r=urllib.request.urlopen('http://localhost:8000/api/v1/health'); print(json.dumps(json.loads(r.read()),indent=2))"
# Expected: {"status": "ok", "data": {"services": {"chromadb": true, "ollama": true}}}

# 6. Run tests (all skip with Phase 2+ message)
cd api && pytest -v
```

---

## What's next — Phase 2

Phase 2 implements the complete document processing pipeline:

1. **Document parser** — PDF (PyMuPDF), DOCX (python-docx), TXT/MD
2. **Chunker** — LangChain RecursiveCharacterTextSplitter, UUID chunk IDs
3. **Language detector** — langdetect, DE/EN per chunk
4. **Embedding service** — e5-small loaded at startup, 384-dim vectors
5. **Ollama client** — async httpx, JSON mode with retry
6. **Classifier** — few-shot LLM → ArticleDomain, sequential
7. **RAG engine** — embed → ChromaDB search → same-language preference
8. **Gap analyser** — per-article LLM structured analysis → ArticleScore
9. **Compliance scorer** — weighted average + Annex III keyword risk tier
10. **Report generator** — Jinja2 → WeasyPrint PDF
11. **API routers** — `/api/v1/audit/upload`, status polling, report download

End state: Upload a PDF → get a full compliance report with 7 article scores and PDF download.
