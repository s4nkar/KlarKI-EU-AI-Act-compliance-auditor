# Phase 2 Build Log — Document Pipeline + Ollama Integration

**Status:** Complete
**Tests:** 30 passed, 10 skipped (Phase 4/5 stubs), 0 failed
**Date:** 2026-03-28

---

## What Phase 2 delivers

A complete document-to-compliance-report pipeline. Upload a PDF, DOCX, or TXT file containing company policies/documentation, and the system:

1. Parses the file into raw text
2. Splits the text into overlapping chunks
3. Detects the language of each chunk (DE/EN)
4. Classifies each chunk against EU AI Act article domains using an LLM
5. Retrieves the most relevant regulatory requirements from ChromaDB via semantic search
6. Runs a structured gap analysis per article using the LLM
7. Aggregates scores, computes overall compliance score, and classifies the risk tier
8. Returns a full `ComplianceReport` JSON and a downloadable PDF

---

## Files built

### Services (`api/services/`)

#### `document_parser.py`
Dispatches to PyMuPDF for PDFs (page-by-page extraction), python-docx for DOCX (paragraphs + table cells), and plain UTF-8 reading for TXT/MD. Raises `ValueError` for unsupported extensions so the router can return HTTP 415 cleanly. Handles German characters (äöüß) correctly.

**Why it matters:** KlarKI must accept real company documents. German SMEs use all three formats; DOCX is especially common for policy manuals.

#### `chunker.py`
Uses LangChain's `RecursiveCharacterTextSplitter` (512 chars, 50 overlap) to split raw text into `DocumentChunk` objects with UUID4 IDs. The overlap prevents relevant sentences from being split across chunks and losing context.

**Why it matters:** ChromaDB stores and retrieves fixed-size chunks. The overlap ensures a requirement spanning two chunks isn't missed by the gap analyser.

#### `language_detector.py`
Runs `langdetect` on the first 500 characters of a chunk. Returns `'de'` or `'en'`; defaults to `'en'` on failure. Language is stored on each `DocumentChunk`.

**Why it matters:** KlarKI targets German SMEs, so DE documents are expected. The RAG engine prefers same-language regulatory passages for higher-quality gap analysis.

#### `embedding_service.py`
Loads `intfloat/multilingual-e5-small` (384-dim, CPU) once at FastAPI startup via the lifespan event. Wraps `model.encode()` in `asyncio.to_thread` to avoid blocking the event loop.

**Why it matters:** Multilingual E5 produces language-agnostic embeddings, so a German policy chunk and an English regulatory passage land in the same vector space — critical for cross-language retrieval.

#### `ollama_client.py`
Async `httpx` wrapper for Ollama's `/api/generate` endpoint. `generate_json()` uses `format='json'`, extracts JSON from the response text (handles cases where Ollama wraps JSON in markdown fences), and retries once on parse failure.

**Why it matters:** Phi-3 Mini occasionally wraps its JSON output in markdown. The extraction fallback makes the pipeline resilient to model formatting inconsistency without needing a larger (slower) model.

#### `classifier.py`
Reads `prompts/classify_chunk.txt` at module load. For each chunk, sends a few-shot classification prompt to Ollama and maps the response to `ArticleDomain`. Processes chunks sequentially — Ollama handles one request at a time. Logs progress every 10 chunks.

**Why it matters:** Classifying chunks by domain is the key step that routes each piece of company documentation to the correct Article (9-15) for gap analysis. Without accurate classification, a chunk about audit logs would be compared against risk management requirements, producing meaningless gaps.

#### `rag_engine.py`
Given a classified chunk, embeds it, queries both the `eu_ai_act` and `compliance_checklist` ChromaDB collections, and sorts results by: same language first, then cosine distance. Returns the top-k most relevant regulatory passages.

**Why it matters:** Grounding gap analysis in actual regulatory text (rather than relying on LLM world knowledge) produces specific, citation-backed gaps. The compliance_checklist collection adds 85 structured requirements (Article 9-15) for precise coverage.

#### `gap_analyser.py`
For each article domain, concatenates the relevant user chunks with retrieved regulatory passages and sends a structured prompt (`prompts/gap_analysis.txt`) to Ollama. Parses the JSON response into `ArticleScore`. Edge cases handled:

- **No chunks for an article** → score=0, single CRITICAL gap ("No documentation found"), LLM not called
- **LLM score out of range** → clamped to 0-100
- **Text truncated** → user_text and reg_text each capped at 3000 chars to fit within phi3:mini's 2048-token context

**Why it matters:** The gap analyser is the core value proposition of KlarKI. It produces the "what are you missing and how severe is it" output that an SME compliance officer acts on.

#### `compliance_scorer.py`
Aggregates `ArticleScore` objects into a `ComplianceReport`. Missing articles get `score=0`. Equal 1/7 weight per article for the `overall_score`. `classify_risk_tier()` scans chunk text for keywords:

- **PROHIBITED**: biometric mass surveillance, emotion recognition in schools/workplaces
- **HIGH**: biometric identification, medical diagnosis, recruitment/HR screening, credit scoring, critical infrastructure
- **LIMITED**: chatbot, recommendation, NLP
- **MINIMAL**: fallback

**Why it matters:** The risk tier drives the urgency framing of the report. A HIGH-risk system has strict Article 9-15 obligations; MINIMAL risk does not. Classifying this correctly from the documentation itself (without asking the user) is key to the "zero-friction" UX goal.

#### `report_generator.py`
Renders `templates/report.html` via Jinja2 and passes the result to WeasyPrint to produce a PDF byte stream. Runs in `asyncio.to_thread` to avoid blocking.

**Why it matters:** Compliance officers need a printable artefact to file with internal governance processes. PDF is the expected format for audit documentation in German enterprises.

### Routers (`api/routers/`)

#### `routers/audit.py`
Three endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/audit/upload` | POST | Accept file or raw_text, validate extension+size, save to disk, start background pipeline |
| `/api/v1/audit/{audit_id}` | GET | Return current `AuditResponse` (status + report if complete) |
| `/api/v1/audit/{audit_id}/status` | GET | Return just the `AuditStatus` enum value |

The pipeline runs as a `BackgroundTask` and transitions through status stages: `UPLOADING → PARSING → CLASSIFYING → ANALYSING → SCORING → COMPLETE`. Status is stored in an in-memory dict `_audits`.

**Why it matters:** The frontend polls the status endpoint every 2 seconds while the audit is running. The stage transitions let the UI show a step-by-step progress indicator so users know the system is working, not hung.

#### `routers/reports.py`
Two endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/reports/{audit_id}/pdf` | GET | Stream PDF bytes with `Content-Disposition: attachment` |
| `/api/v1/reports/{audit_id}/json` | GET | Return full `ComplianceReport` as JSON |

Returns 404 if audit not found, 409 if audit is still running, 500 if report is missing despite complete status.

### Prompts (`api/prompts/`)

#### `classify_chunk.txt`
Few-shot prompt with 5 examples mapping text snippets to domain labels. The examples cover each common pattern (adversarial testing → security, training data → data_governance, etc.) to anchor Phi-3 Mini's classification.

#### `gap_analysis.txt`
Structured prompt that presents user documentation and regulatory text side-by-side and requests a specific JSON schema. The schema is embedded in the prompt to ensure Phi-3 Mini produces parseable output.

### Report template (`api/templates/report.html`)
HTML template rendered by WeasyPrint. Sections: cover page (audit ID, date, risk tier badge), overall score, per-article score table, gap cards grouped by severity, recommendations list.

---

## Test results (Phase 2)

```
30 passed, 10 skipped in 7.72s
```

| Test file | Result | Coverage |
|---|---|---|
| test_api_audit.py | 6/6 pass | Health check, upload validation, raw text upload, 404 for unknown audit |
| test_api_reports.py | 3/3 pass | 404 for unknown, 409 for in-progress |
| test_chunker.py | 5/5 pass | UUID assignment, size, source file, sequential index, empty input |
| test_gap_analyser.py | 3/3 pass | No chunks → score=0 + critical gap, valid score, score clamping |
| test_language_detector.py | 4/4 pass | English, German, short-text fallback, non-DE/EN defaults to EN |
| test_parser.py | 4/4 pass | TXT, German chars, MD, unsupported extension |
| test_scorer.py | 5/5 pass | Prohibited/high/minimal tiers, weighted average, missing articles filled |
| test_classifier.py | 2 skipped | Need live Ollama — Phase 4/5 |
| test_rag.py | 2 skipped | Need live ChromaDB with data |
| test_emotion_module.py | 3 skipped | Phase 4 |
| test_risk_wizard.py | 3 skipped | Phase 4 |

---

## Known issues and limitations

1. **In-memory audit store** — `_audits` dict is lost on container restart. A KV store (Redis) or SQLite file would persist audits across restarts. Acceptable for Phase 2; the report JSON/PDF can be re-requested before restart.

2. **Sequential Ollama calls** — classifying 30 chunks takes ~3 minutes on phi3:mini at CPU speed. Ollama processes one request at a time (enforced by `OLLAMA_MAX_LOADED_MODELS=1`). Parallelism would cause OOM on a 4 GB VRAM card. Chunked batching or async queuing could help in Phase 5.

3. **PDF parsing quality** — PyMuPDF extracts text layer only. Scanned/image PDFs will produce empty text. OCR (Tesseract) is not included. German SMEs that scan their policies would get an empty audit.

4. **Context window** — phi3:mini at `OLLAMA_NUM_CTX=2048` means prompts are truncated at 3000 chars each (user + regulatory text). Longer documents are analysed on their first 3000 chars only. phi3:medium or a longer-context model would improve this.

---

## How to run the full pipeline end-to-end

```bash
# 1. Build knowledge base (once)
docker exec klarki-api python scripts/build_knowledge_base.py --host klarki-chromadb --port 8000

# 2. Pull Ollama model (once, ~2.3 GB download)
docker exec klarki-ollama ollama pull phi3:mini

# 3. Upload a document
curl -X POST http://localhost:8000/api/v1/audit/upload \
  -F "file=@my_ai_policy.pdf"
# returns: {"status":"success","data":{"audit_id":"<uuid>"}}

# 4. Poll status
curl http://localhost:8000/api/v1/audit/<uuid>/status

# 5. Download report
curl http://localhost:8000/api/v1/reports/<uuid>/pdf -o report.pdf
curl http://localhost:8000/api/v1/reports/<uuid>/json

# 6. Run tests
docker exec klarki-api sh -c 'python -m pytest /tests/ -v --asyncio-mode=auto'
```

---

## Architecture decisions

| Decision | Rationale |
|---|---|
| CPU torch in Docker image | Avoids pulling 2.6 GB CUDA wheel. E5-small inference is fast enough on CPU for embedding. |
| Multi-stage Dockerfile | Keeps gcc/build tools out of the runtime image. Runtime image is ~600 MB smaller. |
| Sequential LLM calls | phi3:mini on RTX 3050 Ti (4 GB VRAM) runs OOM with concurrent requests. Sequential is safer. |
| asyncio.to_thread for CPU work | Embedding and PDF rendering are CPU-bound. Offloading prevents event-loop stalls under concurrent HTTP requests. |
| JSON format in Ollama request | `format='json'` instructs Ollama to constrain the model's output tokens to valid JSON characters — higher parse success rate than asking in prompt alone. |
| In-memory audit store | Simplest approach for Phase 2. Redis/SQLite upgrade is clean and isolated to audit.py if needed later. |
