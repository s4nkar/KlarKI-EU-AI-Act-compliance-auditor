# KlarKI — EU AI Act + GDPR Compliance Auditor

Local-first compliance auditing for AI systems. Upload your policy documents and get a scored gap analysis against **EU AI Act Articles 9–15** and **GDPR** — entirely on your own machine. No data leaves your network.

---

## What it does

Upload a PDF, DOCX, or plain-text policy document → KlarKI:

1. Parses and chunks the document
2. Classifies each section against the 7 EU AI Act article domains
3. Retrieves the relevant regulatory requirements from a local vector database
4. Runs a structured gap analysis using a local LLM (Phi-3 Mini)
5. Returns per-article compliance scores (0–100), identified gaps, and actionable recommendations
6. Generates a downloadable PDF report

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Docker Desktop | 4.x+ | With Docker Compose v2 |
| VRAM | 4 GB+ | For Ollama GPU acceleration (CPU fallback works but is slower) |
| Disk space | ~6 GB | ~2.3 GB model + ~3.5 GB images |
| RAM | 8 GB+ | |

> **GPU:** An NVIDIA GPU is detected automatically. On CPU-only machines the audit runs slower (~10–15 min per document instead of ~2–5 min).

---

## Quick start

### 1. Clone and configure

```bash
git clone <repo-url>
cd klarki
cp .env.example .env
```

The defaults in `.env` work out of the box. No API keys required.

### 2. Start all services

```bash
docker compose up -d
```

This starts four containers:

| Container | Role | Port |
|---|---|---|
| `klarki-frontend` | React UI (Vite dev server) | 5173 |
| `klarki-api` | FastAPI backend | 8000 |
| `klarki-chromadb` | Vector database | 8001 |
| `klarki-ollama` | Local LLM server | 11434 |

Wait ~30 seconds for all containers to become healthy:

```bash
docker compose ps
```

### 3. Build the knowledge base (one-time)

Seeds ChromaDB with EU AI Act and GDPR regulatory text:

```bash
docker exec klarki-api python scripts/build_knowledge_base.py --host klarki-chromadb --port 8000
```

Takes ~2 minutes. Only needs to run once (data persists in the `chroma_data` volume).

### 4. Download the LLM model (one-time, ~2.3 GB)

```bash
docker exec klarki-ollama ollama pull phi3:mini
```

### 5. Open the app

Go to **http://localhost:5173**

Upload a document or paste policy text → click **Start Audit** → wait 2–5 minutes → view your compliance dashboard.

---

## Project structure

```
klarki/
├── api/                  # FastAPI backend
│   ├── routers/          # HTTP endpoints (audit, reports)
│   ├── services/         # Pipeline: parse → chunk → classify → RAG → score
│   ├── prompts/          # LLM prompt templates
│   └── templates/        # PDF report HTML template
├── frontend/             # React + TypeScript + Tailwind UI
│   └── src/
│       ├── pages/        # Upload, Dashboard, ArticleDetail
│       └── components/   # ScoreRadial, ArticleCard, GapCard, etc.
├── scripts/              # Knowledge base builder, model export
├── tests/                # pytest test suite
├── docker-compose.yml
└── .env.example
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health check |
| `POST` | `/api/v1/audit/upload` | Upload file or raw text, start audit |
| `GET` | `/api/v1/audit/{id}` | Get audit status + report |
| `GET` | `/api/v1/audit/{id}/status` | Lightweight status poll |
| `GET` | `/api/v1/reports/{id}/json` | Download report as JSON |
| `GET` | `/api/v1/reports/{id}/pdf` | Download report as PDF |

Accepted file types: `.pdf`, `.docx`, `.txt`, `.md` — max 10 MB.

---

## Running tests

```bash
docker exec klarki-api sh -c 'python -m pytest /tests/ -v --asyncio-mode=auto'
```

Expected: **30 passed, 10 skipped** (skipped = Phase 4/5 stubs not yet implemented).

---

## Useful commands

```bash
# View live API logs
docker compose logs -f klarki-api

# Rebuild knowledge base after regulatory text updates
docker exec klarki-api python scripts/build_knowledge_base.py --host klarki-chromadb --port 8000

# Stop everything
docker compose down

# Stop and wipe all data (vector DB + uploads)
docker compose down -v
```

---

## Environment variables

All variables are set in `.env`. Key ones:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `phi3:mini` | LLM model name (any Ollama-compatible model) |
| `UPLOAD_MAX_SIZE_MB` | `10` | Max upload file size |
| `DEBUG` | `true` | Enables structured pretty-print logs |
| `CHROMADB_HOST` | `http://klarki-chromadb:8000` | ChromaDB service URL |

---

## Tech stack

- **Backend:** FastAPI · ChromaDB · sentence-transformers (multilingual-e5-small) · Ollama (Phi-3 Mini Q4)
- **Frontend:** React 18 · TypeScript · Vite · Tailwind CSS
- **PDF parsing:** PyMuPDF · python-docx
- **Reports:** WeasyPrint · Jinja2
- **Containers:** Docker Compose (no Kubernetes, no cloud)

---

## Privacy

- All processing is local — no data is sent to external APIs
- Uploaded documents are deleted from disk after the audit completes
- The LLM runs entirely inside the `klarki-ollama` container
- ChromaDB stores only regulatory text (EU AI Act + GDPR), not your documents

---

## Troubleshooting

**Audit completes with score 0 on all articles**
→ The LLM model isn't loaded. Run `docker exec klarki-ollama ollama pull phi3:mini`.

**PDF download fails**
→ Check `docker compose logs klarki-api` for WeasyPrint errors. Ensure `pydyf==0.11.0` is installed (pinned in `requirements.txt`).

**ChromaDB connection error on startup**
→ Wait a few more seconds — ChromaDB takes ~10 s to initialise. The API retries automatically.

**Frontend changes not reflected**
→ Vite uses polling for file watching in Docker on Windows. Changes take up to 1 second to hot-reload. If HMR doesn't fire, restart: `docker compose restart klarki-frontend`.

**GPU not detected**
→ Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and restart Docker Desktop. The `docker-compose.yml` includes the GPU deploy block — it falls back to CPU if no GPU is found.
