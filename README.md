# KlarKI — EU AI Act + GDPR Compliance Auditor

Local-first compliance auditing for AI systems. Upload your policy documents and get a scored gap analysis against **EU AI Act Articles 9–15** and **GDPR** — entirely on your own machine. No data leaves your network.

---

## What it does

Upload a PDF, DOCX, or plain-text policy document → KlarKI:

1. Parses and chunks the document into sections
2. Classifies each section against the 7 EU AI Act article domains
3. Retrieves matching regulatory requirements from a local vector database (ChromaDB)
4. Runs a structured gap analysis using a local LLM
5. Scans for Article 5 prohibited uses (emotion recognition in workplaces/schools)
6. Returns per-article compliance scores (0–100), identified gaps, and recommendations
7. Generates a downloadable PDF report — the report header shows which classifier backend was used

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Docker Desktop | 4.x+ | With Docker Compose v2 |
| Disk space | ~6 GB | ~2.3 GB Ollama model + ~3.5 GB images |
| RAM | 8 GB+ | |
| VRAM | 4 GB+ | Optional — GPU accelerates Ollama; CPU fallback works |

---

## Three commands

```bash
./run.sh setup    # First-time: start containers + pull Ollama model + seed ChromaDB
./run.sh up       # Day-to-day: start (or restart) all containers
./run.sh test     # Run the full test suite inside the API container
```

> On Linux/Mac you can also use `make setup`, `make up`, `make test`.

---

## First-time setup (detailed)

### 1. Clone and configure

```bash
git clone <repo-url>
cd klarki
cp .env.example .env
```

The defaults in `.env` work out of the box. No API keys required.

### 2. Run setup

```bash
./run.sh setup
```

This does everything in one shot:
- Starts all containers (`klarki-api`, `klarki-chromadb`, `klarki-ollama`, `klarki-frontend`)
- Pulls the Phi-3 Mini model into Ollama (~2.3 GB, one-time)
- Seeds ChromaDB with EU AI Act + GDPR regulatory text (~2 min, one-time)

### 3. Open the app

Go to **http://localhost**

Upload a document or paste policy text → click **Start Audit** → view your compliance dashboard.

---

## Containers

| Container | Role | Port |
|---|---|---|
| `klarki-frontend` | React UI (Nginx) | 80 |
| `klarki-api` | FastAPI backend | 8000 |
| `klarki-chromadb` | Vector database | 8001 |
| `klarki-ollama` | Local LLM server | 11434 |

---

## Classifier backends

Every compliance report shows which backend classified the document. There are two:

### Ollama / phi3:mini (default)
- Enabled when `USE_TRITON=false` (the default)
- Uses Phi-3 Mini 3.8B (Q4) running inside the `klarki-ollama` container
- Works on CPU and GPU
- Handles both German and English documents well
- ~2–4 seconds per chunk

### Triton / gbert-base (Phase 5, GPU recommended)
- Enabled when `USE_TRITON=true`
- Uses a fine-tuned `deepset/gbert-base` BERT model exported to ONNX
- Served via NVIDIA Triton Inference Server (batched, gRPC)
- ~50–100× faster than Ollama per chunk
- **German-first:** gbert-base is trained on German text. English documents are still classified correctly (the model was fine-tuned on mixed DE/EN data) but the Ollama backend is more balanced for English-heavy documents.
- Requires training and export before use (see Phase 5 below)

> The `classifier_backend` field appears in the dashboard stats and in the PDF report header so you always know how a report was generated.

---

## What BERT trains on

The BERT classifier is fine-tuned on **240 fixed, generic EU AI Act examples** in `training/data/clause_labels.jsonl` — 30 per article domain, mixed German and English. These examples teach the model to recognise which EU AI Act article a text chunk belongs to.

**Company documents are never used as training data.** They are the *input* to the trained classifier, not the training corpus. The 240 examples are the same for every KlarKI installation.

---

## Phase 5: GPU-accelerated Triton backend

```bash
./run.sh triton
```

This runs the full Phase 5 pipeline:
1. Fine-tunes `deepset/gbert-base` on `training/data/clause_labels.jsonl`
2. Trains the spaCy NER model on `training/data/ner_annotations.jsonl`
3. Exports both models to ONNX
4. Starts the Triton container
5. Switches `.env` to `USE_TRITON=true` and restarts the API

**Requires:** NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

To run the latency benchmark afterwards:
```bash
./run.sh bench
```

To revert to Ollama: set `USE_TRITON=false` in `.env` and run `./run.sh up`.

---

## Project structure

```
klarki/
├── run.sh                # Setup / up / test / triton / bench in one place
├── Makefile              # Same commands for Linux/Mac (make setup, make test)
├── api/
│   ├── routers/          # HTTP endpoints (audit, reports, wizard)
│   ├── services/         # Pipeline: parse → chunk → classify → RAG → score
│   ├── prompts/          # LLM prompt templates
│   └── templates/        # PDF report (WeasyPrint + Jinja2)
├── frontend/             # React 18 + TypeScript + Vite + Tailwind
│   └── src/
│       ├── pages/        # Upload, Dashboard, ArticleDetail, RiskWizard
│       └── components/   # ScoreRadial, ArticleCard, GapCard, EmotionWarning
├── scripts/
│   ├── setup.py          # Orchestrates all init stages
│   ├── build_knowledge_base.py
│   ├── export_onnx.py
│   └── benchmark_triton.py
├── training/
│   ├── train_classifier.py   # Fine-tune gbert-base (Phase 5)
│   ├── train_ner.py          # Train spaCy NER (Phase 5)
│   └── data/                 # 240 classifier examples + 25 NER examples
├── model_repository/         # Triton model configs (Phase 5)
├── tests/                    # pytest suite (36 tests)
└── docker-compose.yml
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health (chromadb + ollama status) |
| `POST` | `/api/v1/audit/upload` | Upload file or raw text, start audit |
| `GET` | `/api/v1/audit/{id}` | Get audit status + full report |
| `GET` | `/api/v1/audit/{id}/status` | Lightweight status poll |
| `GET` | `/api/v1/reports/{id}/json` | Download report as JSON |
| `GET` | `/api/v1/reports/{id}/pdf` | Download report as PDF |
| `GET` | `/api/v1/wizard/questions` | Annex III risk wizard questions |
| `POST` | `/api/v1/wizard/classify` | Submit wizard answers → risk tier |

Accepted file types: `.pdf`, `.docx`, `.txt`, `.md` — max 10 MB.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `phi3:mini` | LLM model (any Ollama-compatible model) |
| `USE_TRITON` | `false` | `true` to use BERT/Triton instead of Ollama |
| `UPLOAD_MAX_SIZE_MB` | `10` | Max upload file size |
| `DEBUG` | `true` | Pretty-print structured logs |
| `TRITON_HOST` | `klarki-triton` | Triton server hostname |
| `TRITON_GRPC_PORT` | `8001` | Triton gRPC port (inside Docker network) |

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI · Uvicorn · Pydantic v2 |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers multilingual-e5-small (local) |
| LLM (default) | Ollama · Phi-3 Mini 3.8B Q4 |
| LLM (Phase 5) | Triton Inference Server · gbert-base ONNX |
| NER (Phase 5) | spaCy · de_core_news_sm |
| Frontend | React 18 · TypeScript · Vite · Tailwind CSS |
| PDF parsing | PyMuPDF · python-docx |
| Reports | WeasyPrint · Jinja2 |
| Containers | Docker Compose |

---

## Privacy

- All processing is local — no data is sent to external APIs or cloud services
- Uploaded documents are deleted from disk immediately after the audit completes
- The LLM runs entirely inside the `klarki-ollama` container
- ChromaDB stores only regulatory text (EU AI Act + GDPR), never your documents
- The PDF report is generated locally and served directly to your browser

---

## Troubleshooting

**Audit fails / score is 0 on all articles**
→ The LLM model is not loaded. Run `./run.sh setup` or manually:
`docker exec klarki-ollama ollama pull phi3:mini`

**`ModuleNotFoundError: No module named 'tritonclient'` when `USE_TRITON=true`**
→ The API image was built before Phase 5 packages were added. Rebuild:
`docker compose up -d --build klarki-api`
Also ensure `USE_TRITON=false` if you haven't run `./run.sh triton` yet.

**PDF download fails**
→ Check `docker compose logs klarki-api` for WeasyPrint errors.
Ensure `pydyf==0.11.0` is in `requirements.txt` (0.12+ breaks WeasyPrint 62.x).

**Docker network error on `docker compose up`**
→ Stale network reference from a failed previous start. Fix:
`docker compose down --remove-orphans && docker network prune -f && docker compose up -d`

**Frontend changes not hot-reloading**
→ Vite uses polling on Windows Docker volumes (configured in `vite.config.ts`).
Changes reload within ~500 ms. If stuck, restart: `docker compose restart klarki-frontend`.

**GPU not detected by Ollama or Triton**
→ Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and restart Docker Desktop. Both services fall back to CPU automatically.

**ChromaDB connection error on startup**
→ ChromaDB takes ~10 s to initialise. The API retries on startup — wait a moment and refresh.
