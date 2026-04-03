# KlarKI

**Local-first EU AI Act + GDPR compliance auditing.** Upload a policy document, get a scored gap analysis against EU AI Act Articles 9–15 and GDPR. No data leaves your machine.

---

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Commands](#commands)
- [Classifier backends](#classifier-backends)
- [API reference](#api-reference)
- [Testing](#testing)
- [Environment variables](#environment-variables)
- [Tech stack](#tech-stack)
- [Privacy](#privacy)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Licence](#licence)

---

## How it works

### Step 1 — Risk Assessment wizard

Answer 9 plain-language yes/no questions covering all Annex III categories. KlarKI classifies your system as **Prohibited**, **High**, **Limited**, or **Minimal** risk. This result is shown alongside your audit for reference — it does **not** influence document scores.

### Step 2 — Document audit

Upload a PDF, DOCX, or plain-text policy document. KlarKI:

1. Parses and chunks the document into sections
2. Classifies each section against 7 EU AI Act article domains
3. Retrieves matching regulatory requirements from a local ChromaDB vector database
4. Runs a structured gap analysis via a local LLM (deterministic - same input always produces the same output)
5. Scans for Article 5 prohibited uses (emotion recognition in workplaces / schools)
6. Returns per-article compliance scores (0–100), identified gaps, and remediation recommendations
7. Generates a downloadable PDF report

### Step 3 — Results dashboard

| Panel | What it shows |
|---|---|
| Risk tier comparison | Wizard self-assessment vs document-derived tier |
| Overall score | Weighted average across Articles 9–15 |
| Article cards (×7) | Score + gap severity; card colour driven by worst gap, not count |

Click any article card for:

- **Why this score?** — LLM reasoning behind the score
- **Which regulation exactly?** — The ChromaDB-retrieved regulatory passages used in the analysis
- **Can I defend this in an audit?** — Remediation checklist; verdict based on Critical/Major gap count

### Classifier metrics

`/metrics` shows evaluation Metricx of RAG, Spacy NER and gBERT classifier performance: macro F1, per-class precision/recall/F1, and a confusion matrix heatmap.

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Docker Desktop | 4.x+ | With Docker Compose v2 |
| Python | 3.10–3.12 | For setup scripts (not inside containers) |
| Disk space | ~8 GB | ~2.3 GB Ollama model + images + training artefacts |
| RAM | 8 GB minimum | 16 GB recommended |
| GPU (optional) | 4 GB VRAM | Recommended for BERT training; CPU fallback is automatic |

---

## Quick start

```bash
git clone https://github.com/s4nkar/KlarKI-EU-AI-Act-compliance-auditor.git
cd KlarKI-EU-AI-Act-compliance-auditor
cp .env.example .env

./run.sh setup    # First-time init — safe to re-run; completed stages are skipped automatically
```

Open **http://localhost:5173/**, complete the **Risk Assessment** wizard, then continue to **Upload Docs**.

> **Linux/Mac:** you can use `make` equivalents — `make setup`, `make up`, `make test`, etc.

---

## Commands

| Command | What it does |
|---|---|
| `./run.sh setup` | Full first-time setup (pull model, build knowledge base, train BERT + NER, export ONNX) |
| `./run.sh up` | Start containers using Ollama/phi3:mini (default day-to-day mode) |
| `./run.sh triton` | Start containers using Triton/BERT (requires NVIDIA GPU + Container Toolkit) |
| `./run.sh retrain` | Regenerate training data, retrain BERT + NER, re-export ONNX |
| `./run.sh test` | Run the full test suite inside the API container |
| `./run.sh bench` | Latency benchmark: Ollama vs Triton (standalone script, not part of the test suite) |
| `./run.sh down` | Stop all containers |
| `./run.sh logs` | Tail API logs |
| `./run.sh clean` | Full wipe — containers, volumes, and ChromaDB data |

`setup` is **idempotent** — each stage checks for its own outputs and skips itself if they exist. To force a full retrain from scratch: `./run.sh retrain`. To target a single stage:

```bash
python scripts/setup.py --only train-bert
python scripts/setup.py --only knowledge-base --rebuild-kb
python scripts/setup.py --retrain --gen-per-class 300    # retrain with more data
```

This runs the full Phase 2 pipeline:
1. Fine-tunes `deepset/gbert-base` on `training/data/clause_labels.jsonl`
2. Trains the spaCy NER model on `training/data/ner_annotations.jsonl`
3. Exports both models to ONNX
4. Starts the Triton container
5. Switches `.env` to `USE_TRITON=true` and restarts the API

### Ollama / phi3:mini (default)

Active when `USE_TRITON=false`. Phi-3 Mini 3.8B (Q4) runs inside the `klarki-ollama` container — no GPU required. All calls use `temperature=0`, `seed=42`, and `top_k=1`, so the same document always produces the same output.

### Triton / gbert-base

Active when `USE_TRITON=true`. A fine-tuned `deepset/gbert-base` model exported to ONNX, served via NVIDIA Triton. Roughly 50–100× faster per chunk, GPU-batched. Trained on mixed EN/DE data; German-primary but handles English well.

Requires: NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Run `./run.sh setup` before switching to Triton mode.

Every compliance report records which backend was used (`classifier_backend` field in the report JSON).

---

## Containers

| Container | Role | Port |
|---|---|---|
| `klarki-frontend` | React UI (Nginx) | 80 |
| `klarki-api` | FastAPI backend | 8000 |
| `klarki-chromadb` | Vector database | 8001 |
| `klarki-ollama` | Local LLM server | 11434 |
| `klarki-triton` | Triton inference (`--profile triton`) | 8002 (HTTP) / 8003 (gRPC) |

---

## API reference

The API is unauthenticated by design — it is intended for local use only and should not be exposed to untrusted networks.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health (ChromaDB + Ollama status) |
| `POST` | `/api/v1/audit/upload` | Upload file or raw text, start audit. Accepts optional `wizard_risk_tier` form field. |
| `GET` | `/api/v1/audit/{id}` | Get audit status + full report |
| `GET` | `/api/v1/audit/{id}/status` | Lightweight status poll |
| `GET` | `/api/v1/reports/{id}/json` | Download report as JSON |
| `GET` | `/api/v1/reports/{id}/pdf` | Download report as PDF |
| `GET` | `/api/v1/wizard/questions` | Annex III risk wizard questions |
| `POST` | `/api/v1/wizard/classify` | Submit wizard answers → risk tier |
| `GET` | `/api/v1/metrics/classifier` | BERT classifier metrics |

**Accepted file types:** `.pdf`, `.docx`, `.txt`, `.md` — max 10 MB.

### Report schema

```jsonc
{
  "audit_id": "...",
  "risk_tier": "minimal",           // Derived from document content (keyword scan)
  "wizard_risk_tier": "high",       // Self-assessed via wizard — informational only, not used in scoring
  "overall_score": 58.0,
  "classifier_backend": "ollama/phi3:mini",
  "article_scores": [
    {
      "article_num": 9,
      "score": 75.0,
      "score_reasoning": "...",      // LLM explanation of score
      "regulatory_passages": [...],  // ChromaDB passages used in analysis
      "gaps": [
        { "severity": "major", "title": "...", "description": "..." }
      ],
      "recommendations": ["..."]
    }
  ]
}
```

---

## Testing

```bash
./run.sh test          # Inside the API Docker container — matches production environment
make test-local        # Local Python env — faster for iteration, no containers needed
```

After a run, HTML reports are written to `tests/reports/` (gitignored):


Open `tests/reports/report.html` for pass/fail with tracebacks, or `tests/reports/coverage/index.html` for line-level coverage.

### Test coverage

| File | What it covers | Type |
|---|---|---|
| `test_api_audit.py` | Upload endpoint, file types, status codes | Integration |
| `test_api_reports.py` | Report retrieval, 404/409 states | Integration |
| `test_chunker.py` | Chunking, UUID assignment, index ordering | Unit |
| `test_parser.py` | PDF/DOCX/TXT/MD parsing, German encoding | Unit |
| `test_language_detector.py` | EN/DE detection, short-text fallback | Unit |
| `test_scorer.py` | Risk tier keywords, weighted score average | Unit |
| `test_gap_analyser.py` | Score clamping, empty-doc critical gap, LLM mock | Unit |
| `test_classifier.py` | Label parsing, Ollama path, error fallback, Triton path | Unit |
| `test_rag.py` | Flattening, top-k, language ranking, filter logic, Precision@3 | Unit |
| `test_report_generator.py` | PDF bytes, template name, error propagation | Unit |
| `test_emotion_module.py` | Article 5 prohibition (workplace/education vs commercial) | Unit |
| `test_risk_wizard.py` | Annex III PROHIBITED/HIGH/MINIMAL classification logic | Unit |

The RAG test asserts **≥ 80% Precision@3** across 7 golden queries (one per Article 9–15) using an in-memory ChromaDB — no live services required.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `phi3:mini` | LLM model name |
| `USE_TRITON` | `false` | Set to `true` to use Triton/BERT instead of Ollama |
| `UPLOAD_MAX_SIZE_MB` | `10` | Max upload file size in MB |
| `DEBUG` | `true` | Pretty-print structured logs |
| `TRITON_HOST` | `klarki-triton` | Triton server hostname (internal Docker network) |
| `TRITON_GRPC_PORT` | `8003` | Triton gRPC port (internal Docker network) |

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI · Uvicorn · Pydantic v2 · Python 3.11 |
| Vector DB | ChromaDB |
| Embeddings | `multilingual-e5-small` via sentence-transformers (local, SHA-256 cached) |
| LLM (default) | Ollama · Phi-3 Mini 3.8B Q4 |
| BERT classifier | `deepset/gbert-base` fine-tuned · ONNX Runtime |
| Inference server | NVIDIA Triton |
| NER | spaCy · blank German model (`spacy.blank("de")`) |
| Frontend | React 18 · TypeScript · Vite · Tailwind CSS |
| PDF parsing | PyMuPDF · python-docx |
| Reports | WeasyPrint · Jinja2 |
| Containers | Docker Compose |

---

## Privacy

- All processing runs locally — no data is sent to external APIs or cloud services
- Uploaded documents are deleted from disk immediately after the audit completes
- ChromaDB stores only regulatory text (EU AI Act + GDPR) — never your documents
- PDF reports are generated locally and served directly to your browser

---

## Troubleshooting

**Audit fails / all scores are 0**

Ollama model not loaded. Run `./run.sh setup`, or pull manually:
```bash
docker exec klarki-ollama ollama pull phi3:mini
```

**`USE_TRITON=true` but audit still fails**

ONNX models must be built first. Run `./run.sh setup` (full pipeline), then `./run.sh triton`. If the API image is stale: `docker compose up -d --build klarki-api`

**BERT training loss stuck / F1 below 0.5**

Not enough training data. Regenerate with more examples per class:
```bash
python scripts/setup.py --retrain --gen-per-class 300
```

**PDF download fails**

Check `docker compose logs klarki-api` for WeasyPrint errors. Ensure `pydyf==0.11.0` in `requirements.txt` — version 0.12+ breaks WeasyPrint 62.x.

**Docker network error on `docker compose up`**

Stale network from a failed previous start:
```bash
docker compose down --remove-orphans && docker network prune -f && docker compose up -d
```

**`ModuleNotFoundError: torch` during setup (Windows)**

Multiple Python versions — pip installed torch into a different interpreter. Use:
```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r training/requirements-training.txt
```

**GPU not detected by Ollama or Triton**

Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and restart Docker Desktop. Both services fall back to CPU automatically.

**ChromaDB connection error on startup**

ChromaDB takes ~10 s to initialise. The API retries on startup — wait and refresh.

**Same document gives different results between runs**

All LLM calls use `temperature=0` and `seed=42` — output should be identical. If you see variation, ensure the Ollama container was fully restarted after `./run.sh up`.

**Frontend changes not hot-reloading**

Vite uses polling on Windows Docker volumes — changes reload within ~500 ms. If stuck: `docker compose restart klarki-frontend`

**Start completely fresh**
```bash
./run.sh clean && ./run.sh setup
```

---

## Contributing

Training data architecture, model choices, and dataset generation details are documented in [`docs/training.md`](docs/training.md). Please open an issue before submitting large changes.

To report a **security vulnerability**, please do not open a public issue- email [s4nkar.connect@example.com] instead.

---

## Licence

[MIT](LICENSE) — see `LICENSE` for details.
