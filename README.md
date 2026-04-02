# KlarKI — EU AI Act + GDPR Compliance Auditor

Local-first compliance auditing for AI systems. Assess your risk tier, upload your policy documents, and get a scored gap analysis against **EU AI Act Articles 9–15** and **GDPR** — entirely on your own machine. No data leaves your network.

---

## How it works

### Step 1 — Annex III Risk Assessment (wizard)

Answer 9 plain-language yes/no questions covering all Annex III categories. KlarKI classifies your AI system as **Prohibited**, **High**, **Limited**, or **Minimal** risk. This result is recorded alongside your audit for comparison — it does **not** affect the document audit scores.

### Step 2 — Document Audit

Upload a PDF, DOCX, or plain-text policy document → KlarKI:

1. Parses and chunks the document into sections
2. Classifies each section against 7 EU AI Act article domains
3. Retrieves matching regulatory requirements from a local vector database (ChromaDB)
4. Runs a structured gap analysis using a local LLM (**deterministic** — same document always produces the same output)
5. Scans for Article 5 prohibited uses (emotion recognition in workplaces / schools)
6. Returns per-article compliance scores (0–100), identified gaps, and recommendations
7. Generates a downloadable PDF report

### Step 3 — Audit Results Dashboard

| Panel | What it shows |
|---|---|
| Risk Tier Comparison | Wizard self-assessment vs document-derived tier — with an explicit note that the wizard result is informational only |
| Overall score | Weighted average across Articles 9–15 |
| 7 Article cards | Score + gap severity; card colour driven by **worst gap** (not the number) |

### Article Detail (click any card)

| Section | What it answers |
|---|---|
| **Why this score?** | LLM reasoning behind the score; colour follows worst gap severity |
| **Which regulation exactly?** | The actual ChromaDB-retrieved regulatory passages used in the gap analysis |
| **Can I defend this in an audit?** | Remediation checklist; verdict based on Critical/Major gap count |

### Classifier Metrics (`/metrics`)

View BERT classifier performance: macro F1, per-class precision/recall/F1, and a heat-map confusion matrix. Populated after training.

---

## Reproducibility

All LLM calls use `temperature=0`, `seed=42`, and `top_k=1` — same document input always produces the same classification and gap analysis output. Embeddings are deterministic by design and cached in memory (SHA-256 keyed) so repeated audit runs skip re-embedding already-seen chunks.

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Docker Desktop | 4.x+ | With Docker Compose v2 |
| Python | 3.10–3.12 | For setup scripts (not inside containers) |
| Disk space | ~8 GB | ~2.3 GB Ollama model + images + training deps |
| RAM | 8 GB+ | 16 GB recommended for training |
| VRAM | 4 GB+ | Optional — GPU recommended for BERT training; CPU fallback works |

---

## Quick start

```bash
git clone <repo-url>
cd klarki
cp .env.example .env

./run.sh setup    # Complete first-time init (see below)
```

Open **http://localhost** — complete the **Step 1: Risk Assessment** wizard, then click **Continue to Step 2: Upload Docs**.

---

## Commands

```bash
./run.sh setup     # Complete first-time init (see Setup Pipeline below)
./run.sh up        # Start containers in Ollama mode (day-to-day)
./run.sh triton    # Switch to Triton/BERT mode (requires prior setup)
./run.sh retrain   # Regenerate training data + retrain BERT + re-export ONNX
./run.sh test      # Run the full test suite inside the API container
./run.sh bench     # Latency benchmark: Ollama vs Triton
./run.sh down      # Stop all containers
./run.sh logs      # Tail API logs
./run.sh clean     # Full wipe (containers + volumes + ChromaDB data)
```

> Linux/Mac users can also use `make setup`, `make up`, `make test`, etc.

---

## Setup pipeline

`./run.sh setup` runs the full one-shot pipeline — no manual steps required. Re-running is **safe**: each long-running stage checks for existing outputs and skips itself automatically.

| # | Stage | What happens | Auto-skip condition |
|---|---|---|---|
| 1 | seed-ollama | Pulls `phi3:mini` into Ollama (~2.3 GB, one-time) | Model already present in Ollama |
| 2 | knowledge-base | Chunks EU AI Act + GDPR → embeds → stores in ChromaDB | Pass `--rebuild-kb` to force |
| 3 | generate-data | Generates 2,400 synthetic BERT training sentences via Ollama | `clause_labels.jsonl` already has enough rows (committed to git — skipped on fresh clone) |
| 4 | train-bert | Fine-tunes `deepset/gbert-base` (8-class classifier) | `bert_classifier/model.safetensors` exists |
| 5 | generate-ner-data | Generates ~320 annotated NER sentences via Ollama | `ner_annotations.jsonl` already has enough rows (committed to git — skipped on fresh clone) |
| 6 | train-ner | Trains spaCy NER model (blank German base, no lookup-data dep) | `spacy_ner_model/model-final/` exists |
| 7 | export-bert | Exports fine-tuned BERT → ONNX | `bert_clause_classifier/1/model.onnx` exists |
| 8 | export-e5 | Exports multilingual-e5-small → ONNX | `e5_embeddings/1/model.onnx` exists |
| 9 | benchmark | Latency comparison Ollama vs Triton | — |

To **force** any stage to re-run, use `./run.sh retrain` (stages 3–8) or pass flags directly:

```bash
./run.sh retrain                                       # regenerate data + retrain everything
python scripts/setup.py --only train-bert              # retrain BERT only
python scripts/setup.py --gen-overwrite --only generate-data  # regenerate BERT data only
python scripts/setup.py --retrain --gen-per-class 300  # retrain with more data
```

Each stage prints a progress header:
```
  +-- Stage 4/9  [########................] 3/9  ~12 min remaining
  |   Train BERT classifier
  +------------------------------------------------------
```

BERT training prints a per-epoch macro F1 summary. spaCy NER training shows per-epoch loss and F1.

**GPU recommended** for stages 4–9 (RTX 3050 Ti sufficient). Everything falls back to CPU — training stages will just be slower (~3× longer).

After `setup` completes you have two modes available:

```bash
./run.sh up       # Ollama mode   — USE_TRITON=false (default)
./run.sh triton   # Triton mode   — USE_TRITON=true  (requires NVIDIA GPU + Container Toolkit)
```

---

## Training data architecture

### Regulatory source files

The knowledge base and BERT training data are both derived from official EU law, stored as plain-text files in `data/regulatory/`:

```
data/regulatory/
  eu_ai_act/
    article_5.txt    Article 5  — Prohibited AI practices
    article_9.txt    Article 9  — Risk management system
    article_10.txt   Article 10 — Data and data governance
    article_11.txt   Article 11 — Technical documentation
    article_12.txt   Article 12 — Record-keeping / logging
    article_13.txt   Article 13 — Transparency
    article_14.txt   Article 14 — Human oversight
    article_15.txt   Article 15 — Accuracy, robustness, cybersecurity
  gdpr/
    article_5.txt    Article 5  — Data processing principles
    article_6.txt    Article 6  — Lawfulness of processing
    article_24.txt   Article 24 — Controller responsibility
    article_25.txt   Article 25 — Privacy by design
    article_30.txt   Article 30 — Records of processing activities
    article_35.txt   Article 35 — Data protection impact assessment
```

Each file contains both English and German text (`=== EN ===` / `=== DE ===` sections). To add a new article, create a `.txt` file in the same format and re-run:

```bash
python scripts/setup.py --only knowledge-base --rebuild
```

### Synthetic training corpus

Both training data files are committed to this repo so a fresh clone skips data generation entirely and goes straight to model training:

| File | Contents | Size |
|---|---|---|
| `training/data/clause_labels.jsonl` | 2,400 labelled sentences for BERT classifier | ~1.5 MB |
| `training/data/ner_annotations.jsonl` | ~320 NER-annotated sentences for spaCy | ~200 KB |

How the BERT corpus was generated:
- For each of the 8 label classes, the actual regulation text is injected into the Ollama prompt as a reference
- Ollama generates realistic compliance document sentences grounded in the official article language
- 150 examples per class × 8 classes × 2 languages (EN + DE) = 2,400 total
- Deduplication is applied at generation time

How the NER corpus was generated:
- 4 entity labels: `ARTICLE`, `OBLIGATION`, `RISK_TIER`, `REGULATION`
- Ollama generates sentences with entity spans; offsets are verified via `text.find()` (LLM offsets are never trusted)
- 40 sentences per label × 4 labels × 2 languages (EN + DE) = ~320 total

**Company documents are never used as training data.** They are the *input* to the trained classifier, not the corpus.

To regenerate both corpora from scratch (overwrites committed files):

```bash
./run.sh retrain
# or just the data generation stages:
python scripts/setup.py --gen-overwrite --only generate-data --only generate-ner-data
```

---

## Classifier backends

Every compliance report shows which backend classified the document.

### Ollama / phi3:mini (default)

- Active when `USE_TRITON=false`
- Phi-3 Mini 3.8B (Q4) inside the `klarki-ollama` container
- Works on CPU and GPU, no GPU required
- ~2–4 seconds per chunk
- Deterministic: `temperature=0`, `seed=42`, `top_k=1`

### Triton / gbert-base (Phase 5)

- Active when `USE_TRITON=true`
- Fine-tuned `deepset/gbert-base` exported to ONNX, served via NVIDIA Triton
- ~50–100× faster per chunk, GPU batched
- Trained on mixed DE/EN data; German-primary but handles English well
- Requires NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

View classifier performance at `/metrics` — macro F1, per-class precision/recall/F1, and confusion matrix.

---

## Containers

| Container | Role | Port |
|---|---|---|
| `klarki-frontend` | React UI (Nginx) | 80 |
| `klarki-api` | FastAPI backend | 8000 |
| `klarki-chromadb` | Vector database | 8001 |
| `klarki-ollama` | Local LLM server | 11434 |
| `klarki-triton` | Triton inference (Phase 5, `--profile triton`) | 8002/8003 |

---

## Project structure

```
klarki/
├── run.sh                        # All commands in one place
├── data/
│   └── regulatory/               # Official EU AI Act + GDPR source text (tracked in git)
├── api/
│   ├── routers/                  # HTTP endpoints (audit, reports, wizard, metrics)
│   ├── services/                 # Pipeline: parse → chunk → classify → RAG → score
│   │   ├── ollama_client.py      # Deterministic LLM calls (temp=0, seed=42)
│   │   └── embedding_service.py  # SHA-256 keyed in-memory embedding cache
│   ├── models/schemas.py         # All Pydantic models
│   ├── prompts/                  # LLM prompt templates (gap_analysis asks for score reasoning)
│   └── templates/report.html     # PDF report (WeasyPrint + Jinja2)
├── frontend/
│   └── src/
│       ├── pages/                # RiskWizard (Step 1) → Upload (Step 2) → Dashboard → ArticleDetail → ClassifierMetrics
│       └── components/           # ScoreRadial, ArticleCard (gap-severity colours), GapCard, EmotionWarning
├── scripts/
│   ├── setup.py                  # Full pipeline orchestrator with stage progress bars
│   ├── build_knowledge_base.py
│   ├── generate_training_data.py
│   ├── export_onnx.py
│   └── benchmark_triton.py
├── training/
│   ├── train_classifier.py       # Fine-tune gbert-base; per-epoch F1 progress callback
│   ├── train_ner.py              # spaCy NER; spacy.blank("de") — no spacy-lookups-data dep
│   ├── requirements-training.txt
│   ├── bert_classifier/          # Generated by setup — all output gitignored (reproducible via make retrain)
│   ├── spacy_ner_model/          # Generated by setup (model-final gitignored; metrics tracked)
│   └── data/
│       ├── clause_labels.jsonl   # Pre-built BERT corpus (2,400 examples, tracked in git)
│       └── ner_annotations.jsonl # Pre-built NER corpus  (~320 examples, tracked in git)
├── model_repository/             # Triton configs tracked (config.pbtxt, model.py); ONNX weights gitignored
└── tests/
    ├── conftest.py               # Shared fixtures (in-memory ChromaDB, mocked Ollama)
    ├── test_*.py                 # 12 test files, ~55 tests, all run offline (no live services)
    └── reports/                  # Generated by ./run.sh test — gitignored
```

---

## Testing

### Run the full test suite

```bash
./run.sh test          # inside the API Docker container (accurate, matches prod)
make test-local        # local Python env — no containers needed (faster iteration)
```

After the run, two HTML reports are written to `tests/reports/` (gitignored):

| Report | Open with |
|---|---|
| `tests/reports/report.html` | Browser — full pass/fail with tracebacks |
| `tests/reports/coverage/index.html` | Browser — line-level coverage highlighting |

To copy reports out of the Docker container:
```bash
docker cp klarki-api:/tests/reports ./tests/reports
```

### Test suite structure

| File | What it tests | Type |
|---|---|---|
| `test_api_audit.py` | Upload endpoint, file types, status codes | Integration |
| `test_api_reports.py` | Report retrieval, 404/409 states | Integration |
| `test_chunker.py` | Chunking, UUID assignment, index ordering | Unit |
| `test_parser.py` | PDF/DOCX/TXT/MD parsing, German encoding | Unit |
| `test_language_detector.py` | EN/DE detection, short-text fallback | Unit |
| `test_scorer.py` | Risk tier keywords, weighted score average | Unit |
| `test_gap_analyser.py` | Score clamping, empty-doc critical gap, LLM mock | Unit |
| `test_classifier.py` | `_parse_label`, Ollama path, error fallback, Triton path | Unit |
| `test_rag.py` | `_flatten_result`, top-k, language ranking, filter logic, **precision@3 accuracy** | Unit |
| `test_report_generator.py` | PDF bytes, template name, error propagation | Unit |
| `test_emotion_module.py` | Article 5 prohibition (workplace/education vs commercial) | Unit |
| `test_risk_wizard.py` | Annex III wizard PROHIBITED/HIGH/MINIMAL logic | Unit |

### RAG accuracy

`test_rag_precision_golden_dataset` measures **Precision@3** — for each of 7 golden queries (one per article 9–15), it checks whether the expected article appears in the top-3 retrieved regulatory passages. The test asserts ≥ 80% precision. No live infrastructure required — uses an in-memory ChromaDB.

### AB testing (Ollama vs Triton)

`./run.sh bench` runs a latency + accuracy comparison between the two classifier backends over 50 sample documents. This is a **product-level benchmark**, not a pytest test — use it to decide whether to switch to `USE_TRITON=true` in production.

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health (chromadb + ollama status) |
| `POST` | `/api/v1/audit/upload` | Upload file or raw text, start audit (accepts optional `wizard_risk_tier` form field) |
| `GET` | `/api/v1/audit/{id}` | Get audit status + full report |
| `GET` | `/api/v1/audit/{id}/status` | Lightweight status poll |
| `GET` | `/api/v1/reports/{id}/json` | Download report as JSON |
| `GET` | `/api/v1/reports/{id}/pdf` | Download report as PDF |
| `GET` | `/api/v1/wizard/questions` | Annex III risk wizard questions |
| `POST` | `/api/v1/wizard/classify` | Submit wizard answers → risk tier |
| `GET` | `/api/v1/metrics/classifier` | BERT classifier metrics (macro F1, per-class P/R/F1, confusion matrix) |

Accepted file types: `.pdf`, `.docx`, `.txt`, `.md` — max 10 MB.

---

## Report schema highlights

```jsonc
{
  "audit_id": "...",
  "risk_tier": "minimal",            // Document-derived, keyword scan
  "wizard_risk_tier": "high",        // Self-assessed via wizard (informational only)
  "overall_score": 58.0,
  "classifier_backend": "ollama/phi3:mini",
  "article_scores": [
    {
      "article_num": 9,
      "score": 75.0,
      "score_reasoning": "...",       // LLM explanation of score
      "regulatory_passages": [...],   // ChromaDB passages used in analysis
      "gaps": [{ "severity": "major", "title": "...", "description": "..." }],
      "recommendations": ["..."]
    }
  ]
}
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `phi3:mini` | LLM model name |
| `USE_TRITON` | `false` | `true` to use Triton/BERT instead of Ollama |
| `UPLOAD_MAX_SIZE_MB` | `10` | Max upload file size |
| `DEBUG` | `true` | Pretty-print structured logs |
| `TRITON_HOST` | `klarki-triton` | Triton server hostname |
| `TRITON_GRPC_PORT` | `8001` | Triton gRPC port (inside Docker network) |

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI · Uvicorn · Pydantic v2 · Python 3.11 |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers `multilingual-e5-small` (local, cached) |
| LLM (default) | Ollama · Phi-3 Mini 3.8B Q4 (deterministic: temp=0, seed=42) |
| BERT (Phase 5) | `deepset/gbert-base` fine-tuned · ONNX Runtime |
| Inference (Phase 5) | NVIDIA Triton Inference Server |
| NER (Phase 5) | spaCy · blank German model |
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
- PDF reports are generated locally and served directly to your browser

---

## Troubleshooting

**Audit fails / all scores are 0**
→ Ollama model not loaded. Run `./run.sh setup` or manually:
```bash
docker exec klarki-ollama ollama pull phi3:mini
```

**`USE_TRITON=true` but audit still fails**
→ ONNX models need to be trained first. Run `./run.sh setup` (full pipeline), then `./run.sh triton`.
If the API image is stale: `docker compose up -d --build klarki-api`

**spaCy NER training fails with `[E955] lexeme_norm`**
→ Already fixed: KlarKI uses `spacy.blank("de")` which does not require `spacy-lookups-data`.
If you see this error you are running an older version — pull the latest code.

**`ModuleNotFoundError: torch` during setup**
→ Multiple Python versions on Windows — pip installed torch into a different interpreter. Use:
```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r training/requirements-training.txt
```

**Docker network error on `docker compose up`**
→ Stale network from a failed previous start:
```bash
docker compose down --remove-orphans && docker network prune -f && docker compose up -d
```

**BERT training loss stuck / F1 below 0.5**
→ Not enough training data or too few epochs. Regenerate with more examples:
```bash
python scripts/setup.py --retrain --gen-per-class 300
```

**PDF download fails**
→ Check `docker compose logs klarki-api` for WeasyPrint errors.
Ensure `pydyf==0.11.0` in `requirements.txt` (0.12+ breaks WeasyPrint 62.x).

**Frontend changes not hot-reloading**
→ Vite uses polling on Windows Docker volumes. Changes reload within ~500 ms.
If stuck: `docker compose restart klarki-frontend`

**GPU not detected by Ollama or Triton**
→ Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and restart Docker Desktop.
Both services fall back to CPU automatically.

**ChromaDB connection error on startup**
→ ChromaDB takes ~10 s to initialise. The API retries on startup — wait and refresh.

**Same document gives slightly different results between runs**
→ Should not happen — all LLM calls use `temperature=0` and `seed=42`.
If you see variation, check that the Ollama container was fully restarted after a `./run.sh up`.

**Want to start completely fresh**
```bash
./run.sh clean    # wipes containers, volumes, ChromaDB data
./run.sh setup    # rebuild everything from scratch
```
