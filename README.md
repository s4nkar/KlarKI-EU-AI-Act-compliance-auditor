# KlarKI | EU AI Act + GDPR Compliance Auditor

> **Local-first. Privacy-preserving. Fully open-source.**
> Upload a policy document and receive a scored gap analysis against EU AI Act Articles 3–15 and GDPR entirely on your own hardware. No data ever leaves your machine.

---

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3-61dafb)](https://react.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-purple)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Open Source](https://img.shields.io/badge/Open%20Source-Collaborate-brightgreen)](https://github.com/s4nkar/KlarKI-EU-AI-Act-compliance-auditor)

---

## Contents

- [What KlarKI Does](#what-klarki-does)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Hardware & Model Choices](#hardware--model-choices)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Pretrained Models (Optional)](#pretrained-models-optional)
- [Commands](#commands)
- [Setup Stages](#setup-stages)
- [Classifier Backends](#classifier-backends)
- [Containers & Profiles](#containers--profiles)
- [API Reference](#api-reference)
- [Report Schema](#report-schema)
- [Environment Variables](#environment-variables)
- [Testing](#testing)
- [Training Pipeline](#training-pipeline)
- [Privacy](#privacy)
- [Troubleshooting](#troubleshooting)
- [Contributing & Collaboration](#contributing--collaboration)
- [Licence](#licence)

---

## What KlarKI Does

KlarKI is a **local-first EU AI Act + GDPR compliance auditor** for organisations assessing their AI systems against European regulatory frameworks. It is built for German SMEs as a primary audience but handles English documents throughout.

1. Complete a **9-question Annex III risk wizard** to self-assess your AI system's risk tier.
2. Upload a policy document (PDF, DOCX, TXT, or MD).
3. KlarKI runs a **fully async multi-agent audit pipeline** — parsing, actor detection, legal applicability gating, hybrid RAG retrieval, LangGraph gap analysis, and deterministic evidence mapping — entirely on-device.
4. Download a scored **PDF or JSON compliance report** with actor panel, applicability gate, evidence coverage, and per-article gap analysis.

---

## Key Features

### Legal Decision Hierarchy (Deterministic – No LLM)

| Feature | Detail |
|---|---|
| **Article 3 Actor Classification** | ML ensemble + 39 EN/DE regex patterns → Provider / Deployer / Importer / Distributor |
| **Article 5 Prohibition Detection** | 9 prohibited-practice patterns + specialist ML classifier (social scoring, emotion recognition, real-time biometric) |
| **Article 6 + Annex III Applicability Gate** | 4-step deterministic tree: prohibited → 8 Annex III categories (60+ patterns) → Annex I safety signals → ML augmentation |
| **3 Specialist ML Classifiers** | Separate fine-tuned gBERT models for actor detection, high-risk classification, and prohibited practice detection |
| **Short-Circuit for Non-Applicable Articles** | Articles 9–15 are skipped (score=100, zero LLM calls) for minimal-risk systems |

### Multi-Agent Gap Analysis (LangGraph)

| Feature | Detail |
|---|---|
| **3-Node LangGraph per Article** | `legal_agent → technical_agent → synthesis_agent` - runs concurrently across all applicable articles |
| **Hybrid RAG Retrieval** | BM25 + ChromaDB vector + Reciprocal Rank Fusion + cross-encoder re-ranking, filtered by article and regulation |
| **Optional OpenSearch BM25** | Drop-in server-side BM25 replacement (`USE_OPENSEARCH=true`, `--profile opensearch`) |
| **Deterministic Outputs** | `temperature=0, seed=42, top_k=1` on all LLM calls — same document always gives same result |

### Document Processing

| Feature | Detail |
|---|---|
| **Multi-format Parsing** | PDF (PyMuPDF), DOCX (python-docx), TXT, MD |
| **Table Extraction** | pdfplumber extracts PDF tables as structured text |
| **OCR Fallback** | pytesseract + Pillow for scanned/image-based PDFs (optional) |
| **Bilingual** | EN + DE document detection and classification throughout |
| **Proposition Chunking** | Heading-aware splitter that splits colon-introduced obligation lists into individual propositions (512 chars max, section metadata stored per chunk) |

### Evidence & Scoring

| Feature | Detail |
|---|---|
| **Deterministic Evidence Mapping** | 22 canonical evidence artefacts × synonym regex + NLI Cross-Encoder fallback |
| **Obligation Coverage** | Per-obligation evidence: fully satisfied / partially satisfied / missing |
| **Confidence Score** | Mean of actor confidence + evidence coverage + chunk classification ratio |
| **Human Review Flag** | Auto-triggered when confidence < 70% or actor is unknown |
| **Applicable-Articles-Only Scoring** | Overall score averages only applicable articles - minimal-risk systems score 100% |

### ML Training & Data Pipeline

| Feature | Detail |
|---|---|
| **BERT Domain Classifier** | Fine-tuned `deepset/gbert-base` (8 classes: Articles 9-15 + unrelated) |
| **spaCy NER** | 8 compliance entity types (ARTICLE, OBLIGATION, ACTOR, AI_SYSTEM, RISK_TIER, PROCEDURE, REGULATION, PROHIBITED_USE) |
| **Weak Supervision Pipeline** | Regex-based auto-labeling from regulatory text - no LLM, runs in seconds |
| **Synthetic Data Generation** | Async Ollama-grounded generation for BERT + specialist classifiers (bilingual EN/DE) |
| **ONNX Export for Triton** | BERT + e5-small exported to ONNX for GPU-accelerated Triton inference |
| **Version Management** | VersionManager tracks data, model, and metric versions for reproducibility |

### Infrastructure & Observability

| Feature | Detail |
|---|---|
| **6 Docker services** | API, ChromaDB, Ollama, Frontend, Triton (GPU opt-in), OpenSearch (opt-in) |
| **Structured Logging** | structlog with per-request context |
| **Prometheus-style Metrics** | `/monitoring` endpoint for real-time system health |
| **Classifier Metrics Dashboard** | `/metrics` - BERT, NER, specialist classifier F1 / precision / recall, confusion matrices |
| **6 Claude Code Agents** | code-reviewer, compliance-reviewer, architecture-reviewer, doc-generator, test-writer, agent-watcher |

---

## How It Works

### Step 1 - Risk Assessment Wizard

Answer 9 plain-language yes/no questions covering all Annex III categories. KlarKI uses a decision-tree classifier to determine if your AI system is **Prohibited**, **High**, **Limited**, or **Minimal** risk. This self-assessment is compared against the automated determination from the audit pipeline.

### Step 2 - Document Ingestion & Parsing

Upload a PDF, DOCX, TXT, or MD policy document (max 10 MB). The pipeline runs:

- **PyMuPDF** prose extraction → **pdfplumber** table extraction (tab-separated rows appended) → **pytesseract** OCR for scanned PDFs (optional, guarded by `ImportError`)
- **Legal-unit chunker** splits on headings first, then paragraphs, then sentences (512 chars, 50 overlap, UUID4 per chunk)
- **Language detection** (EN/DE via langdetect; fallback to EN for < 100 chars)
- **NER Phase 1 (entity extraction)** - spaCy `de_core_news_lg` extracts 8 entity types and writes them to `chunk.metadata["ner_entities"]`; runs **before** the legal gate so actor classification and applicability detection can read NER results
- **NER Phase 2 (domain correction)** - after BERT/Triton chunk classification, already-extracted entities are used to correct UNRELATED chunks that contain exactly one explicit Article 9–15 reference

### Step 3 - Legal Decision Hierarchy (Deterministic, Runs in Parallel)

No LLM is called in this stage. NER entities extracted in Step 2 are available to both tasks. Actor classification and applicability gate run concurrently via `asyncio.gather`:

**A. Actor Classification (Article 3)**
```
ML path  ──► predict_actor(text)  ──► confidence ≥ 0.80? ──► result
                                                           │
                                          No ─────────────►│
                                                           ▼
Pattern fallback ──► 39 EN+DE patterns (14 provider, 13 deployer, 6 importer, 6 distributor)
                 ──► NER AI_SYSTEM entities with first-person ownership prefix
                 │    ("our AI system", "unser KI-System") → additional PROVIDER signals
                 ──► confidence = matched_class / total_signals
                 ──► Default: DEPLOYER (most SMEs are deployers)

Output: ActorClassification(actor_type, confidence, matched_signals, reasoning)
```

**B. Applicability Gate (Article 6 + Annex III) — 4-Step Tree**
```
Step 1 (Article 5 Prohibited):
  9 regex patterns (subliminal techniques, social scoring, real-time biometric,
  emotion recognition in workplace/education)
  + NER PROHIBITED_USE entities → fed directly into Article 5 detection
  + ML predict_prohibited() at confidence ≥ 0.85
  → is_prohibited=True → applicable_articles=[5] → STOP

Step 2 (Annex III - 8 Categories, 60+ patterns):
  BIOMETRIC | CRITICAL_INFRASTRUCTURE | EDUCATION | EMPLOYMENT
  ESSENTIAL_SERVICES | LAW_ENFORCEMENT | MIGRATION | JUSTICE
  + NER RISK_TIER entities ("high-risk", "hochriskant") → fed into Annex III detection
  → AnnexIIIMatch list with matched_keywords per category

Step 3 (Article 6(1) Annex I - Safety Component Signals):
  14 patterns (CE marking, MDR/IVDR, notified body, Class IIa/III medical devices)
  → annex_i_triggered = True if ≥ 2 signals matched

Step 4 (ML Augmentation):
  predict_high_risk() at confidence ≥ 0.85 → catches Annex III misses

is_high_risk = Step2 OR Step3 OR ML
applicable_articles = [9,10,11,12,13,14,15] if high_risk, else []
```

### Step 4 - Chunk Classification

Each document chunk is classified to one of 7 `ArticleDomain` values (Articles 9–15) via:
- **Default**: Ollama / phi3:mini (CPU-friendly, no GPU required)
- **GPU fast-path**: NVIDIA Triton / fine-tuned gBERT ONNX (~50–100× faster)

`classify_chunks` returns `(chunks, backend_str)` — the backend string records any Triton→Ollama fallback that occurred at runtime and is written to the report's `classifier_backend` field.

Immediately after classification, **NER Phase 2 domain correction** runs: already-extracted NER entities are used to correct UNRELATED chunks that contain exactly one explicit Article 9–15 reference, recovering chunks the classifier missed.

Domain assignments inform which chunks are sent to each article's gap analyser.

### Step 5 - Hybrid RAG + LangGraph Gap Analysis

Runs concurrently across all applicable articles (`asyncio.gather`). For each article:

**Retrieval (Hybrid RAG)**
```
BM25 index ──────────────────────────────────────────────► top-10 keyword matches
ChromaDB vector (e5-small embeddings) ───────────────────► top-10 semantic matches
                                                            │
                                          Reciprocal Rank Fusion (k=60)
                                                            │
                                    Cross-Encoder re-rank (ms-marco-MiniLM-L-6-v2)
                                                            │
                                          Filtered regulatory passages
```

**LangGraph 3-Node Pipeline (3 Ollama calls per article)**
```
START
  │
  ▼
legal_agent_node
  Extracts strict checklist from top-8 regulatory passages
  → extracted_requirements: list[str]
  │
  ▼
technical_agent_node
  Evaluates top-15 user chunks against each requirement
  → evidence_findings: dict[requirement → "Found: ..." | "Missing: ..."]
  │
  ▼
synthesis_agent_node
  Compiles findings into scored gap report
  → score (0-100), gaps[], recommendations[], reasoning
  │
  ▼
END → ArticleScore
```

Non-applicable articles: **score=100, zero LLM calls.**
No-chunk articles: **score=0, critical gap, zero LLM calls.**

### Step 6 - Evidence Mapping (Deterministic - No LLM)

After gap analysis, obligation schemas in `data/obligations/**/*.jsonl` are filtered by actor type and applicable articles. For each required evidence artefact:

1. **Fast path**: pre-compiled regex synonym dictionary (22 canonical terms × synonym lists, e.g. "risk register" → 8 synonyms including "risikokatalog")
2. **Slow path**: NLI Cross-Encoder (`cross-encoder/nli-deberta-v3-small`) — premise: chunk text, hypothesis: "This document contains a [term]." → ENTAILMENT class predicted with score ≥ 0.5 = match

Output: `EvidenceMap` with per-obligation coverage (fully satisfied / partially satisfied / missing).

### Step 7 - Scoring & Confidence

```
Overall score     = mean(scores of applicable_articles only)
Confidence score  = mean(actor.confidence, 0.5 + evidence_coverage/2, classified/total)
requires_human_review = confidence < 0.70 OR actor_type == UNKNOWN

Risk tier (authoritative from applicability engine):
  is_prohibited → PROHIBITED
  is_high_risk  → HIGH
  else          → MINIMAL
```

### Step 8 - Results Dashboard

| Panel | What It Shows |
|---|---|
| Human review banner | Amber warning when confidence < 70% or actor is unknown |
| Risk tier comparison | Wizard self-assessment vs. document-derived tier |
| Actor classification | Detected Article 3 role + confidence + matched signals |
| Applicability gate | Risk tier + matched Annex III categories with keywords |
| Evidence coverage | % of legal obligations evidenced in the document |
| Overall score | Average across applicable Articles 9–15 only |
| Article cards (×7) | Score + gap severity; card colour driven by worst gap |
| Classifier metrics (`/metrics`) | BERT, NER, specialist classifier F1/precision/recall, confusion matrices |

Click any article card for:
- **Why this score?** - LLM reasoning
- **Which regulation?** - ChromaDB-retrieved regulatory passages
- **Can I defend this in an audit?** - Remediation checklist; verdict based on Critical/Major gap count

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI 0.111 · Uvicorn · Pydantic v2 · Python 3.11 |
| **Agent Workflow** | LangGraph · LangChain Core |
| **Vector Database** | ChromaDB ≥ 1.0.0 (3 collections: eu_ai_act, gdpr, compliance_checklist) |
| **Search (opt-in)** | OpenSearch 2.13 (server-side BM25 replacement) |
| **Embeddings** | `intfloat/multilingual-e5-small` via sentence-transformers (local, CPU) |
| **BM25 (default)** | rank-bm25 (in-memory, partitioned by collection + article) |
| **Re-ranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **NLI Evidence** | `cross-encoder/nli-deberta-v3-small` (lazy-loaded) |
| **LLM (default)** | Ollama · Phi-3 Mini 3.8B Q4 |
| **BERT Classifier** | `deepset/gbert-base` fine-tuned (8-class, ONNX export) |
| **Specialist Classifiers** | 3× fine-tuned gBERT: actor, risk, prohibited |
| **Inference Server** | NVIDIA Triton 24.02 (GPU opt-in via `--profile triton`) |
| **NER** | spaCy 3.7 · `de_core_news_lg` · 8 entity types |
| **Document Parsing** | PyMuPDF · python-docx · pdfplumber (tables) · pytesseract (OCR, optional) |
| **Frontend** | React 18 · TypeScript 5.4 · Vite · Tailwind CSS |
| **HTTP Client** | Axios (frontend) · httpx async (backend) |
| **Reports** | WeasyPrint 62.x · Jinja2 3.1 |
| **Containers** | Docker Compose v2 (6 services, 3 optional profiles) |
| **Logging** | structlog 24.2 (structured, per-request context) |
| **Version Tracking** | VersionManager (data/model/metric traceability) |

---

## Hardware & Model Choices

### Our Setup (4 GB VRAM) - What We Used

This project was developed and tested on a machine with **4 GB VRAM** (NVIDIA RTX 3050 Ti). With constrained hardware, we chose:

- **LLM: Ollama phi3:mini (3.8B Q4)** - Runs comfortably on CPU or low-VRAM GPUs. Inference is ~5–10 s per chunk but fully deterministic. The entire audit pipeline is CPU-capable.
- **BERT: deepset/gbert-base (110M params)** - Small enough to train on a 4 GB GPU in ~20–30 min per model.
- **Embeddings: multilingual-e5-small** - Designed for CPU inference; ~80 MB model.
- **NLI Cross-Encoder: nli-deberta-v3-small** - Lightweight NLI model for evidence matching.

### Scale to Your Hardware

You are not limited to our choices. KlarKI is built to be swapped:

| Component | Our Choice (4 GB) | Better Hardware Options |
|---|---|---|
| LLM | `phi3:mini` (3.8B Q4) | `llama3.1:8b`, `mistral:7b`, `gemma2:9b`, `llama3.3:70b` |
| BERT Classifier | `deepset/gbert-base` (110M) | `deepset/gbert-large`, `xlm-roberta-large` |
| Embeddings | `multilingual-e5-small` | `multilingual-e5-large`, `bge-m3` |
| NLI Re-ranking | `nli-deberta-v3-small` | `nli-deberta-v3-large`, `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| Inference Backend | Ollama (CPU) | NVIDIA Triton (GPU-batched, 50–100× faster) |

To use a different Ollama model, simply set `OLLAMA_MODEL=llama3.1:8b` in your `.env`. To enable GPU-batched inference, set `USE_TRITON=true` and run `./run.sh triton`.

### Data Generation at Your Scale

Synthetic training data generation is fully configurable:

```bash
# Default (our hardware — balances quality and speed)
python scripts/setup.py --gen-per-class 400

# Faster (low RAM / slow Ollama)
python scripts/setup.py --gen-per-class 100 --skip-generate

# Higher quality (more VRAM / faster GPU)
python scripts/setup.py --gen-per-class 1000

# Use a better LLM for generation
OLLAMA_MODEL=llama3.1:8b python scripts/setup.py --gen-per-class 400
```

The NER data pipeline requires **no LLM at all** - it uses deterministic template expansion and runs in seconds on any hardware.

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Docker Desktop | 4.x+ | With Docker Compose v2 |
| Docker | 4.x+ with Compose v2 | All pipeline steps run inside containers — no host Python required |
| Disk space | ~8–12 GB | ~2.3 GB Ollama model + images + training artefacts |
| RAM | 8 GB minimum | 16 GB recommended for parallel training |
| GPU (optional) | 4 GB VRAM+ | For BERT training and Triton inference; CPU fallback is automatic |
| NVIDIA Container Toolkit | Latest | Only required for `--profile triton` (GPU inference) |

---

## Quick Start

```bash
git clone https://github.com/s4nkar/KlarKI-EU-AI-Act-compliance-auditor.git
cd KlarKI-EU-AI-Act-compliance-auditor

cp .env.example .env           # Configure environment (defaults work out of the box)

./run.sh setup                 # First-time init (safe to re-run; completed stages are skipped)
```

Open **http://localhost** (port 80, nginx). Complete the **Risk Assessment** wizard, then **Upload Docs**.

For development with hot reload: `./run.sh dev` — frontend on **http://localhost:3000**.

> **Windows users:** use `./run.sh` commands. **Linux/Mac:** `make` aliases are available — `make setup`, `make up`, `make test`, etc.

---

## Pretrained Models (Optional)

> **Skipping local training is completely optional.** If you have decent hardware and want full control over your models, ignore this section and let `./run.sh setup` train everything from scratch (~30–60 min).

KlarKI ships 5 fine-tuned models on HuggingFace Hub. If you want to skip local training entirely, download them before running setup:

```bash
pip install huggingface-hub>=0.26.0
python scripts/download_pretrained.py
```

That's it. The models are placed exactly where the pipeline expects them. Then start the stack as normal:

```bash
./run.sh up
```

### Which path is right for you?

| Scenario | What to do |
|---|---|
| **You have a GPU / don't mind waiting 30–60 min** | Skip this section. Run `./run.sh setup` — trains on your own data, gives you full reproducibility and version history. |
| **You want to get started immediately** | Run `python scripts/download_pretrained.py` first, then `./run.sh setup --skip-train`. Setup still runs the knowledge base, Ollama, and ChromaDB stages — just skips training. |
| **You want a single model (e.g. only NER)** | `python scripts/download_pretrained.py --model ner` |
| **You already downloaded but want the latest** | `python scripts/download_pretrained.py --force` |

### What gets downloaded

| Model | HuggingFace Repo | Task |
|---|---|---|
| `bert` | `s4nkar/klarki-bert-classifier` | 8-class article domain classifier (Articles 9–15 + unrelated) |
| `actor` | `s4nkar/klarki-actor-classifier` | Article 3 actor role: provider / deployer / importer / distributor |
| `risk` | `s4nkar/klarki-risk-classifier` | Article 6 + Annex III high-risk binary classifier |
| `prohibited` | `s4nkar/klarki-prohibited-classifier` | Article 5 prohibited practice binary classifier |
| `ner` | `s4nkar/klarki-ner-spacy` | spaCy NER — `de_core_news_lg` fine-tuned on KlarKI data, 8 EU AI Act entity types |

All models are based on `deepset/gbert-base` (BERT classifiers) or `de_core_news_lg` (spaCy NER), fine-tuned on KlarKI's bilingual EN/DE regulatory training data.

> **Note:** Downloaded models are treated as authoritative by the version manager — `./run.sh setup` will not overwrite them unless you explicitly pass `--retrain`.

---

## Commands

| Command | What It Does |
|---|---|
| `./run.sh setup` | Full first-time init (Ollama, knowledge base, data gen, BERT/NER/specialist training). ONNX export and benchmark skipped by default — run `./run.sh triton` separately. |
| `./run.sh up` | Production mode (nginx, compiled images, port 80). GPU auto-detected. |
| `./run.sh dev` | Start containers in dev mode (hot reload, source volumes, frontend on port 3000) |
| `./run.sh triton` | Export ONNX models (if needed) + start Triton GPU inference. Requires NVIDIA GPU + Container Toolkit. |
| `./run.sh retrain` | Regenerate all training data and retrain all models |
| `./run.sh test` | Run full test suite inside the API container |
| `./run.sh logs` | Tail API logs |
| `./run.sh down` | Stop all containers |
| `./run.sh clean` | Full wipe - containers, volumes, ChromaDB data |

`setup` is **idempotent** - each stage skips itself if its outputs already exist.

**Target a single stage:**
```bash
python scripts/setup.py --only train-bert
python scripts/setup.py --only knowledge-base --rebuild-kb
python scripts/setup.py --only train-specialist
python scripts/setup.py --retrain --gen-per-class 300
python scripts/setup.py --skip-phase5    # Infrastructure-only mode (no ML)
```

---

## Setup Stages

| Stage | What It Does | Skip Flag |
|---|---|---|
| `seed-ollama` | Pull phi3:mini into Ollama cache | `--skip-seed` |
| `seed-nli` | Download NLI cross-encoder | `--skip-seed` |
| `knowledge-base` | Build ChromaDB from EU AI Act + GDPR regulatory text | `--skip-kb` |
| `build-graph` | Generate obligation JSONL from regulatory text | — |
| `generate-data` | Synthetic BERT training data via Ollama (~6,400 bilingual examples) | `--skip-generate` |
| `train-bert` | Fine-tune deepset/gbert-base (8 Article domains) | `--skip-train` |
| `generate-specialist-data` | Synthetic actor / risk / prohibited training data | `--skip-generate` |
| `train-specialist` | Fine-tune 3 specialist gBERT classifiers | `--skip-train` |
| `generate-ner-data` | Deterministic NER template expansion (no LLM) | `--skip-generate` |
| `train-ner` | Train spaCy NER model (60 epochs) | `--skip-train` |
| `export-bert` | BERT → ONNX for Triton (run by `./run.sh triton`, not `./run.sh setup`) | `--skip-export` |
| `export-e5` | e5-small → ONNX for Triton (run by `./run.sh triton`, not `./run.sh setup`) | `--skip-export` |
| `benchmark` | Ollama vs Triton latency comparison (run by `./run.sh triton`, not `./run.sh setup`) | `--skip-benchmark` |

---

## Classifier Backends

### Ollama / phi3:mini (Default)

Active when `USE_TRITON=false`. Phi-3 Mini 3.8B (Q4) runs inside the `klarki-ollama` container - **no GPU required**. Inference is ~5–10 s per chunk. All calls use `temperature=0`, `seed=42`, `top_k=1` - fully deterministic. Works on any modern laptop.

Compatible models (set `OLLAMA_MODEL` in `.env`):
```
phi3:mini       # Default - 3.8B, works on 4 GB VRAM or CPU
llama3.2:1b     # Fastest - 1B, minimal RAM
llama3.1:8b     # Best quality - 8B, requires 8 GB+ RAM
gemma2:2b       # Compact alternative - 2B
mistral:7b      # Strong reasoning - 7B
```

### Triton / gBERT (GPU Fast-Path)

Active when `USE_TRITON=true`. A fine-tuned `deepset/gbert-base` model exported to ONNX, served via NVIDIA Triton. Roughly **50–100× faster per chunk**, GPU-batched. Trained on mixed EN/DE data.

Requires: NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
./run.sh setup   # Build ONNX models first
./run.sh triton  # Then start in Triton mode
```

Every compliance report records the backend used in the `classifier_backend` field.

---

## Containers & Profiles

| Container | Role | Port | Profile |
|---|---|---|---|
| `klarki-frontend` | React SPA + Nginx reverse proxy | 80 | default |
| `klarki-api` | FastAPI backend (Uvicorn) | 8000 | default |
| `klarki-chromadb` | Vector database (ChromaDB) | 8001 | default |
| `klarki-ollama` | Local LLM inference server | 11434 | default |
| `klarki-training` | Python 3.11 container for all training jobs | — | `--profile training` (ephemeral, used by `./run.sh setup/retrain`) |
| `klarki-triton` | NVIDIA Triton ONNX inference | 8002 (HTTP) / 8003 (gRPC) | `--profile triton` |
| `klarki-opensearch` | Server-side BM25 search | 9200 | `--profile opensearch` |

```bash
# Production mode (Ollama + ChromaDB, GPU auto-detected)
./run.sh up

# Dev mode (hot reload, source volumes, frontend on port 3000)
./run.sh dev

# GPU inference (exports ONNX if needed, starts Triton)
./run.sh triton

# Manual docker compose examples (for reference)
docker compose up -d                           # Ollama + ChromaDB default mode
docker compose --profile triton up -d          # GPU inference
docker compose --profile opensearch up -d      # With OpenSearch BM25
```

---

## API Reference

The API is unauthenticated by design - intended for local use only. Do not expose to untrusted networks.

All routes are prefixed `/api/v1`. Responses use the `APIResponse` envelope: `{"status", "data", "error"}`.

### Audit

| Method | Path | Description |
|---|---|---|
| `POST` | `/audit/upload` | Upload file (PDF/DOCX/TXT/MD ≤10 MB) or `raw_text` form field. Optional `wizard_risk_tier`. Returns `{audit_id}`. |
| `GET` | `/audit/{id}` | Fetch full `AuditResponse` + `ComplianceReport` when `status=COMPLETE`. |
| `GET` | `/audit/{id}/status` | Lightweight status poll: `uploading \| parsing \| classifying \| analysing \| scoring \| complete \| failed` |

### Reports

| Method | Path | Description |
|---|---|---|
| `GET` | `/reports/{id}/pdf` | Stream PDF bytes (WeasyPrint). Returns 404 if unknown, 409 if not yet complete. |
| `GET` | `/reports/{id}/json` | Full `ComplianceReport` as JSON. |

### Wizard

| Method | Path | Description |
|---|---|---|
| `GET` | `/wizard/questions` | 9 Annex III yes/no questions (id + text, EN/DE). |
| `POST` | `/wizard/classify` | `{"answers": {"q1": bool, …, "q9": bool}}` → `{"risk_tier": "prohibited"\|"high"\|"limited"\|"minimal"}` |

### Metrics & Monitoring

| Method | Path | Description |
|---|---|---|
| `GET` | `/metrics/classifier` | BERT per-class precision/recall/F1 + confusion matrix. |
| `GET` | `/metrics/ner` | spaCy NER per-entity-label F1. |
| `GET` | `/metrics/specialists` | Actor / risk / prohibited specialist classifier metrics. |
| `GET` | `/metrics/versions` | Model version registry - active versions, history, data traceability. |
| `GET` | `/metrics/evaluation` | Latest eval results from `tests/evaluation/results/`. |

### System

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health - probes ChromaDB + Ollama. `{"services": {"chromadb": bool, "ollama": bool}}` |

---

## Report Schema

```jsonc
{
  "audit_id": "...",
  "language": "en",
  "risk_tier": "high",                  // Authoritative: Article 6 + Annex III gate result
  "wizard_risk_tier": "high",           // Self-assessed via wizard - shown in comparison panel
  "overall_score": 58.0,               // Average across applicable articles only (0–100)
  "confidence_score": 0.82,            // 0–1; below 0.70 triggers human review
  "requires_human_review": false,
  "classifier_backend": "ollama/phi3:mini",    // Actual backend used at runtime; reflects any Triton→Ollama fallback

  // Phase 3 - Actor Classification (Article 3)
  "actor": {
    "actor_type": "deployer",           // provider | deployer | importer | distributor | unknown
    "confidence": 0.91,
    "matched_signals": ["uses the system", "under its authority"],
    "reasoning": "..."
  },

  // Phase 3 - Applicability Gate (Article 6 + Annex III)
  "applicability": {
    "is_high_risk": true,
    "is_prohibited": false,
    "annex_i_triggered": false,
    "annex_iii_matches": [
      {
        "category": 4,
        "category_name": "Employment",
        "matched_keywords": ["recruitment", "CV screening"],
        "evidence_required": ["data protection impact assessment", "human oversight procedure"]
      }
    ],
    "applicable_articles": [9, 10, 11, 12, 13, 14, 15],
    "reasoning": "..."
  },

  // Phase 3 - Deterministic Evidence Coverage
  "evidence_map": {
    "total_obligations": 12,
    "fully_satisfied": 4,
    "partially_satisfied": 5,
    "missing": 3,
    "overall_coverage": 0.54,
    "items": [
      {
        "obligation_id": "...",
        "article": 9,
        "requirement": "...",
        "evidence_required": ["risk register", "risk assessment methodology"],
        "satisfied_evidence": ["risk register"],
        "missing_evidence": ["risk assessment methodology"],
        "coverage": 0.5
      }
    ]
  },

  // Per-Article Gap Analysis (7 articles)
  "article_scores": [
    {
      "article_num": 9,
      "domain": "risk_management",
      "score": 75.0,
      "score_reasoning": "...",
      "gaps": [
        { "severity": "major", "title": "...", "description": "..." }
      ],
      "recommendations": ["..."],
      "regulatory_passages": [...],
      "chunk_count": 12
    }
  ],

  // Article 5 scan
  "emotion_flag": {
    "detected": false,
    "is_prohibited": false,
    "explanation": "..."
  }
}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `phi3:mini` | LLM model (swap for larger models on better hardware) |
| `USE_TRITON` | `false` | `true` to use NVIDIA Triton/gBERT instead of Ollama |
| `USE_OPENSEARCH` | `false` | `true` to use OpenSearch for server-side BM25 |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Local sentence-transformer model |
| `UPLOAD_MAX_SIZE_MB` | `10` | Max upload file size in MB |
| `DEBUG` | `true` | Pretty-print structured logs |
| `OLLAMA_HOST` | `http://klarki-ollama:11434` | Ollama server URL (internal Docker network) |
| `CHROMADB_HOST` | `http://klarki-chromadb:8000` | ChromaDB server URL |
| `TRITON_HOST` | `klarki-triton` | Triton hostname (internal Docker network) |
| `TRITON_GRPC_PORT` | `8003` | Triton gRPC port |
| `UPLOAD_DIR` | `/data/uploads` | Temporary upload storage (files deleted post-audit) |

---

## Testing

```bash
./run.sh test          # Full test suite inside the API Docker container
make test-local        # Local Python env - faster for iteration
```

HTML reports are written to `tests/reports/` (gitignored):
- `tests/reports/report.html` - Pass/fail with tracebacks
- `tests/reports/coverage/index.html` - Line-level coverage

### Test Coverage

| File | Type | What It Covers |
|---|---|---|
| `test_api_audit.py` | Integration | Upload endpoint, MIME types, size limit (413), status codes |
| `test_api_reports.py` | Integration | PDF/JSON retrieval, 404/409 states |
| `test_parser.py` | Unit | PDF/DOCX/TXT/MD parsing, German encoding, table extraction |
| `test_chunker.py` | Unit | Heading-aware splitting, UUID assignment, section metadata |
| `test_language_detector.py` | Unit | EN/DE detection, short-text fallback |
| `test_classifier.py` | Unit | Label normalisation, Ollama/Triton dispatch, error fallback |
| `test_emotion_module.py` | Unit | Article 5 prohibition detection, context awareness |
| `test_risk_wizard.py` | Unit | Annex III PROHIBITED/HIGH/MINIMAL classification logic |
| `test_scorer.py` | Unit | Applicable-articles-only average, confidence, human review flag |
| `test_gap_analyser.py` | Unit | Score clamping, short-circuit logic, LLM mock, JSON parsing |
| `test_rag.py` | Unit | BM25 + vector, RRF, cross-encoder, **Precision@3 ≥ 80%** gate (7 golden queries) |
| `test_report_generator.py` | Unit | PDF bytes, Jinja2 template rendering |
| `test_emotion_module.py` | Unit | Article 5 detection (workplace/education vs commercial contexts) |
| `test_ner.py` | Unit | NER enrichment, domain correction for article-referencing chunks |

**Hard regression gate**: Precision@3 ≥ 80% across 7 golden RAG queries (one per Article 9–15) using in-memory ChromaDB - no live services required.

---

## Training Pipeline

### BERT Domain Classifier

All training runs inside the `klarki-training` Docker container (Python 3.11). Use `./run.sh setup` or `./run.sh retrain` — do not invoke training scripts directly on the host.

Fine-tuned `deepset/gbert-base` (110M params), 8 classes mapping to EU AI Act articles:

| Label | Article |
|---|---|
| `risk_management` | Article 9 |
| `data_governance` | Article 10 |
| `technical_documentation` | Article 11 |
| `record_keeping` | Article 12 |
| `transparency` | Article 13 |
| `human_oversight` | Article 14 |
| `security` | Article 15 |
| `unrelated` | - |

Training data: ~6,400 bilingual examples (400 per class per language, 85% train / 15% validation, stratified by class, seed=42) generated via async Ollama with a semaphore of 6 concurrent requests.

```bash
# Via run.sh (recommended)
./run.sh setup                # Full pipeline with BERT/NER/specialist training
./run.sh retrain              # Regenerate all training data and retrain models

# Direct script invocation (inside training container)
docker compose --profile training run --rm klarki-training python scripts/generate_bert_training_data.py --n-per-class 400 --languages en,de
docker compose --profile training run --rm klarki-training python training/train_classifier.py --epochs 5 --batch-size 16
```

### 3 Specialist Classifiers

All share the same architecture (fine-tuned gBERT):

| Classifier | Classes | Confidence Threshold | Used By |
|---|---|---|---|
| **Actor** | provider, deployer, importer, distributor | ≥ 0.80 | `actor_classifier.py` |
| **Risk** | high_risk, not_high_risk | ≥ 0.85 | `applicability_engine.py` |
| **Prohibited** | prohibited, not_prohibited | ≥ 0.85 | `applicability_engine.py` |

```bash
python scripts/generate_specialist_training_data.py --type all
python training/train_specialist_classifiers.py --type all
```

### spaCy NER Model

8 entity types - `de_core_news_lg` base with custom NER head. Training uses **deterministic template expansion (no LLM)**:

```
ARTICLE | OBLIGATION | ACTOR | AI_SYSTEM | RISK_TIER | PROCEDURE | REGULATION | PROHIBITED_USE
```

```bash
python scripts/generate_ner_data.py --n-templates 5000
python training/train_ner.py --epochs 60
```

### Weak Supervision Pipeline

Automatically labels regulatory text using regex patterns - runs in seconds with no LLM:

```bash
python scripts/build_weak_supervision_labels.py --type all
```

---

## Privacy

- **No external API calls** - all inference runs locally (Ollama, Triton, ChromaDB, spaCy)
- **Documents deleted immediately** after the audit pipeline completes
- **ChromaDB stores only regulatory text** (EU AI Act + GDPR) - never user documents
- **In-memory audit store only** - results are not persisted to disk or database
- **PDF reports generated and served locally** - never uploaded anywhere

---

## Troubleshooting

**Audit fails or all article scores are 0**

Ollama model not loaded. Run `./run.sh setup`, or pull manually:
```bash
docker exec klarki-ollama ollama pull phi3:mini
```

**`USE_TRITON=true` but audit still fails**

ONNX models must be built before switching to Triton mode:
```bash
./run.sh setup    # Builds ONNX models
./run.sh triton   # Then start in Triton mode
```

If the API image is stale: `docker compose up -d --build klarki-api`

**BERT training loss stuck / F1 below 0.5**

Increase training examples per class:
```bash
python scripts/setup.py --retrain --gen-per-class 600
```

**Specialist classifiers fall back to pattern matching**

Train the specialist classifiers first:
```bash
python scripts/setup.py --only generate-specialist-data
python scripts/setup.py --only train-specialist
```

**PDF download fails (WeasyPrint error)**

Check `docker compose logs klarki-api`. Ensure `pydyf==0.11.0` in `requirements.txt` — version 0.12+ breaks WeasyPrint 62.x.

**Docker network error on `docker compose up`**

Stale network from a failed previous start:
```bash
docker compose down --remove-orphans && docker network prune -f && docker compose up -d
```

**`./run.sh triton` says 'No NVIDIA GPU detected'**

Triton requires a CUDA-capable GPU with NVIDIA Container Toolkit. Use `./run.sh up` for CPU/Ollama mode.

**Ollama not using GPU after setup on a machine with GPU**

GPU acceleration is auto-detected via `nvidia-smi`. If not working, check that NVIDIA Container Toolkit is installed.

**GPU not detected by Ollama or Triton**

Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and restart Docker Desktop. Both services fall back to CPU automatically.

**ChromaDB connection error on startup**

ChromaDB takes ~10 s to initialise. The API retries on startup - wait a moment and refresh.

**Same document gives different results between runs**

All LLM calls use `temperature=0` and `seed=42` - output should be identical. Ensure the Ollama container was fully restarted: `docker compose restart klarki-ollama`

**Frontend changes not hot-reloading (Windows)**

Vite uses polling on Windows Docker volumes - changes reload within ~500 ms. If stuck:
```bash
docker compose restart klarki-frontend
```

**Start completely fresh**
```bash
./run.sh clean && ./run.sh setup
```

---

## Contributing & Collaboration

KlarKI is **fully open source** and free to use, modify, and build upon under the MIT licence.

### Ways to Contribute

- **New regulatory coverage** - Add GDPR Article 2 territorial scope gate, GDPR actor normalisation (controller/processor → ActorType), or new EU AI Act articles
- **Obligation schemas** - Expand `data/obligations/` JSONL files for Articles 9–15 and Article 5 (currently only `article_6_annex_iii.jsonl` is complete)
- **Phase 3 tests** - Write tests for `actor_classifier.py`, `applicability_engine.py`, `evidence_mapper.py`, and `agent_graph.py`
- **Better models** - If you have more hardware, train with larger base models and open a PR with updated metrics
- **Languages** - Extend beyond EN/DE to French, Italian, Spanish (EUR-Lex provides official translations)
- **Frontend** - Improve the evidence citation view, add article-level evidence breakdown in PDF reports

### Getting Started

```bash
git clone https://github.com/s4nkar/KlarKI-EU-AI-Act-compliance-auditor.git
cd KlarKI-EU-AI-Act-compliance-auditor
cp .env.example .env
./run.sh setup --skip-phase5    # Quick infra-only setup (skips ML training)
```

Full technical documentation is in [`docs/`](docs/index.md) — architecture, training pipeline, inference pipeline, RAG system, model inventory, and configuration guide.

Architecture, training pipeline, ML conventions, and service contracts are fully documented in [`CLAUDE.md`](CLAUDE.md).

### Reporting Issues

- **Bugs / feature requests**: [Open a GitHub issue](https://github.com/s4nkar/KlarKI-EU-AI-Act-compliance-auditor/issues)
- **Security vulnerabilities**: Do not open a public issue - email [s4nkar.sub.inr@gmail.com](mailto:s4nkar.sub.inr@gmail.com) directly

---

## Licence

[MIT](LICENSE) - free to use, modify, and distribute. See `LICENSE` for details.

---

*Built for German SMEs navigating EU AI Act compliance. Tested on 4 GB VRAM - scales to any hardware.*
