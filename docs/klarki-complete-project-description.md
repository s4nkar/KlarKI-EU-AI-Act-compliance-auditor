# KlarKI — EU AI Act + GDPR Compliance Auditor

**Local-first. Privacy-preserving. Deterministic.**

KlarKI is an on-device EU AI Act and GDPR compliance auditor built for organisations assessing whether their AI systems meet European regulatory requirements. The primary audience is German SMEs, though the pipeline handles EN and DE documents throughout. No data ever leaves the machine: all inference runs across six Docker services, uploaded documents are deleted immediately after processing, and in-memory audit results are never persisted to disk.

Users begin with a 9-question Annex III risk wizard, then upload a policy document (PDF, DOCX, TXT, or MD). The platform returns a scored compliance report as PDF or JSON, covering actor classification, applicability reasoning, evidence coverage, per-article gap analysis, and audit defensibility verdicts.

Developed and validated on 4 GB VRAM hardware, but all components are independently swappable.

---

## Architecture

### Container topology

```
Browser
   │
   ▼
klarki-frontend  :80
   Nginx reverse proxy
   Serves React 18 SPA
   Proxies /api/* → klarki-api:8000
   │
   ▼
klarki-api  :8000
   FastAPI + Uvicorn · Python 3.11
   All business logic and ML models in-process
   │
   ├──► klarki-chromadb  :8001
   │      ChromaDB vector database
   │      Collections: eu_ai_act, gdpr, compliance_checklist
   │      Persistent via Docker volume chroma_data
   │
   ├──► klarki-ollama  :11434
   │      Ollama · phi3:mini 3.8B Q4 (default)
   │      GPU-accelerated when NVIDIA GPU present (auto-detected)
   │
   └──► klarki-triton  :8002/8003  [optional — ./run.sh triton]
          NVIDIA Triton Inference Server
          Serves BERT ONNX + e5 ONNX

klarki-training  [ephemeral — --profile training]
   All training jobs: data generation, BERT fine-tuning, NER training
   Mounts project root as /workspace
   Talks to klarki-ollama and klarki-chromadb

klarki-opensearch  :9200  [optional — --profile opensearch]
   Drop-in BM25 replacement for large document corpora
```

All inter-service communication runs on the `klarki-net` bridge network using Docker internal DNS. The base `docker-compose.yml` handles production. A GPU overlay (`docker-compose.gpu.yml`) adds NVIDIA reservations to Ollama automatically when `nvidia-smi` is detected at startup.

### Storage layout

Docker volumes persist across restarts:

| Volume | Contents |
|---|---|
| `chroma_data` | ChromaDB index and embeddings |
| `ollama_data` | Ollama model cache (~2.2 GB for phi3:mini) |
| `upload_data` | Temporary user uploads (deleted post-pipeline) |

Host-mounted training artifacts:

```
training/artifacts/
  bert_classifier/           gbert-base fine-tuned, 8-class
  actor_classifier/          4-class actor role detector
  risk_classifier/           binary high-risk detector
  prohibited_classifier/     binary prohibition detector
  spacy_ner_model/           de_core_news_lg + custom NER head
model_repository/
  bert_clause_classifier/1/model.onnx
  e5_embeddings/1/model.onnx
```

---

## Inference pipeline

Every audit is triggered by `POST /api/v1/audit/upload` and runs as a FastAPI `BackgroundTask`. The browser polls status every 2 seconds.

### Stage 1 — Parsing

Document parsing covers four formats: PDF prose extraction via PyMuPDF, table extraction via pdfplumber (output as tab-separated rows appended to prose), OCR fallback for scanned PDFs via pytesseract, and DOCX extraction via python-docx.

The text is then passed through `proposition_chunk_text()`, a heading-aware splitter that:

- detects section headings via 6 regex patterns (Markdown `##`, `1.1` numbered, `Article N`, ALLCAPS, etc.)
- splits colon-introduced obligation lists into individual propositions ("Provider shall: (a)...; (b)..." becomes two chunks sharing the obligation stem)
- never splits conditional clauses ("shall X if Y" stays intact)
- sub-splits sections over 800 chars by paragraph and merges fragments under 80 chars into the preceding chunk

Each chunk carries a UUID, the extracted text, its section heading, and an `is_proposition` flag. Language detection (EN/DE via langdetect, fallback to EN for text under 100 chars) runs once on the full document.

### Stage 2 — Legal decision hierarchy (deterministic, no LLM)

Two tasks run concurrently via `asyncio.to_thread`.

**Actor classification (Article 3)**

The ML path runs `predict_actor()` on the first 2,000 characters. If confidence ≥ 0.80, that result is used directly. Otherwise, 39 bilingual EN/DE regex patterns cover 14 provider signals, 13 deployer signals, 6 importer signals, and 6 distributor signals. Confidence is computed as `matched_class_signals / total_signals`. The default when nothing matches is DEPLOYER — the most common role for SMEs.

**Applicability gate (Article 6 + Annex III) — 4-step deterministic tree**

Step 1 scans for Article 5 prohibitions: 9 patterns covering subliminal manipulation, social scoring, workplace and education emotion recognition, and real-time biometric identification in public spaces, augmented by `predict_prohibited()` at confidence ≥ 0.85. A prohibition hit sets `applicable_articles=[5]` and halts further processing — no LLM calls, no RAG.

Step 2 matches against Annex III across 8 domain categories (BIOMETRIC, CRITICAL_INFRASTRUCTURE, EDUCATION, EMPLOYMENT, ESSENTIAL_SERVICES, LAW_ENFORCEMENT, MIGRATION, JUSTICE) using 60+ patterns.

Step 3 detects Article 6(1) Annex I safety-component signals: 14 patterns covering CE marking, MDR/IVDR references, notified body mentions, and Class IIa/III medical device language. `annex_i_triggered = len(hits) >= 2` — the two-hit threshold avoids false positives on documents that mention safety in passing.

Step 4 runs `predict_high_risk()` at confidence ≥ 0.85 to catch Annex III cases that pattern matching missed.

`is_high_risk = Step2 OR Step3 OR ML`. For minimal-risk systems, `applicable_articles = []` and all seven article analysers immediately return score=100 with zero LLM calls. This is the biggest latency saving in the pipeline.

### Stage 3 — Chunk classification and NER enrichment

Each document chunk is classified to one of 7 `ArticleDomain` values (Articles 9–15) or `UNRELATED`.

In Ollama mode (default), a few-shot prompt is sent per chunk to phi3:mini — roughly 5–10s per chunk sequentially. In Triton mode, chunks are tokenised locally with `BertTokenizer`, batched to 32, sent via gRPC to the BERT ONNX model on Triton, and returned as logits — roughly 50ms per batch-of-32, a 50–100× throughput improvement.

spaCy NER then runs on every chunk (capped at 1,000 chars each) to extract 8 entity types: ARTICLE, OBLIGATION, ACTOR, AI_SYSTEM, RISK_TIER, PROCEDURE, REGULATION, PROHIBITED_USE. Entity metadata is attached to each chunk and used downstream in report generation and evidence mapping. The NER model also performs domain correction: if a chunk was classified as UNRELATED but NER finds exactly one Article reference in the 9–15 range, the domain is corrected to the matching ArticleDomain. This recovers short procedural paragraphs that BERT misses.

If the NER model hasn't been trained, chunks are returned unchanged — the pipeline degrades gracefully.

### Stage 4 — Multi-agent gap analysis

Chunks are grouped by domain into 7 buckets, and `asyncio.gather` runs all applicable articles concurrently.

For each article, `process_article()` checks three conditions first: if the article is not in `applicable_articles`, return score=100 immediately; if no document chunks matched this domain, return score=0 with "No evidence found" — in both cases no LLM or ChromaDB call is made.

For applicable articles with evidence, `retrieve_requirements()` runs the full hybrid RAG pipeline (described separately below), then `analyse_article()` invokes a linear 3-node LangGraph `StateGraph`:

`legal_agent_node` takes the top 8 regulatory passages and extracts a strict compliance checklist — concrete requirements the audited document must satisfy.

`technical_agent_node` evaluates the user's document chunks against each requirement, producing found / partial / missing classifications per requirement.

`synthesis_agent_node` compiles the final gap report: a 0–100 score, gap descriptions, remediation recommendations, and reasoning.

Each node is independently error-handled and returns safe defaults (empty lists, score=30) on exception. Total Ollama calls per applicable article: 3. All calls use `temperature=0, seed=42, top_k=1` — same document always produces the same report.

### Stage 5 — Scoring

`evidence_mapper.py` loads obligation schemas from `data/obligations/**/*.jsonl`, filters by the current actor type and applicable articles, then checks for 22 canonical compliance artefacts (risk registers, DPIAs, technical documentation, human oversight procedures, conformity assessment records, and others).

The fast path uses regex synonym dictionaries — roughly 8 synonyms per canonical term including German-language variants ("risk register" → "risikokatalog", "gefährdungsregister", etc.). The slow path, triggered only when regex fails, uses `cross-encoder/nli-deberta-v3-small` to check entailment: premise = chunk text, hypothesis = "This document contains a \<term\>." An ENTAILMENT prediction counts as a semantic match.

Each obligation's evidence is classified as fully satisfied, partially satisfied, or missing. Overall evidence coverage is aggregated as a percentage.

`compliance_scorer.py` computes the final scores:

- Risk tier: PROHIBITED if `is_prohibited`, HIGH if `is_high_risk`, MINIMAL otherwise
- Overall compliance score: mean of applicable articles only; non-applicable articles (score=100) are excluded from the mean; a minimal-risk system scores 100%
- Confidence score: `mean(actor_confidence, 0.5 + evidence_coverage/2, classified_chunks/total_chunks)`
- `requires_human_review = confidence < 0.70 OR actor_type == UNKNOWN`
- Audit defensibility verdict: based on Critical and Major gap counts per article

The uploaded file is deleted from disk at this point. The `ComplianceReport` is stored in-memory keyed by audit UUID.

Reports are served as PDF (WeasyPrint renders a Jinja2 template) or JSON via `GET /api/v1/reports/{id}/pdf` and `GET /api/v1/reports/{id}/json`.

---

## RAG system

KlarKI uses a hybrid retrieval pipeline: BM25 keyword search combined with ChromaDB dense vector search, merged via Reciprocal Rank Fusion, and re-ranked by a cross-encoder.

### Retrieval flow

```
chunk (text + domain + language)
   │
   ├── BM25 (rank_bm25, in-memory, partitioned by article_num)
   │     query: lowercased, punctuation-stripped chunk text
   │     filter: article_num = DOMAIN_TO_ARTICLE[chunk.domain]
   │     → top 10 per collection (eu_ai_act, compliance_checklist)
   │
   └── Vector search (ChromaDB + multilingual-e5-small)
         query: embed(chunk.text)
         where: {"article_num": N}
         → top 10 per collection
   │
   ▼
Language-aware sort (same-language passages ranked first, not hard-filtered)
   │
   ▼
RRF merge  score = Σ 1/(60 + rank_i), deduplicated by passage id → ~15 candidates
   │
   ▼
Cross-encoder re-rank (ms-marco-MiniLM-L-6-v2)
   input: [(chunk.text, passage.text)] per candidate
   output: relevance score, sorted descending → top 5
   runs via asyncio.to_thread (non-blocking)
   falls back to RRF order if unavailable
   │
   ▼
List[RegulatoryPassage] (top 5) → LangGraph
```

### Metadata filtering

The article_num filter is the primary latency saver. Every regulatory passage in ChromaDB carries `article_num` metadata. The BM25 index is partitioned by `article_num` at startup. Retrieving for domain `risk_management` only searches article_9 passages — reducing the candidate space from ~500 total to ~25–30.

The applicability gate provides an even larger saving: if an article is not in `applicable_articles`, `retrieve_requirements()` returns `[]` immediately with no ChromaDB or BM25 query. For a minimal-risk system, the entire RAG + LangGraph path is skipped for all seven articles.

An optional regulation filter (e.g. `"eu_ai_act"`) prevents GDPR passages appearing in EU AI Act article analysis when both regulation types are queried.

### ChromaDB collections

| Collection | Content | Approximate size |
|---|---|---|
| `eu_ai_act` | Articles 5, 9–15 (EN + DE) | ~200 chunks |
| `gdpr` | Articles 5, 6, 24, 25, 30, 35 (EN + DE) | ~150 chunks |
| `compliance_checklist` | Structured requirement sentences | ~85 chunks |

`eu_ai_act` and `compliance_checklist` are queried on every RAG request. `gdpr` is populated but not yet wired into the live retrieval path (planned for GDPR gate in Phase D).

### OpenSearch (optional BM25 backend)

For large document corpora in production, BM25 can be swapped to OpenSearch by starting the `opensearch` profile and setting `USE_OPENSEARCH=true`. OpenSearch provides persistent indexing, server-side metadata filtering, and BM25+ with field boosting. Vector search remains in ChromaDB — OpenSearch does not replace it. For public demo use, `rank_bm25` in-memory is adequate.

---

## ML models

KlarKI uses 8 models at inference time: 5 trained locally during setup, 3 downloaded pre-trained.

### Trained locally

**BERT clause classifier** (`training/artifacts/bert_classifier/`)

Fine-tuned `deepset/gbert-base` (110M parameters) for 8-class sequence classification. Input: document chunk text, max 256 tokens. Output: ArticleDomain label + confidence. Training data: 6,400 synthetic examples generated by Ollama from actual regulatory text (400 examples × 8 classes × EN + DE). Target: macro F1 ≥ 0.93 on validation set. In inference, classifies every chunk into one of the 7 compliance domains or UNRELATED. Can run via Ollama few-shot prompt or Triton ONNX batch inference.

**Actor classifier** (`training/artifacts/actor_classifier/`)

Same base model, 4-class: provider, deployer, importer, distributor. Input: first 2,000 characters of the document. Used at confidence ≥ 0.80; otherwise falls back to 39 bilingual regex patterns. Default: DEPLOYER.

**Risk classifier** (`training/artifacts/risk_classifier/`)

Binary: high_risk / not_high_risk. Used at confidence ≥ 0.85 to augment the Annex III pattern detection — catches cases the 60+ regex patterns miss.

**Prohibited classifier** (`training/artifacts/prohibited_classifier/`)

Binary: prohibited / not_prohibited. Used at confidence ≥ 0.85 to augment Article 5 pattern detection. Either the 9 regex patterns or the ML model can trigger `is_prohibited=True`.

**spaCy NER model** (`training/artifacts/spacy_ner_model/model-final`)

Base: `de_core_news_lg` (pre-trained German, tok2vec + 560k word vectors). Custom NER head for 8 entity types. Training data: 5,000 records generated via deterministic template expansion (no Ollama required). Runs after BERT classification on every chunk, attaches entity metadata, and performs conservative domain correction for UNRELATED chunks referencing a single explicit article number.

All trained models are lazy-loaded on first use and cached. If a model hasn't been trained, the system falls back to regex patterns or returns chunks unchanged — the audit pipeline never crashes because a model is missing.

### Pre-trained, downloaded at runtime

**multilingual-e5-small** (`intfloat/multilingual-e5-small`): 384-dimensional sentence embeddings, 100+ languages. Downloaded at API startup, loaded once into `app.state.embeddings`. Embeds document chunks for ChromaDB and query vectors for retrieval. Also exportable to ONNX for Triton.

**ms-marco cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`): passage re-ranking. Scores (chunk, passage) pairs for relevance after RRF merge, keeps top 5. Falls back to RRF order if unavailable.

**nli-deberta cross-encoder** (`cross-encoder/nli-deberta-v3-small`): NLI entailment for evidence mapping. Checks whether a chunk entails that the document contains a specific compliance artefact. Pre-downloaded in Stage 1.5 of setup to avoid cold-start on first audit.

---

## Training pipeline

All training runs inside the `klarki-training` Docker container (Python 3.11). Host Python version is irrelevant.

### Stage order

```
Stage 1    seed-ollama              Pull phi3:mini into Ollama cache
Stage 1.5  seed-nli                 Download NLI cross-encoder
Stage 2    knowledge-base           Chunk + embed regulatory text → ChromaDB
Stage 2.5  build-graph              Extract obligation schemas → JSONL
Stage 3    generate-data            Synthetic BERT data via Ollama
Stage 4    train-bert               Fine-tune gbert-base (8 classes)
Stage 3.5  generate-specialist-data Actor/risk/prohibited data via Ollama
Stage 4.5  train-specialist         Fine-tune 3 specialist classifiers
Stage 5    generate-ner-data        Deterministic template expansion (no Ollama)
Stage 6    train-ner                Train spaCy NER model
Stage 7    export-bert              BERT → ONNX  (Triton only, skipped by default)
Stage 8    export-e5                e5-small → ONNX  (Triton only, skipped by default)
Stage 9    benchmark                Ollama vs Triton latency (Triton only)
```

Each stage auto-skips if its output already exists. Stages 7–9 are skipped by `./run.sh setup` and `./run.sh retrain` — they run automatically with `./run.sh triton`.

### Knowledge base (Stage 2)

Regulatory text lives in `data/regulatory/**/*.txt` with a standard header (ARTICLE, REGULATION, DOMAIN, EN and DE sections). The pipeline parses each file, splits EN and DE sections separately, chunks with `RecursiveCharacterTextSplitter(chunk_size=512, overlap=50)`, embeds with `multilingual-e5-small` on CPU, and upserts to ChromaDB with metadata `{article_num, regulation, domain, lang, title}`.

### Obligation schemas (Stage 2.5)

Ollama (phi3:mini) reads each article and extracts machine-readable obligation schemas defining what evidence a compliant organisation must produce, which actor types it applies to, and what the obligation requires. Output: `data/obligations/eu_ai_act/*.jsonl`. Used at inference time by `evidence_mapper.py`.

### BERT training data (Stage 3)

Ollama generates realistic synthetic sentences that would appear in compliance documents, using actual regulatory article text as context. Default: 400 examples × 8 classes × 2 languages = 6,400 total. Labels map to articles: `risk_management` (Art. 9), `data_governance` (Art. 10), `technical_documentation` (Art. 11), `record_keeping` (Art. 12), `transparency` (Art. 13), `human_oversight` (Art. 14), `security` (Art. 15), `unrelated` (non-compliance text).

### BERT training (Stage 4)

Base model: `deepset/gbert-base` (110M parameters). Train/val split: 85/15, stratified, seed=42. Default hyperparameters: 12 epochs with early stopping (patience=3, monitors eval macro F1, typically stops at 5–8), batch size 16 (effective 32 via gradient accumulation ×2), learning rate 2e-5 with cosine decay, max length 256 tokens. Production target: macro F1 ≥ 0.93. Below 0.85: training script prints a warning.

### NER data and training (Stages 5–6)

NER data generation is fully deterministic — no Ollama required. Template expansion uses controlled vocabularies and sentence templates from actual regulatory text to produce records with exact character offsets. Default: 5,000 records.

NER training uses `de_core_news_lg` as the base. Default: 60 epochs, batch 32, early stopping patience 10.

### Weak supervision (alternative)

`scripts/build_weak_supervision_labels.py` produces specialist classifier training data using regex rules on regulatory text — no Ollama, fully deterministic, runs in seconds. Not called by the setup pipeline; opt-in when Ollama is unavailable or when reproducible training data is required.

### Version management

`training/version_manager.py` tracks data content hashes and model versions. `--retrain` skips retraining when data hashes haven't changed. `--force-retrain` always produces a new version.

---

## Configuration

### Runtime (.env)

Key settings:

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_MODEL` | `phi3:mini` | Any Ollama-compatible model; pull before switching |
| `USE_TRITON` | `false` | Set automatically by `./run.sh triton` |
| `USE_OPENSEARCH` | `false` | Set manually; requires opensearch profile |
| `UPLOAD_MAX_SIZE_MB` | `10` | Enforced in FastAPI route |
| `DEBUG` | `false` | Enables verbose structlog output |

All LLM calls hardcode `temperature=0, seed=42` for deterministic output. Changing these without re-evaluating output quality is not recommended.

### Inference backends

**Triton** (GPU): `./run.sh triton` checks for `nvidia-smi`, exports BERT + e5 to ONNX if needed, starts the Triton container, sets `USE_TRITON=true`, and restarts the API. Chunk classification goes from ~5–10s/chunk (Ollama) to ~50ms/batch-32 (Triton ONNX). LangGraph gap analysis still uses Ollama. Switch back with `./run.sh up`.

**OpenSearch**: start the `opensearch` profile, index regulatory text with `python scripts/build_knowledge_base.py --opensearch`, set `USE_OPENSEARCH=true`, restart the API. BM25 queries go to OpenSearch; vector search stays in ChromaDB; RRF and cross-encoder are unchanged.

### Confidence thresholds (hardcoded)

| Threshold | File | Meaning |
|---|---|---|
| Actor ML ≥ 0.80 | `actor_classifier.py` | Below this, use regex fallback |
| Risk ML ≥ 0.85 | `applicability_engine.py` | Below this, ML doesn't augment patterns |
| Prohibited ML ≥ 0.85 | `applicability_engine.py` | Below this, ML doesn't augment patterns |
| Annex I signals ≥ 2 | `applicability_engine.py` | Two-hit threshold to avoid false positives |
| Confidence < 0.70 | `compliance_scorer.py` | Report auto-flagged for human review |

---

## Privacy guarantees

- No external API calls: all inference runs locally (Ollama, Triton, ChromaDB, spaCy, cross-encoders).
- Uploaded documents are deleted from disk immediately after the audit pipeline completes.
- ChromaDB stores only regulatory text (EU AI Act + GDPR), never user documents.
- Audit results are in-memory only, keyed by UUID, never written to disk or a database.
- PDF reports are generated and served locally.

---

## Quick start

```bash
cp .env.example .env
./run.sh setup          # first run — builds everything (~30–60 min)
./run.sh up             # day-to-day (production, Nginx on :80)
./run.sh dev            # development (Vite hot reload on :3000)
./run.sh triton         # GPU inference via Triton (NVIDIA GPU required)
./run.sh retrain        # smart retrain (skips if data hashes unchanged)
./run.sh clean          # full reset
```

### Adding a new regulatory article

1. Add `data/regulatory/<regulation>/article_N.txt` with the standard header (ARTICLE, REGULATION, DOMAIN, EN and DE sections).
2. Add the domain to `DOMAIN_ARTICLE_MAP` in `scripts/generate_bert_training_data.py`.
3. Add the `ArticleDomain` enum value in `api/models/schemas.py`.
4. Run `./run.sh setup --retrain`.

---

## Key design decisions

**Local-first**: zero external API calls. Ollama runs the LLM on-device; all ML models are in-process or served locally via Triton.

**Applicability-gated**: the 4-step deterministic tree runs before any LLM is invoked. Non-applicable articles are never sent to the LLM, preserving legal correctness while eliminating redundant computation. For a minimal-risk system, the entire RAG + LangGraph path is skipped.

**Deterministic**: `temperature=0, seed=42, top_k=1` on all LLM calls. The same document always produces the same report.

**Graceful degradation**: every ML model has a fallback. BERT classifiers fall back to regex patterns; specialist classifiers return `None` and pattern matching takes over; the NER model returns chunks unchanged if absent. No part of the audit pipeline crashes because a model hasn't been trained.

**Dual backends**: Ollama (CPU-friendly, works on any hardware) or Triton (GPU, ~50–100× faster for BERT classification). The switch is a single command with no code changes.

**Bilingual**: EN and DE supported throughout — document parsing, chunk classification, regex patterns, regulatory text embedding, and evidence synonym matching all handle both languages.
