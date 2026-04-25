# ML Models

KlarKI uses 8 models at inference time. Five are trained locally during setup; three are downloaded pre-trained.

## Trained locally during setup

### 1. BERT clause classifier (`training/artifacts/bert_classifier/`)

| Property | Value |
|---|---|
| Base model | `deepset/gbert-base` (German BERT, 110M parameters) |
| Task | 8-class sequence classification |
| Input | Document chunk text (max 256 tokens) |
| Output | ArticleDomain label + confidence |
| Training data | `training/data/clause_labels.jsonl` (6,400 synthetic examples) |
| Train/val split | 85% / 15%, stratified, seed=42 |
| Target metric | Macro F1 ≥ 0.93 on validation set |

**Role in inference:** Classifies every user document chunk into one of 8 compliance domains. This determines which regulatory article each chunk is tested against. Domain corrections from the NER model can override `UNRELATED` classifications after this step.

**Inference backend:** Ollama (few-shot prompt) or Triton (ONNX, GPU batch inference).

---

### 2. Actor classifier (`training/artifacts/actor_classifier/`)

| Property | Value |
|---|---|
| Base model | `deepset/gbert-base` |
| Task | 4-class classification |
| Classes | provider, deployer, importer, distributor |
| Input | First 2,000 characters of the document |
| Output | Actor type + confidence |
| Training data | `training/data/actor_labels.jsonl` |

**Role in inference:** Determines the user's Article 3 role. Applied at the start of every audit. If ML confidence ≥ 0.80, the ML result is used directly. Otherwise, 39 regex patterns (EN+DE) are used as fallback. Default when nothing matches: DEPLOYER (most SMEs are deployers).

**Why this matters:** The actor role determines which obligations apply. A PROVIDER has different requirements than a DEPLOYER under Articles 9–15.

---

### 3. Risk classifier (`training/artifacts/risk_classifier/`)

| Property | Value |
|---|---|
| Base model | `deepset/gbert-base` |
| Task | Binary classification |
| Classes | high_risk, not_high_risk |
| Input | Concatenated chunk texts (first 2,000 chars) |
| Output | Label + confidence |
| Training data | `training/data/risk_labels.jsonl` |

**Role in inference:** Augments the Annex III pattern detection in `applicability_engine.py`. Only applied when ML confidence ≥ 0.85 — catches Annex III high-risk cases that the 60+ regex patterns missed. Either the pattern OR the ML model can trigger high-risk classification.

---

### 4. Prohibited classifier (`training/artifacts/prohibited_classifier/`)

| Property | Value |
|---|---|
| Base model | `deepset/gbert-base` |
| Task | Binary classification |
| Classes | prohibited, not_prohibited |
| Input | Concatenated chunk texts |
| Output | Label + confidence |
| Training data | `training/data/prohibited_labels.jsonl` |

**Role in inference:** Augments Article 5 prohibition detection. Only applied when ML confidence ≥ 0.85. Either the 9 regex patterns OR the ML model can trigger `is_prohibited=True`, which sets `applicable_articles=[5]` and stops further processing (no LLM gap analysis needed).

---

### 5. spaCy NER model (`training/artifacts/spacy_ner_model/model-final`)

| Property | Value |
|---|---|
| Base model | `de_core_news_lg` (pre-trained German, tok2vec + 560k vectors) |
| Task | Named entity recognition |
| Entity types | 8 (ARTICLE, OBLIGATION, ACTOR, AI_SYSTEM, RISK_TIER, PROCEDURE, REGULATION, PROHIBITED_USE) |
| Input | Chunk text (capped at 1,000 chars) |
| Output | Span list with labels |
| Training data | `training/data/ner_annotations.jsonl` (5,000 records) |

**Role in inference:** Runs after BERT classification on every chunk.
1. Adds `ner_entities` metadata dict to each chunk (used in report and evidence mapping)
2. Domain correction: if chunk.domain is UNRELATED but NER finds exactly one Article reference (9–15), the domain is corrected to the matching ArticleDomain — recovers short procedural paragraphs BERT missed

If the model hasn't been trained yet, all functions return chunks unchanged (graceful degradation).

---

## Pre-trained, downloaded at runtime

### 6. multilingual-e5-small (`intfloat/multilingual-e5-small`)

| Property | Value |
|---|---|
| Source | HuggingFace Hub (downloaded at API startup) |
| Task | Sentence embedding |
| Dimension | 384 |
| Languages | 100+ (EN + DE both well-supported) |
| Runs on | CPU (inside API container) |

**Role in inference:** Embeds document chunks for ChromaDB vector search and embeds query vectors for retrieval. Loaded once at startup and stored on `app.state.embeddings`. Also exportable to ONNX for Triton (`./run.sh triton` does this automatically).

---

### 7. ms-marco cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

| Property | Value |
|---|---|
| Source | HuggingFace Hub (downloaded on first use) |
| Task | Passage re-ranking (pointwise relevance scoring) |
| Input | (query text, passage text) pairs |
| Output | Relevance score per pair |

**Role in inference:** RAG re-ranking step. After BM25 + vector search + RRF merge produces ~15 candidate regulatory passages, this model scores each (chunk, passage) pair and keeps the top 5. Falls back to RRF order if unavailable.

---

### 8. nli-deberta cross-encoder (`cross-encoder/nli-deberta-v3-small`)

| Property | Value |
|---|---|
| Source | HuggingFace Hub (downloaded at setup stage 1.5) |
| Task | Natural language inference (entailment detection) |
| Input | (premise, hypothesis) pairs |
| Output | ENTAILMENT / NEUTRAL / CONTRADICTION |

**Role in inference:** Evidence mapping slow path. For each required evidence term (e.g. "risk register"), if regex synonym matching fails, this model checks: "Does this chunk *entail* that the document contains a risk register?" Premise = chunk.text, Hypothesis = "This document contains a risk register." If ENTAILMENT is predicted → evidence matched.

Pre-downloaded in Stage 1.5 of setup (`seed-nli`) to avoid cold-start on first audit.

---

## Model loading and graceful degradation

All trained models (1–5) are lazy-loaded on first use and cached. If a model hasn't been trained yet:
- BERT / specialist classifiers → `ml_classifiers.predict_*()` returns `None` → system falls back to regex patterns
- spaCy NER → `enrich_chunks_with_ner()` returns chunks unchanged (no domain corrections, no entity metadata)
- The audit pipeline never crashes because a trained model is missing

Pre-trained models (6–8) are downloaded automatically during setup or on first use.

## Model versioning

If `training/version_manager.py` is available, `setup.py` tracks:
- Data content hashes — detects when training data actually changed
- Model versions — saves each training run to a versioned directory, promotes the best
- `--retrain` mode uses data hashes to skip retraining when data didn't change
- `--force-retrain` always trains a new version regardless
