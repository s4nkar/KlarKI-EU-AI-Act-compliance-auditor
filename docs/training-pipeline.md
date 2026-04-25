# Training Pipeline

All training runs inside the `klarki-training` Docker container (Python 3.11). The host machine's Python version is irrelevant. Run with `./run.sh setup` (first time) or `./run.sh retrain` (subsequent).

## Stage order

```
Stage 1    seed-ollama              Pull phi3:mini into Ollama cache
Stage 1.5  seed-nli                 Download NLI cross-encoder to cache
Stage 2    knowledge-base           Chunk + embed regulatory text → ChromaDB
Stage 2.5  build-graph              Extract obligation schemas from regulatory text → JSONL
Stage 3    generate-data            Synthetic BERT training data via Ollama
Stage 4    train-bert               Fine-tune deepset/gbert-base (8 classes)
Stage 3.5  generate-specialist-data Synthetic actor/risk/prohibited data via Ollama
Stage 4.5  train-specialist         Fine-tune 3 specialist classifiers
Stage 5    generate-ner-data        Deterministic NER template expansion (no Ollama)
Stage 6    train-ner                Train spaCy NER model
Stage 7    export-bert              BERT → ONNX  (Triton only, skipped by default)
Stage 8    export-e5                e5-small → ONNX  (Triton only, skipped by default)
Stage 9    benchmark                Ollama vs Triton latency (Triton only, skipped by default)
```

Stages 7–9 are skipped by `./run.sh setup` and `./run.sh retrain` by default. They run automatically when you call `./run.sh triton`.

Each stage auto-skips if its output already exists. Pass `--retrain` to re-run if data changed, `--force-retrain` to always produce a new model version.

---

## Stage 2: ChromaDB knowledge base

**Input:** `data/regulatory/**/*.txt`

Each file has this format:
```
ARTICLE: 9
REGULATION: eu_ai_act
DOMAIN: risk_management

=== EN ===
Article 9 — Risk Management System
...

=== DE ===
Artikel 9 — Risikomanagementsystem
...
```

**Process:**
1. Parse each file, extract EN and DE sections separately
2. Chunk with `RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)`
3. Embed each chunk with `intfloat/multilingual-e5-small` (runs locally on CPU)
4. Upsert to ChromaDB with metadata: `{article_num, regulation, domain, lang, title}`

**Output collections:**

| Collection | Content | Approximate size |
|---|---|---|
| `eu_ai_act` | Articles 5, 9–15 (EN + DE) | ~200 chunks |
| `gdpr` | Articles 5, 6, 24, 25, 30, 35 (EN + DE) | ~150 chunks |
| `compliance_checklist` | Structured requirement sentences | ~85 chunks |

**To rebuild:** `./run.sh setup --rebuild-kb` or `python scripts/setup.py --only knowledge-base --rebuild-kb`

---

## Stage 2.5: Obligation schemas ("build-graph")

**Input:** `data/regulatory/**/*.txt` via Ollama (phi3:mini)

Ollama reads each article and extracts machine-readable obligation schemas: what evidence a compliant organisation must produce, which actor types it applies to, what the obligation requires.

**Output:** `data/obligations/eu_ai_act/article_6_annex_iii.jsonl`

Each record:
```json
{
  "article_num": 6,
  "obligation_id": "annex_iii_biometric",
  "actor": ["provider", "deployer"],
  "evidence_required": ["risk register", "conformity assessment", "technical documentation"],
  "description": "High-risk AI systems in biometric identification..."
}
```

Used at inference time by `evidence_mapper.py` to check what documents the audited organisation must have. This is NOT a graph in the ML sense — it's a structured checklist derived from law text.

**To regenerate:** `python scripts/setup.py --only build-graph --gen-overwrite`

---

## Stage 3: BERT training data generation

**How it works:** Ollama (phi3:mini) is given the actual regulatory article text as context and asked to generate realistic synthetic sentences that would appear in compliance documents.

**Default output size:**
```
8 classes × 400 examples × 2 languages (EN + DE) = 6,400 synthetic examples
```

Each record:
```json
{"text": "The risk management system shall identify and analyse the known risks...",
 "label": "risk_management", "lang": "en", "source": "generated"}
```

**Labels → articles:**
| Label | Article | What it represents |
|---|---|---|
| `risk_management` | Article 9 | Risk registers, hazard identification, mitigation |
| `data_governance` | Article 10 | Training data quality, bias, labelling |
| `technical_documentation` | Article 11 | System architecture, design docs, Annex IV |
| `record_keeping` | Article 12 | Audit logs, event logs, traceability |
| `transparency` | Article 13 | User disclosures, capability notices |
| `human_oversight` | Article 14 | Human control mechanisms, override capability |
| `security` | Article 15 | Robustness, adversarial attacks, accuracy thresholds |
| `unrelated` | — | Non-compliance text (general business writing) |

**Output file:** `training/data/clause_labels.jsonl`

**To change training data size:**
```bash
# Double the default (800 per class × 8 × 2 = 12,800 examples)
python scripts/setup.py --only generate-data --gen-per-class 800 --gen-overwrite

# Quick smoke test (20 per class)
python scripts/setup.py --only generate-data --gen-per-class 20 --gen-overwrite

# Via run.sh retrain with custom size
python scripts/setup.py --retrain --gen-per-class 600
```

---

## Stage 3.5: Specialist classifier data generation

Same process as BERT data, but for three binary/multi-class classifiers:

| File | Classes | Purpose |
|---|---|---|
| `actor_labels.jsonl` | provider, deployer, importer, distributor | Detect Article 3 actor role |
| `risk_labels.jsonl` | high_risk, not_high_risk | Detect Annex III high-risk systems |
| `prohibited_labels.jsonl` | prohibited, not_prohibited | Detect Article 5 prohibitions |

**To change specialist training data size:**
```bash
# The --gen-per-class flag applies to specialist data too
python scripts/setup.py --only generate-specialist-data --gen-per-class 300 --gen-overwrite
```

---

## Stage 4: BERT training

**Base model:** `deepset/gbert-base` (110M parameters, German BERT)

**Train/val split:** 85% train / 15% validation, stratified by class, `seed=42`

There is no separate holdout test set in the automated pipeline. Evaluation against hand-crafted golden datasets is in `tests/evaluation/`.

**Default hyperparameters:**
```
Epochs:       12 (with early stopping — usually stops at 5–8)
Batch size:   16 (effective 32 via gradient accumulation ×2)
Learning rate: 2e-5 with cosine decay
Max length:   256 tokens
Early stopping: yes — monitors eval macro F1, patience=3 epochs
```

**Training targets:**
- Production target: macro F1 ≥ 0.93 on validation set
- Acceptable: macro F1 ≥ 0.85
- Below 0.85: training script prints a warning; consider more data or epochs

**Output:** `training/artifacts/bert_classifier/` (weights, config, metrics.json, checkpoints)

**To change BERT training parameters:**
```bash
python scripts/setup.py --only train-bert --bert-epochs 20 --bert-batch 8

# Or directly:
python training/scripts/train_classifier.py \
  --data training/data/clause_labels.jsonl \
  --output training/artifacts/bert_classifier \
  --epochs 20 --batch-size 8 --lr 3e-5
```

---

## Stage 4.5: Specialist classifiers training

Three classifiers trained separately using the same BERT fine-tuning approach:

| Classifier | Directory | Used in inference when |
|---|---|---|
| Actor | `training/artifacts/actor_classifier/` | ML confidence ≥ 0.80 → ML wins; else pattern fallback |
| Risk | `training/artifacts/risk_classifier/` | ML confidence ≥ 0.85 → augments Annex III pattern detection |
| Prohibited | `training/artifacts/prohibited_classifier/` | ML confidence ≥ 0.85 → augments Article 5 pattern detection |

If a model isn't trained yet, `ml_classifiers.predict_*()` returns `None` and the system falls back entirely to regex patterns. The pipeline degrades gracefully — it never crashes because a specialist model is missing.

---

## Stage 5: NER data generation

**No Ollama required** — fully deterministic template expansion. Runs in seconds.

Builds `training/data/ner_annotations.jsonl` from:
1. Sentences extracted from `data/regulatory/**/*.txt` (real regulatory text)
2. Template expansion: controlled vocabularies × sentence templates → records with exact character offsets

**Default size:** 5,000 records

**Entity types trained:**
| Entity | Examples |
|---|---|
| `ARTICLE` | "Article 9", "Art. 14", "GDPR Article 35" |
| `OBLIGATION` | "must document", "shall maintain", "are required to" |
| `ACTOR` | "providers", "operators", "notified bodies" |
| `AI_SYSTEM` | "high-risk AI system", "emotion recognition system" |
| `RISK_TIER` | "high-risk", "prohibited", "hochriskant" |
| `PROCEDURE` | "conformity assessment", "risk management system" |
| `REGULATION` | "EU AI Act", "GDPR", "DSGVO" |
| `PROHIBITED_USE` | "social scoring", "real-time biometric surveillance" |

**To change NER data size:**
```bash
python scripts/setup.py --only generate-ner-data --ner-templates 10000 --gen-overwrite
```

---

## Stage 6: NER training

**Base model:** `de_core_news_lg` (spaCy pre-trained German, tok2vec + 560k word vectors)

**Default hyperparameters:**
```
Epochs:    60
Batch:     32
Patience:  10 (early stopping)
```

**Output:** `training/artifacts/spacy_ner_model/model-final`

**To change NER training parameters:**
```bash
python training/scripts/train_ner.py \
  --data training/data/ner_annotations.jsonl \
  --output training/artifacts/spacy_ner_model \
  --epochs 80 --patience 15
```

---

## Retraining after data changes

```bash
# Smart retrain: only retrains if data hash changed since last run
./run.sh retrain

# Force retrain everything regardless
python scripts/setup.py --force-retrain

# Retrain only one model
python scripts/setup.py --only train-bert
python scripts/setup.py --only train-specialist
python scripts/setup.py --only train-ner

# Regenerate data only (no retrain)
python scripts/setup.py --only generate-data --gen-overwrite
python scripts/setup.py --only generate-specialist-data --gen-overwrite

# Full pipeline with custom data size
python scripts/setup.py --retrain --gen-per-class 600 --ner-templates 8000
```

## Weak supervision (alternative data source)

`scripts/build_weak_supervision_labels.py` is a standalone script that produces `actor_labels.jsonl`, `risk_labels.jsonl`, and `prohibited_labels.jsonl` using regex rules on regulatory text — no Ollama required. It is **not called by the setup pipeline** and is an opt-in alternative:

```bash
# Generate specialist data without Ollama
python scripts/build_weak_supervision_labels.py --type all

# Then train on it
python scripts/setup.py --only train-specialist
```

Use this if Ollama is unavailable or you want reproducible, deterministic training data.
