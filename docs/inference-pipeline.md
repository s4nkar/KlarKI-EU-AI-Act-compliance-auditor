# Inference Pipeline

Every audit runs through the same async pipeline, triggered by `POST /api/v1/audit/upload` as a FastAPI background task. The browser polls `GET /api/v1/audit/{id}/status` every 2 seconds until `complete` or `failed`.

## Full pipeline

```
POST /api/v1/audit/upload
  file (PDF/DOCX/TXT/MD, max 10 MB) or raw_text field
  optional: wizard_risk_tier
  │
  ▼ status: PARSING
┌─────────────────────────────────────────────────────────────────┐
│ document_parser.py                                              │
│   PyMuPDF (fitz)    → prose text from PDF                      │
│   pdfplumber        → table extraction (tab-separated rows)    │
│   pytesseract       → OCR fallback for scanned/image PDFs      │
│   python-docx       → DOCX extraction                          │
│   plain read        → TXT / MD                                 │
│   → raw_text (string)                                          │
│                                                                 │
│ proposition_chunk_text()                                        │
│   1. Detect section headings via 6 regex patterns               │
│      (## Markdown, 1.1 Numbered, Article N, ALLCAPS, etc.)     │
│   2. Split colon-introduced obligation lists:                   │
│      "Provider shall: (a) maintain logs; (b) ensure trace"      │
│      → Chunk A: "Provider shall: (a) maintain logs"            │
│      → Chunk B: "Provider shall: (b) ensure traceability"      │
│      Conditional clauses ("shall X if Y") are never split      │
│   3. Sub-split sections > 800 chars by paragraph               │
│   4. Merge fragments < 80 chars into preceding chunk            │
│   Each chunk: chunk_id, text, section_heading, is_proposition  │
│   → List[DocumentChunk]                                        │
│                                                                 │
│ detect_language()                                               │
│   langdetect → "en" | "de"                                     │
│   Fallback: "en" for text < 100 chars                          │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼ status: CLASSIFYING
┌─────────────────────────────────────────────────────────────────┐
│ [concurrent via asyncio.to_thread]                              │
│                                                                 │
│ actor_classifier.py                                             │
│   Who is the user? (Article 3 definition)                      │
│   ML path: ml_classifiers.predict_actor(raw_text[:2000])       │
│     if confidence ≥ 0.80 → ML result used directly             │
│   Pattern fallback: 39 EN+DE regex patterns                     │
│     14 PROVIDER + 13 DEPLOYER + 6 IMPORTER + 6 DISTRIBUTOR    │
│   Default: DEPLOYER (most SMEs are deployers)                  │
│   → ActorClassification(actor_type, confidence, signals)       │
│                                                                 │
│ applicability_engine.py   [4-step deterministic gate]          │
│   Step 1: Article 5 prohibited patterns (9 patterns EN+DE)     │
│     + predict_prohibited() ML if confidence ≥ 0.85             │
│     → if triggered: is_prohibited=True, articles=[5], STOP     │
│   Step 2: Annex III (8 category pattern sets, 60+ patterns)    │
│     BIOMETRIC, CRITICAL_INFRASTRUCTURE, EDUCATION,             │
│     EMPLOYMENT, ESSENTIAL_SERVICES, LAW_ENFORCEMENT,           │
│     MIGRATION, JUSTICE                                          │
│   Step 3: Article 6(1) Annex I safety-component signals        │
│     14 patterns (CE marking, MDR/IVDR, notified body, etc.)    │
│     annex_i_triggered = len(hits) ≥ 2 (avoids false positives) │
│   Step 4: predict_high_risk() ML if confidence ≥ 0.85          │
│     catches Annex III cases pattern matching missed             │
│   is_high_risk = Step2 OR Step3 OR ML                          │
│   applicable_articles = [9,10,11,12,13,14,15] or []            │
│   → ApplicabilityResult                                        │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ classify_chunks()  ← BERT inference (Ollama or Triton)         │
│                                                                 │
│ Ollama mode (USE_TRITON=false):                                 │
│   Few-shot prompt per chunk → phi3:mini → label string         │
│   Sequential, ~5–10s per chunk                                  │
│                                                                 │
│ Triton mode (USE_TRITON=true):                                  │
│   Tokenise client-side (BertTokenizer)                         │
│   Batch to Triton gRPC → BERT ONNX inference                   │
│   ~50ms per batch of 32 chunks                                 │
│                                                                 │
│ Each chunk gets:                                               │
│   chunk.domain = ArticleDomain enum value                      │
│   (risk_management / data_governance / ... / unrelated)        │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ enrich_chunks_with_ner()  ← spaCy NER                         │
│                                                                 │
│ Runs the trained spaCy NER model on each chunk (capped at      │
│ 1000 chars per chunk to limit latency)                         │
│                                                                 │
│ For each chunk:                                                 │
│   chunk.metadata["ner_entities"] = {                           │
│     "ARTICLE": ["Article 9", "Art. 14"],                       │
│     "OBLIGATION": ["shall maintain", "must document"],         │
│     "ACTOR": ["providers", "notified bodies"],                 │
│     ...                                                        │
│   }                                                            │
│                                                                 │
│ Domain correction (conservative):                              │
│   if chunk.domain == UNRELATED                                  │
│   AND NER finds exactly ONE Article ref (9–15)                 │
│   → correct domain to matching ArticleDomain                   │
│   (Recovers short paragraphs the BERT classifier missed)       │
│                                                                 │
│ If NER model not trained → chunks returned unchanged (no crash)│
└─────────────────────────────────────────────────────────────────┘
  │
  ▼ status: ANALYSING
┌─────────────────────────────────────────────────────────────────┐
│ Group chunks by domain (7 domain buckets)                      │
│                                                                 │
│ asyncio.gather — all 7 articles run concurrently:             │
│                                                                 │
│ process_article(article_num=9, domain=RISK_MANAGEMENT, ...)    │
│   ├─ if article_num NOT in applicable_articles                 │
│   │    → ArticleScore(score=100, gaps=[], "Not applicable")    │
│   │    → NO LLM call, NO ChromaDB query                        │
│   │                                                            │
│   ├─ if no chunks for this domain                              │
│   │    → ArticleScore(score=0, gaps=["No evidence found"])     │
│   │    → NO LLM call                                           │
│   │                                                            │
│   └─ retrieve_requirements() + analyse_article()              │
│        [see rag.md for RAG detail]                             │
│        LangGraph 3-node graph:                                 │
│          legal_agent_node                                      │
│            Prompt: extract strict checklist from               │
│            top-8 regulatory passages                           │
│            → extracted_requirements: list[str]                 │
│          technical_agent_node                                  │
│            Prompt: evaluate user doc chunks against            │
│            each requirement — found / partial / missing        │
│            → evidence_findings: dict[req → finding]           │
│          synthesis_agent_node                                  │
│            Prompt: compile gap report JSON                     │
│            → score (0–100), gaps[], recommendations[],         │
│               reasoning                                        │
│        = 3 Ollama calls per applicable article                 │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼ status: SCORING
┌─────────────────────────────────────────────────────────────────┐
│ evidence_mapper.py  ← deterministic + NLI cross-encoder        │
│   Load obligation schemas from data/obligations/**/*.jsonl     │
│   Filter by: actor_type.value in ob["actor"]                   │
│          AND: article_num in applicable_articles               │
│                                                                 │
│   For each obligation's evidence_required items:               │
│     Fast path (regex synonym dict):                            │
│       22 canonical terms × ~8 synonyms each                    │
│       e.g. "risk register" → ["risk catalog", "hazard log",    │
│            "risikokatalog", "gefährdungsregister", ...]        │
│     Slow path (NLI, only if regex misses):                     │
│       CrossEncoder("cross-encoder/nli-deberta-v3-small")       │
│       Premise = chunk.text                                     │
│       Hypothesis = "This document contains a <term>."          │
│       → ENTAILMENT class predicted → semantic match            │
│   → EvidenceMap(fully_satisfied, partially_satisfied,          │
│                 missing, overall_coverage %)                   │
│                                                                 │
│ check_emotion_recognition()  ← Article 5 scan                 │
│   Detects emotion recognition / biometric / social scoring    │
│   Context-aware: workplace + education context → prohibited    │
│   → EmotionFlag(detected, is_prohibited, explanation)         │
│                                                                 │
│ compliance_scorer.py                                           │
│   Risk tier (authoritative from applicability):                │
│     is_prohibited → PROHIBITED                                 │
│     is_high_risk  → HIGH                                       │
│     else          → MINIMAL                                    │
│   Overall score = avg(applicable articles only)                │
│     Non-applicable articles score 100, excluded from mean      │
│     Minimal-risk system → 100%                                 │
│   Confidence score:                                            │
│     mean(actor.confidence,                                     │
│          0.5 + evidence_coverage/2,                            │
│          classified_chunks / total_chunks)                     │
│   requires_human_review = confidence < 0.70 OR actor=UNKNOWN  │
│   → ComplianceReport                                          │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼ status: COMPLETE
  Uploaded file deleted from disk
  ComplianceReport stored in-memory (keyed by audit_id UUID)

GET /api/v1/reports/{id}/pdf  → WeasyPrint renders Jinja2 template → PDF bytes
GET /api/v1/reports/{id}/json → ComplianceReport as JSON
```

## LangGraph state machine

The gap analysis uses a linear 3-node `StateGraph[AuditState]` — one graph instance per applicable article, invoked once per article.

```
AuditState (TypedDict):
  Input fields:
    article_num           int
    domain                ArticleDomain
    user_chunks           List[DocumentChunk]   (top 10 by domain)
    regulatory_passages   List[RegulatoryPassage] (from RAG)
    ollama_client         OllamaClient

  Output fields (populated by nodes):
    extracted_requirements  list[str]
    evidence_findings       dict[str, str]
    final_score             int
    gaps                    list[str]
    recommendations         list[str]
    reasoning               str

Graph topology:
  START → legal_agent_node → technical_agent_node → synthesis_agent_node → END

All three nodes are async. Each independently error-handled — returns safe
defaults (empty lists, score=30) on exception rather than crashing.
Total Ollama calls per applicable article: 3
```

## Inference backends

### Ollama (default, USE_TRITON=false)
- phi3:mini 3.8B Q4 — runs on CPU or GPU via Ollama
- Chunk classification: few-shot prompt, one call per chunk, sequential
- Gap analysis: 3 calls per applicable article (legal + technical + synthesis)
- `temperature=0, seed=42, top_k=1` — fully deterministic outputs

### Triton (USE_TRITON=true, GPU required)
- BERT ONNX served via gRPC on port 8003
- Chunk classification: tokenise locally → batch 32 → Triton → logits → argmax
- ~50ms per batch-32 vs ~5–10s per chunk with Ollama
- Gap analysis still uses Ollama (LangGraph agents are LLM-based, not BERT)
- e5-small ONNX also served via Triton for faster embedding
