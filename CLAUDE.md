# KlarKI — EU AI Act + GDPR compliance auditor

Local-first, privacy-preserving compliance auditor for German SMEs. Analyses company documentation against EU AI Act Articles 9-15 and GDPR. Generates per-article compliance scores and actionable gap reports. Runs entirely via `docker compose up` — zero data leaves the machine.

---

## Implementation phases

This project is built in 5 phases. Complete each phase fully before starting the next. Each phase produces a working, testable system.

### Phase 1: Foundation + knowledge base
> Goal: Project skeleton, Docker Compose, ChromaDB populated with EU AI Act + GDPR

### Phase 2: Document pipeline + Ollama integration  
> Goal: Upload → parse → chunk → classify → retrieve → analyse → score → report

### Phase 3: React frontend
> Goal: Upload UI, compliance dashboard, per-article detail view, PDF download

### Phase 4: Emotion recognition module + Annex III wizard
> Goal: Art. 5 prohibited use detection, guided risk tier classification

### Phase 5: Fine-tuned BERT + Triton inference server
> Goal: Replace LLM classification with fast BERT, add spaCy NER, ONNX optimisation

---

## Tech stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Backend | FastAPI + Uvicorn | Python 3.11 |
| Frontend | React + TypeScript + Vite | React 18, Vite 5 |
| Styling | Tailwind CSS | v3 |
| LLM | Ollama + Phi-3 Mini 3.8B Q4 | latest |
| Embeddings | sentence-transformers (multilingual-e5-small) | local, no API |
| Vector DB | ChromaDB | latest |
| PDF parsing | PyMuPDF (fitz) | latest |
| DOCX parsing | python-docx | latest |
| Chunking | LangChain text splitters | latest |
| Reports | WeasyPrint | latest |
| HTTP client | httpx (async) | latest |
| Validation | Pydantic v2 | latest |
| Testing | pytest + pytest-asyncio + httpx | latest |
| Containers | Docker + Docker Compose | latest |
| Phase 5 | Triton Inference Server + ONNX Runtime | 24.02 |
| Phase 5 | deepset/gbert-base + spaCy | latest |

---

## Project structure

```
klarki/
├── CLAUDE.md
├── README.md
├── docker-compose.yml
├── docker-compose.override.yml        # Dev overrides (volume mounts, hot reload)
├── .env.example
├── .env
├── .gitignore
├── .dockerignore
│
├── api/
│   ├── Dockerfile
│   ├── Dockerfile.dev                  # Dev image with hot reload
│   ├── requirements.txt
│   ├── requirements-dev.txt            # pytest, black, ruff, mypy
│   ├── main.py                         # FastAPI app factory, CORS, lifespan events
│   ├── config.py                       # pydantic-settings: Settings class from .env
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── audit.py                    # POST /api/v1/audit/upload
│   │   │                               # GET  /api/v1/audit/{audit_id}
│   │   │                               # GET  /api/v1/audit/{audit_id}/status
│   │   └── reports.py                  # GET  /api/v1/reports/{audit_id}/pdf
│   │                                   # GET  /api/v1/reports/{audit_id}/json
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_parser.py          # Phase 2: PDF/DOCX/TXT → raw text
│   │   ├── chunker.py                  # Phase 2: raw text → list[DocumentChunk]
│   │   ├── language_detector.py        # Phase 2: detect DE/EN per chunk
│   │   ├── embedding_service.py        # Phase 2: sentence-transformers e5-small local
│   │   ├── classifier.py              # Phase 2: LLM few-shot chunk classification
│   │   │                               # Phase 5: swap to BERT via Triton gRPC
│   │   ├── rag_engine.py              # Phase 2: embed → ChromaDB search → top-k
│   │   ├── gap_analyser.py            # Phase 2: LLM structured gap analysis per article
│   │   ├── compliance_scorer.py       # Phase 2: aggregate scores + risk tier (rule-based)
│   │   ├── emotion_module.py          # Phase 4: Art. 5 emotion recognition check
│   │   ├── risk_wizard.py             # Phase 4: Annex III guided classification
│   │   ├── report_generator.py        # Phase 2: WeasyPrint HTML → PDF
│   │   ├── ollama_client.py           # Phase 2: async httpx wrapper for Ollama API
│   │   ├── chroma_client.py           # Phase 1: async ChromaDB client wrapper
│   │   └── triton_client.py           # Phase 5: gRPC client for Triton
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py                  # All Pydantic models
│   │
│   ├── prompts/
│   │   ├── classify_chunk.txt          # Few-shot classification prompt
│   │   └── gap_analysis.txt            # Per-article gap analysis prompt
│   │
│   └── templates/
│       └── report.html                 # WeasyPrint HTML template for PDF report
│
├── frontend/
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── nginx.conf                      # Production: serves built React + proxies /api
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx                     # Router setup
│       ├── api/
│       │   └── client.ts              # Axios instance, base URL from env
│       ├── hooks/
│       │   ├── useAudit.ts            # Upload + poll audit status
│       │   └── useReport.ts           # Fetch report data
│       ├── pages/
│       │   ├── Upload.tsx              # Phase 3: drag-drop + paste text + file select
│       │   ├── Dashboard.tsx           # Phase 3: overall score + 7 article cards
│       │   ├── ArticleDetail.tsx       # Phase 3: gaps, recommendations, severity
│       │   └── RiskWizard.tsx          # Phase 4: guided Annex III questionnaire
│       ├── components/
│       │   ├── Layout.tsx              # Nav bar + main content wrapper
│       │   ├── ScoreRadial.tsx         # Circular progress 0-100 with color coding
│       │   ├── ArticleCard.tsx         # Summary card per article on dashboard
│       │   ├── GapCard.tsx             # Single gap: title, desc, severity badge
│       │   ├── FileDropzone.tsx        # Drag-and-drop area with file type validation
│       │   ├── TextPasteArea.tsx       # Textarea for raw text input
│       │   ├── LoadingSpinner.tsx      # Audit processing state
│       │   ├── ReportDownload.tsx      # PDF download button
│       │   └── EmotionWarning.tsx      # Phase 4: Art. 5 prohibition alert
│       ├── types/
│       │   └── index.ts               # TypeScript interfaces matching API schemas
│       └── utils/
│           └── formatters.ts           # Score color, severity label, date formatting
│
├── scripts/
│   ├── build_knowledge_base.py         # Phase 1: download + chunk + embed + store
│   ├── seed_ollama.sh                  # Phase 2: pull phi3:mini on first run
│   ├── export_onnx.py                  # Phase 5: convert BERT to ONNX
│   └── benchmark_triton.py            # Phase 5: latency comparison
│
├── training/                           # Phase 5 only
│   ├── train_classifier.py
│   ├── train_ner.py
│   └── data/
│       ├── clause_labels.jsonl
│       └── ner_annotations.jsonl
│
├── model_repository/                   # Phase 5 only — Triton model repo
│   ├── bert_clause_classifier/
│   │   ├── 1/model.onnx
│   │   └── config.pbtxt
│   ├── e5_embeddings/
│   │   ├── 1/model.onnx
│   │   └── config.pbtxt
│   ├── spacy_ner/
│   │   ├── 1/model.py
│   │   └── config.pbtxt
│   └── compliance_ensemble/
│       └── config.pbtxt
│
├── chroma_data/                        # Generated by build_knowledge_base.py
│
└── tests/
    ├── conftest.py                     # Shared fixtures: test client, mock Ollama, seeded ChromaDB
    ├── test_parser.py
    ├── test_chunker.py
    ├── test_language_detector.py
    ├── test_classifier.py
    ├── test_rag.py
    ├── test_gap_analyser.py
    ├── test_scorer.py
    ├── test_emotion_module.py          # Phase 4
    ├── test_risk_wizard.py            # Phase 4
    ├── test_api_audit.py              # Integration test
    └── test_api_reports.py            # Integration test
```

---

## Phase 1: Foundation + knowledge base

### What to build

1. **Project skeleton**: all folders, `__init__.py` files, empty modules with docstrings
2. **docker-compose.yml**: klarki-api, klarki-chromadb, klarki-ollama, klarki-frontend (placeholder nginx)
3. **FastAPI app**: `main.py` with health check endpoint, CORS, lifespan
4. **Config**: `config.py` with pydantic-settings, `.env.example`
5. **Pydantic schemas**: all models in `models/schemas.py`
6. **ChromaDB client**: `services/chroma_client.py` — async wrapper
7. **Knowledge base builder**: `scripts/build_knowledge_base.py`
8. **Seed script**: `scripts/seed_ollama.sh`

### `models/schemas.py` — complete schema definitions

```python
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ArticleDomain(str, Enum):
    RISK_MANAGEMENT = "risk_management"
    DATA_GOVERNANCE = "data_governance"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    RECORD_KEEPING = "record_keeping"
    TRANSPARENCY = "transparency"
    HUMAN_OVERSIGHT = "human_oversight"
    SECURITY = "security"
    UNRELATED = "unrelated"

class RiskTier(str, Enum):
    PROHIBITED = "prohibited"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"

class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"

class AuditStatus(str, Enum):
    UPLOADING = "uploading"
    PARSING = "parsing"
    CLASSIFYING = "classifying"
    ANALYSING = "analysing"
    SCORING = "scoring"
    COMPLETE = "complete"
    FAILED = "failed"

class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    language: str = "en"
    domain: ArticleDomain | None = None
    metadata: dict = Field(default_factory=dict)

class GapItem(BaseModel):
    title: str
    description: str
    severity: Severity
    article_num: int

class ArticleScore(BaseModel):
    article_num: int
    domain: ArticleDomain
    score: float = Field(ge=0, le=100)
    gaps: list[GapItem] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    chunk_count: int = 0

class EmotionFlag(BaseModel):
    detected: bool = False
    context: str = ""
    is_prohibited: bool = False
    explanation: str = ""

class ComplianceReport(BaseModel):
    audit_id: str
    created_at: datetime
    source_files: list[str]
    language: str
    risk_tier: RiskTier
    overall_score: float
    article_scores: list[ArticleScore]
    emotion_flag: EmotionFlag
    total_chunks: int
    classified_chunks: int

class AuditResponse(BaseModel):
    audit_id: str
    status: AuditStatus
    report: ComplianceReport | None = None

class APIResponse(BaseModel):
    status: str = "success"
    data: dict | None = None
    error: str | None = None
```

### `scripts/build_knowledge_base.py` — what it does

1. Read EU AI Act full text (DE + EN) from bundled text files or download from EUR-Lex
2. Read GDPR full text (DE + EN)
3. Chunk each article/paragraph separately with metadata: `regulation`, `article_num`, `paragraph`, `domain`, `lang`, `is_annex`, `annex_num`
4. Embed all chunks using `sentence-transformers` `intfloat/multilingual-e5-small` locally
5. Store in ChromaDB collections: `eu_ai_act`, `gdpr`, `compliance_checklist`
6. The `compliance_checklist` collection has ~85 structured requirements from Articles 9-15, each with `requirement_id`, `article_num`, `description`, `severity`
7. Save to `chroma_data/` directory

### Phase 1 acceptance criteria

- [ ] `docker compose up -d` starts api + chromadb + ollama containers
- [ ] `GET /api/v1/health` returns `{"status": "ok", "services": {"chromadb": true, "ollama": true}}`
- [ ] ChromaDB has 3 collections with data
- [ ] All Pydantic schemas importable and validated

---

## Phase 2: Document pipeline + Ollama integration

### What to build

All services in `api/services/` except `emotion_module.py`, `risk_wizard.py`, `triton_client.py`. Both API routers. Prompt templates. Report template. Tests.

### Service specifications

#### `document_parser.py`
```python
async def parse_document(file_path: str, filename: str) -> str:
    """Extract raw text. Supports .pdf (PyMuPDF), .docx (python-docx), .txt/.md.
    Handles German characters (äöüß). Raises ValueError for unsupported types."""
```

#### `chunker.py`
```python
async def chunk_text(raw_text: str, source_file: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[DocumentChunk]:
    """LangChain RecursiveCharacterTextSplitter. Each chunk gets uuid4 chunk_id."""
```

#### `language_detector.py`
```python
async def detect_language(text: str) -> str:
    """Returns 'de' or 'en'. Uses langdetect on first 500 chars. Defaults to 'en'."""
```

#### `embedding_service.py`
```python
class EmbeddingService:
    """Local sentence-transformers multilingual-e5-small. Loaded once at startup via FastAPI lifespan."""
    def __init__(self):
        self.model = SentenceTransformer("intfloat/multilingual-e5-small")
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns list of 384-dim vectors."""
```

#### `ollama_client.py`
```python
class OllamaClient:
    """Async httpx client for Ollama API."""
    async def generate(self, prompt: str, system: str = "") -> str:
    async def generate_json(self, prompt: str, system: str = "") -> dict:
        """Uses format='json'. Retry once on parse failure."""
    async def health_check(self) -> bool:
```
POST to `{OLLAMA_HOST}/api/generate` with `{"model": model, "prompt": prompt, "system": system, "stream": false, "format": "json"}`.

#### `classifier.py`
```python
async def classify_chunks(chunks: list[DocumentChunk], ollama: OllamaClient) -> list[DocumentChunk]:
    """LLM few-shot classification. Reads prompts/classify_chunk.txt. Updates chunk.domain. Sequential."""
```

#### `rag_engine.py`
```python
async def retrieve_requirements(chunk: DocumentChunk, embedding_service: EmbeddingService, chroma_client: ChromaClient, top_k: int = 5) -> list[dict]:
    """Embed chunk → ChromaDB search → prefer same language → fallback any → return top_k."""
```

#### `gap_analyser.py`
```python
async def analyse_article(article_num: int, domain: ArticleDomain, user_chunks: list[DocumentChunk], regulatory_passages: list[dict], ollama: OllamaClient) -> ArticleScore:
    """Concatenate user chunks + regulatory text → Ollama with gap_analysis.txt → parse JSON → ArticleScore."""
```

#### `compliance_scorer.py`
```python
async def score_audit(article_scores: list[ArticleScore], chunks: list[DocumentChunk]) -> ComplianceReport:
    """overall_score = weighted average. risk_tier = classify_risk_tier(chunks)."""

def classify_risk_tier(chunks: list[DocumentChunk]) -> RiskTier:
    """Rule-based Annex III. Scan for keywords: biometric, recruitment, credit score, medical diagnosis, etc.
    Returns PROHIBITED / HIGH / LIMITED / MINIMAL."""
```

#### `report_generator.py`
```python
async def generate_pdf(report: ComplianceReport) -> bytes:
    """Jinja2 template (templates/report.html) → WeasyPrint → PDF bytes."""
```

### Router: `routers/audit.py`
```python
@router.post("/api/v1/audit/upload")  # accepts UploadFile + optional raw_text Form
@router.get("/api/v1/audit/{audit_id}")  # returns AuditResponse
@router.get("/api/v1/audit/{audit_id}/status")  # returns AuditStatus
```
Use FastAPI BackgroundTasks. Store audit state in in-memory dict.

### Router: `routers/reports.py`
```python
@router.get("/api/v1/reports/{audit_id}/pdf")  # returns StreamingResponse PDF
@router.get("/api/v1/reports/{audit_id}/json")  # returns ComplianceReport JSON
```

### Prompt: `prompts/classify_chunk.txt`
```
You classify compliance document sections into EU AI Act article categories.
Given a text chunk, respond with ONLY one label:
- risk_management (Article 9)
- data_governance (Article 10)
- technical_documentation (Article 11)
- record_keeping (Article 12)
- transparency (Article 13)
- human_oversight (Article 14)
- security (Article 15)
- unrelated

Examples:
"The model was evaluated against adversarial perturbations" → security
"Training data consists of 50,000 labelled images" → data_governance
"A certified operator reviews all recommendations" → human_oversight
"The system logs every prediction with timestamp" → record_keeping
"Users are informed this provides AI-assisted recommendations" → transparency

Classify this text. Respond with ONLY the label:
{chunk_text}
```

### Prompt: `prompts/gap_analysis.txt`
```
You are an EU AI Act compliance auditor. Compare documentation against regulatory requirements. Output ONLY valid JSON.

Company documentation about {domain_label} (Article {article_num}):
---
{user_text}
---

EU AI Act requirements (Article {article_num}):
---
{regulatory_text}
---

Respond with this exact JSON:
{"score": <0-100>, "gaps": [{"title": "<max 10 words>", "description": "<specific>", "severity": "critical|major|minor"}], "recommendations": ["<actionable>"]}
```

### Phase 2 acceptance criteria

- [ ] Upload PDF → get ComplianceReport with 7 article scores
- [ ] Upload DOCX → same result
- [ ] Paste raw text → same result
- [ ] German + English documents both work
- [ ] PDF report downloads with all sections
- [ ] Status polling works through all stages
- [ ] All unit tests pass
- [ ] Integration test: upload → complete pipeline → verify output

---

## Phase 3: React frontend

### Pages

**Upload.tsx**: Drag-drop zone (.pdf, .docx, .txt, .md, max 10MB) + textarea for paste + "Start audit" button → POST `/api/v1/audit/upload` → show processing steps → redirect to Dashboard.

**Dashboard.tsx**: Large ScoreRadial (overall), risk tier badge (red/amber/blue/green), grid of 7 ArticleCards, PDF download button.

**ArticleDetail.tsx**: Article score radial + list of GapCards sorted by severity (critical first) + recommendations.

### Components

- `ScoreRadial.tsx`: SVG circle, props: `score, size`. Red <40, amber 40-70, green >70.
- `ArticleCard.tsx`: Props: `ArticleScore`. Article num, domain, score bar, gap count.
- `GapCard.tsx`: Props: `GapItem`. Severity badge + title + description.
- `FileDropzone.tsx`: react-dropzone, validate type + size.
- `LoadingSpinner.tsx`: Step indicator for pipeline stages.
- `Layout.tsx`: Nav with "KlarKI" branding, max-w-6xl container.

### Phase 3 acceptance criteria

- [ ] Upload → processing animation → dashboard with scores
- [ ] Click article card → detail page with gaps
- [ ] PDF download works
- [ ] Responsive layout
- [ ] Error states handled

---

## Phase 4: Emotion module + Annex III wizard

### `emotion_module.py`
```python
async def check_emotion_recognition(chunks: list[DocumentChunk]) -> EmotionFlag:
    """Keyword scan: EMOTION_KEYWORDS + context (WORKPLACE/EDUCATION/COMMERCIAL).
    EMOTION + WORKPLACE/EDUCATION → prohibited. EMOTION + COMMERCIAL → high-risk flag."""
```

### `risk_wizard.py`
```python
async def guided_risk_classification(answers: dict) -> RiskTier:
    """9 yes/no questions covering Annex III categories. Any yes → HIGH. Q9 + workplace → PROHIBITED."""
```

### Frontend: `RiskWizard.tsx` + `EmotionWarning.tsx`

### Phase 4 acceptance criteria
- [ ] "Employee emotion monitoring" doc → PROHIBITED flag
- [ ] "Customer sentiment chatbot" doc → HIGH risk, no prohibition
- [ ] Risk wizard functional with correct tier output
- [ ] EmotionWarning renders on dashboard

---

## Phase 5: Fine-tuned BERT + Triton

### What to build
1. `training/train_classifier.py`: fine-tune deepset/gbert-base, 8-class, ~500 examples
2. `scripts/export_onnx.py`: torch.onnx.export
3. `model_repository/` with config.pbtxt files
4. `api/services/triton_client.py`: gRPC async client
5. Config flag `USE_TRITON` — swap classifier backend transparently
6. `scripts/benchmark_triton.py`: latency comparison

### Phase 5 acceptance criteria
- [ ] BERT >85% F1 on validation
- [ ] ONNX outputs match PyTorch
- [ ] Triton serves ensemble (BERT + e5 + spaCy NER)
- [ ] classifier.py works with both backends
- [ ] Benchmark documented

---

## Code style

- Python: type hints everywhere, Pydantic for all data, async endpoints
- `httpx.AsyncClient` for all HTTP calls
- FastAPI dependency injection for services
- Frontend: functional components, TypeScript strict, named exports, Tailwind
- API envelope: `{"status": "success"|"error", "data": {...}, "error": "..."}`
- Google-style docstrings on every public function
- `structlog` with JSON output in prod, pretty in dev
- No `print()` — use `logger.info/error/debug`

## Commands

```bash
docker compose up -d                          # Start all
docker compose up -d --profile triton         # With Triton (Phase 5)
docker compose logs -f klarki-api             # Watch logs
cd api && uvicorn main:app --reload           # Dev API
cd api && pytest -v                           # Tests
cd frontend && npm run dev                    # Dev frontend
python scripts/build_knowledge_base.py        # Rebuild ChromaDB
```

## Environment variables

```env
API_PORT=8000
API_HOST=0.0.0.0
DEBUG=true
OLLAMA_HOST=http://klarki-ollama:11434
OLLAMA_MODEL=phi3:mini
CHROMADB_HOST=http://klarki-chromadb:8000
EMBEDDING_MODEL=intfloat/multilingual-e5-small
UPLOAD_MAX_SIZE_MB=10
UPLOAD_DIR=/data/uploads
USE_TRITON=false
TRITON_HOST=klarki-triton
TRITON_GRPC_PORT=8001
VITE_API_URL=http://localhost:8000
```

## Constraints

- NEVER add internet access to runtime containers
- NEVER call external LLM APIs
- Accept only: .pdf, .docx, .txt, .md
- Max upload: 10MB
- ChromaDB collections: `eu_ai_act`, `gdpr`, `compliance_checklist`
- Ollama processes one request at a time — sequential, not parallel
- Embedding model loaded once at startup via FastAPI lifespan
- Phase 5 behind `USE_TRITON` flag — never break earlier phases
