# RAG System

KlarKI uses a hybrid retrieval pipeline: BM25 keyword search + dense vector search, merged with Reciprocal Rank Fusion, then re-ranked by a cross-encoder. Metadata filtering is applied at every stage to keep retrieval both fast and precise.

## Why hybrid RAG?

| Method | Strength | Weakness |
|---|---|---|
| BM25 (keyword) | Exact legal term matching ("conformity assessment", "Artikel 9") | Misses paraphrase and German↔English synonyms |
| Vector (dense) | Semantic similarity, handles synonyms and paraphrase | Misses exact rare terms; can drift from legal precision |
| Hybrid (BM25 + vector + RRF) | Gets both exact and semantic matches | More computation |
| Cross-encoder re-rank | Directly scores relevance of each (query, passage) pair | Slow on large candidate sets → run only on RRF output |

## Full retrieval pipeline

```
User chunk (text + domain + language metadata)
  │
  ├── [parallel]
  │     ▼
  │   BM25 search (rank_bm25 in-memory, article_num partitioned)
  │     Query: chunk.text (tokenised, lowercased, punctuation stripped)
  │     Filter: article_num = DOMAIN_TO_ARTICLE[chunk.domain]
  │     Collections: eu_ai_act, compliance_checklist
  │     → top 10 results per collection
  │
  └── [parallel]
        ▼
      Vector search (ChromaDB + e5-small embedding)
        Query: embed(chunk.text) via multilingual-e5-small
        Filter: where={"article_num": N}  ← ChromaDB server-side where-filter
        Optional: where={"article_num": N, "regulation": "eu_ai_act"}
        Collections: eu_ai_act, compliance_checklist
        → top 10 results per collection
  │
  ▼
Language-aware sort (soft preference, not hard filter)
  Each result list sorted: same-language results first
  e.g. German chunk → German regulatory passages ranked higher
  (English passages still included — bilingual docs need both)
  │
  ▼
RRF merge (k=60)
  RRF score = Σ 1 / (60 + rank_i)
  Deduplicates by passage id
  → ~15 unique candidate passages
  │
  ▼
Cross-encoder re-ranking
  Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  Input: [(chunk.text, passage.text), ...] — one pair per candidate
  Output: relevance score per pair
  Sorted descending → top 5 returned
  Runs via asyncio.to_thread (blocking inference, doesn't stall event loop)
  Falls back to RRF order if cross-encoder unavailable
  │
  ▼
List[RegulatoryPassage] (top 5) → passed to LangGraph gap analysis
```

## Metadata filtering — how it reduces latency

This is the biggest latency saver in the pipeline.

### Article_num filter (most important)
Every regulatory passage in ChromaDB is stored with `article_num` metadata. The BM25 index is also partitioned by `article_num` at startup. When retrieving for a chunk of domain `risk_management`, only article_9 passages are searched — reducing the candidate space from ~500 total passages to ~25–30.

```python
# In rag_engine.py
article_num = _DOMAIN_TO_ARTICLE.get(chunk.domain)  # e.g. 9 for risk_management
where_filter = {"article_num": article_num}           # passed to ChromaDB query
# BM25 directly uses the pre-partitioned index: _bm25.search(collection, query, article_num=9)
```

### Applicability gate (even bigger win)
Before RAG runs at all, `applicability_engine.py` determines which articles apply. If `article_num NOT in applicable_articles`, `retrieve_requirements()` returns `[]` immediately — no ChromaDB query, no BM25 search, no cross-encoder. For a MINIMAL-risk system (no articles apply), the entire RAG + LangGraph analysis is skipped for all 7 articles.

```python
if applicable_articles is not None and article_num is not None:
    if article_num not in applicable_articles:
        return []  # immediate return, no queries
```

### Regulation filter (additive)
When the `regulation` parameter is passed (e.g. `"eu_ai_act"`), it's added to the ChromaDB `where` filter and as a post-filter on BM25 results. Prevents GDPR passages appearing in EU AI Act article analysis.

### Language (soft sort)
Not a hard filter — results are sorted to prefer same-language passages. This is intentional: a German document should still match against English regulatory text when German passages don't cover a topic.

## OpenSearch: optional BM25 replacement

By default BM25 runs in-memory (`rank_bm25` library) — fast and requires no extra service. For large document corpora in production, you can swap to OpenSearch:

```bash
# Start OpenSearch container
docker compose --profile opensearch up -d

# Enable in .env
USE_OPENSEARCH=true

# Index regulatory text into OpenSearch (in addition to ChromaDB)
python scripts/build_knowledge_base.py --opensearch
```

When `USE_OPENSEARCH=true`, `rag_engine.py` sends BM25 queries to OpenSearch instead of rank_bm25. Vector search remains in ChromaDB — OpenSearch does not replace ChromaDB.

OpenSearch advantages for production:
- Persistent index (survives API container restart without rebuild)
- Native server-side language/article/regulation filtering
- Better scaling for large corpora (thousands of documents)
- BM25 with Elasticsearch-style scoring (BM25+ with field boosting)

For public demo use, rank_bm25 in-memory is perfectly adequate.

## ChromaDB collections

| Collection | Used for | Article filter |
|---|---|---|
| `eu_ai_act` | EU AI Act articles 5, 9–15 | article_num=5 or 9–15 |
| `gdpr` | GDPR articles 5,6,24,25,30,35 | (queried separately, future) |
| `compliance_checklist` | Structured requirement sentences | article_num maps to domain |

Both `eu_ai_act` and `compliance_checklist` are queried for every RAG request. `gdpr` is currently populated but not yet wired into the RAG retrieval path (planned for GDPR gate in Subphase D).

## Cross-encoder models

Two different cross-encoders are used for two different purposes:

| Model | Where | Purpose |
|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | `rag_engine._rerank()` | Re-rank RAG candidates by relevance to the query chunk |
| `cross-encoder/nli-deberta-v3-small` | `evidence_mapper.py` slow path | NLI entailment: does this chunk prove this evidence term exists? |

They are loaded lazily (on first use) and cached globally. Neither is used during training.
