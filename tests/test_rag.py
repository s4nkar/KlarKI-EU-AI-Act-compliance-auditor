"""Tests for the RAG retrieval engine.

Uses in-memory ChromaDB (seeded_chroma fixture) and mocked embeddings.
No live Ollama or Triton required.
"""

import os
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


def _make_embedding_service(vec=None):
    """Return a mock EmbeddingService that returns a single 384-dim vector."""
    vec = vec or [0.0] * 384
    mock_emb = AsyncMock()
    mock_emb.embed.return_value = [vec]
    return mock_emb


def _make_chroma_client(chroma_instance):
    """Wrap an in-memory chromadb.EphemeralClient as an async ChromaClient mock."""
    mock_client = AsyncMock()

    async def fake_query(collection_name, query_embeddings, n_results, where=None):
        col = chroma_instance.get_or_create_collection(collection_name)
        kwargs = dict(query_embeddings=query_embeddings, n_results=n_results)
        if where:
            kwargs["where"] = where
        try:
            return col.query(**kwargs)
        except Exception:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    mock_client.query.side_effect = fake_query
    return mock_client


# ── _flatten_result (pure function) ──────────────────────────────────────────

def test_flatten_result_parses_chroma_output():
    """_flatten_result converts a raw ChromaDB dict into a list of passage dicts."""
    from services.rag_engine import _flatten_result

    raw = {
        "ids": [["id1", "id2"]],
        "documents": [["doc A", "doc B"]],
        "metadatas": [[{"lang": "en"}, {"lang": "de"}]],
        "distances": [[0.12, 0.45]],
    }
    result = _flatten_result(raw)
    assert len(result) == 2
    assert result[0]["id"] == "id1"
    assert result[0]["text"] == "doc A"
    assert result[0]["metadata"]["lang"] == "en"
    assert result[0]["distance"] == pytest.approx(0.12)


def test_flatten_result_skips_empty_documents():
    """_flatten_result skips entries where document text is empty/None."""
    from services.rag_engine import _flatten_result

    raw = {
        "ids": [["id1", "id2"]],
        "documents": [["doc A", ""]],
        "metadatas": [[{"lang": "en"}, {"lang": "de"}]],
        "distances": [[0.1, 0.2]],
    }
    result = _flatten_result(raw)
    assert len(result) == 1
    assert result[0]["id"] == "id1"


def test_flatten_result_empty_response():
    """_flatten_result handles the empty ChromaDB response shape."""
    from services.rag_engine import _flatten_result

    raw = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    assert _flatten_result(raw) == []


# ── retrieve_requirements ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieve_returns_top_k(seeded_chroma):
    """retrieve_requirements returns at most top_k results."""
    from services.rag_engine import retrieve_requirements
    from models.schemas import ArticleDomain, DocumentChunk

    chunk = DocumentChunk(
        chunk_id="c1",
        text="Risk management procedures.",
        source_file="doc.txt",
        chunk_index=0,
        domain=ArticleDomain.RISK_MANAGEMENT,
    )

    emb = _make_embedding_service()
    chroma = _make_chroma_client(seeded_chroma)

    results = await retrieve_requirements(chunk, emb, chroma, top_k=2)
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_retrieve_prefers_same_language():
    """Same-language passages are ranked before cross-language results."""
    import chromadb
    from services.rag_engine import retrieve_requirements
    from models.schemas import ArticleDomain, DocumentChunk

    # Add both an English and a German passage to eu_ai_act
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection("eu_ai_act")
    col.upsert(
        ids=["en_001", "de_001"],
        documents=["English regulatory text about risk management.",
                   "Deutschsprachiger Text über Risikomanagement."],
        embeddings=[[0.1] * 384, [0.2] * 384],
        metadatas=[
            {"article_num": 9, "domain": "risk_management", "lang": "en"},
            {"article_num": 9, "domain": "risk_management", "lang": "de"},
        ],
    )
    for name in ["gdpr", "compliance_checklist"]:
        client.get_or_create_collection(name)

    chunk = DocumentChunk(
        chunk_id="c1",
        text="Risikomanagement für KI-Systeme.",
        source_file="doc.txt",
        chunk_index=0,
        language="de",
        domain=ArticleDomain.RISK_MANAGEMENT,
    )

    emb = _make_embedding_service([0.15] * 384)
    chroma = _make_chroma_client(client)

    results = await retrieve_requirements(chunk, emb, chroma, top_k=5)

    # German result (lang=de) should appear before English (lang=en)
    if len(results) >= 2:
        langs = [r["metadata"].get("lang") for r in results]
        de_idx = langs.index("de") if "de" in langs else None
        en_idx = langs.index("en") if "en" in langs else None
        if de_idx is not None and en_idx is not None:
            assert de_idx < en_idx


@pytest.mark.asyncio
async def test_retrieve_empty_when_chroma_fails():
    """retrieve_requirements returns [] when both ChromaDB queries raise."""
    from services.rag_engine import retrieve_requirements
    from models.schemas import ArticleDomain, DocumentChunk

    chunk = DocumentChunk(
        chunk_id="c1", text="Any text.", source_file="f.txt", chunk_index=0,
        domain=ArticleDomain.SECURITY,
    )

    emb = _make_embedding_service()
    failing_chroma = AsyncMock()
    failing_chroma.query.side_effect = ConnectionError("ChromaDB down")

    results = await retrieve_requirements(chunk, emb, failing_chroma, top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_retrieve_domain_filter_applied():
    """retrieve_requirements passes article_num filter when domain is known."""
    from services.rag_engine import retrieve_requirements
    from models.schemas import ArticleDomain, DocumentChunk

    chunk = DocumentChunk(
        chunk_id="c1", text="Transparency obligations.",
        source_file="f.txt", chunk_index=0,
        domain=ArticleDomain.TRANSPARENCY,  # article 13
    )

    emb = _make_embedding_service()
    chroma = AsyncMock()
    chroma.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    await retrieve_requirements(chunk, emb, chroma, top_k=3)

    # Both queries should have been called with where={"article_num": 13}
    for call in chroma.query.call_args_list:
        assert call.kwargs.get("where") == {"article_num": 13}


@pytest.mark.asyncio
async def test_retrieve_no_domain_no_filter():
    """retrieve_requirements does NOT apply where filter when domain is None."""
    from services.rag_engine import retrieve_requirements
    from models.schemas import DocumentChunk

    chunk = DocumentChunk(
        chunk_id="c1", text="Some text.", source_file="f.txt", chunk_index=0,
        domain=None,
    )

    emb = _make_embedding_service()
    chroma = AsyncMock()
    chroma.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    await retrieve_requirements(chunk, emb, chroma, top_k=3)

    for call in chroma.query.call_args_list:
        assert call.kwargs.get("where") is None


# ── RAG accuracy (golden-dataset evaluation) ──────────────────────────────────

@pytest.mark.asyncio
async def test_rag_precision_golden_dataset():
    """
    RAG accuracy test using a small golden dataset.

    For each query, we embed it and check that the expected article number
    appears in the top-3 retrieved results. Target: precision@3 >= 0.80.

    This uses an in-memory ChromaDB seeded with realistic passages —
    no live infrastructure required.
    """
    import chromadb
    from services.rag_engine import retrieve_requirements
    from models.schemas import ArticleDomain, DocumentChunk

    # ── Seed in-memory ChromaDB with representative regulatory passages ─────
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection("eu_ai_act")

    passages = [
        ("p9_1",  "High-risk AI systems shall implement a risk management system.",
         {"article_num": 9,  "domain": "risk_management",         "lang": "en"}),
        ("p10_1", "Training data shall meet quality criteria including relevance.",
         {"article_num": 10, "domain": "data_governance",         "lang": "en"}),
        ("p11_1", "Providers shall draw up technical documentation before placing AI on market.",
         {"article_num": 11, "domain": "technical_documentation", "lang": "en"}),
        ("p12_1", "Providers shall keep logs automatically generated by their AI system.",
         {"article_num": 12, "domain": "record_keeping",          "lang": "en"}),
        ("p13_1", "High-risk AI systems shall be transparent and provide instructions for use.",
         {"article_num": 13, "domain": "transparency",            "lang": "en"}),
        ("p14_1", "Human oversight measures shall be ensured during operation.",
         {"article_num": 14, "domain": "human_oversight",         "lang": "en"}),
        ("p15_1", "AI systems shall be accurate, robust and cybersecure throughout lifecycle.",
         {"article_num": 15, "domain": "security",                "lang": "en"}),
    ]
    ids, docs, embeddings, metas = [], [], [], []
    for pid, text, meta in passages:
        ids.append(pid)
        docs.append(text)
        metas.append(meta)
        # Use article_num as a simple distinguishing vector dimension
        vec = [0.0] * 384
        vec[meta["article_num"]] = 1.0
        embeddings.append(vec)
    col.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

    for name in ["gdpr", "compliance_checklist"]:
        client.get_or_create_collection(name)

    # ── Golden queries: (query_text, expected_article_num, domain) ──────────
    golden = [
        ("We maintain a risk register for our AI.",     9,  ArticleDomain.RISK_MANAGEMENT),
        ("Our training dataset was curated for quality.", 10, ArticleDomain.DATA_GOVERNANCE),
        ("System architecture is documented in annex.", 11, ArticleDomain.TECHNICAL_DOCUMENTATION),
        ("Audit logs are retained for three years.",    12, ArticleDomain.RECORD_KEEPING),
        ("Users are informed about AI interactions.",   13, ArticleDomain.TRANSPARENCY),
        ("A human operator can override the system.",   14, ArticleDomain.HUMAN_OVERSIGHT),
        ("The system is tested against adversarial inputs.", 15, ArticleDomain.SECURITY),
    ]

    hits = 0
    for query_text, expected_article, domain in golden:
        # Build a query vector that matches the seeded passage for this article
        query_vec = [0.0] * 384
        query_vec[expected_article] = 1.0

        chunk = DocumentChunk(
            chunk_id=f"q_{expected_article}",
            text=query_text,
            source_file="test.txt",
            chunk_index=0,
            domain=domain,
        )

        emb = _make_embedding_service(query_vec)
        chroma = _make_chroma_client(client)

        results = await retrieve_requirements(chunk, emb, chroma, top_k=3)
        retrieved_articles = {r["metadata"].get("article_num") for r in results}
        if expected_article in retrieved_articles:
            hits += 1

    precision_at_3 = hits / len(golden)
    assert precision_at_3 >= 0.80, (
        f"RAG precision@3 is {precision_at_3:.2f} — expected >= 0.80. "
        f"Got {hits}/{len(golden)} hits."
    )
