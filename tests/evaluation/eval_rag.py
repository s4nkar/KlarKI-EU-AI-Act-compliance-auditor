"""
Evaluation 2 — RAG Retrieval (Recall@K, Precision@K, nDCG, negative controls,
cross-lingual breakdown).

For each query in gold_rag_queries.jsonl, embeds the query and retrieves
the top K passages from ChromaDB.  Checks whether the expected article
number appears in the returned metadata.

Metrics produced:
  Recall@1, Recall@3, Recall@5  — fraction of queries where the correct
  article appeared in the top K results.
  MRR (Mean Reciprocal Rank)    — average of 1/rank of first correct hit.
  Precision@3, Precision@5      — fraction of the top-K passages themselves
  that belong to an expected article (recall only asks "did the right
  article show up anywhere in top K"; precision asks "how much of top K is
  actually relevant" — a query that recalls correctly but pads the rest of
  top K with noise scores well on recall and poorly on precision).
  nDCG@5                        — ranking-quality metric (binary relevance)
  rewarding correct hits appearing earlier in the ranking, not just present.
  Language breakdown             — Recall@3 computed separately for EN vs DE
  queries (detected via langdetect), since the corpus and embeddings are
  explicitly bilingual (multilingual-e5-small; eu_ai_act collection has
  lang=en and lang=de passages per article — see build_knowledge_base.py).
  Negative-query rejection rate  — fraction of gold_rag_negative.jsonl
  queries (deliberately off-topic, e.g. "best pizza toppings") whose
  top-ranked passage scores below a relevance threshold on the same
  cross-encoder used for re-ranking. Retrieval always returns top_k
  candidates regardless of relevance (there's no built-in cutoff), so this
  checks the *score*, not whether results were returned.

Requires: ChromaDB running + eu_ai_act / gdpr collections seeded.
If ChromaDB is unreachable the test is skipped automatically.

Usage:
    python tests/evaluation/eval_rag.py
    python tests/evaluation/eval_rag.py --top-k 5 --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
GOLD_PATH   = Path(__file__).parent / "gold" / "gold_rag_queries.jsonl"
NEGATIVE_PATH = Path(__file__).parent / "gold" / "gold_rag_negative.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Cross-encoder score below which a top-ranked passage is judged "irrelevant"
# for negative-control queries. Empirically calibrated: real on-topic queries
# score ~+1.5 to +2.5 on ms-marco-MiniLM-L-6-v2; genuinely unrelated queries
# ("best pizza toppings") score ~-9 to -11. 0.0 leaves a wide safety margin.
_NEGATIVE_SCORE_THRESHOLD = 0.0

_API_DIR = next((p for p in [REPO_ROOT / "api", Path("/app")] if p.is_dir()), Path("/app"))
sys.path.insert(0, str(_API_DIR))


# ── helpers ────────────────────────────────────────────────────────────────

def load_queries() -> list[dict]:
    queries = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def load_negative_queries() -> list[dict]:
    if not NEGATIVE_PATH.exists():
        return []
    queries = []
    with open(NEGATIVE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def _ndcg_at_k(relevances: list[int], k: int) -> float:
    """Binary-relevance nDCG@k. relevances[i] = 1 if result i is relevant."""
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal[:k]))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


async def _run_async(top_k: int, verbose: bool) -> dict:
    try:
        from services.embedding_service import EmbeddingService
        from services.chroma_client import ChromaClient
        from services.rag_engine import retrieve_requirements, build_bm25_index, _get_cross_encoder
        from services.language_detector import detect_language
        from models.schemas import DocumentChunk
    except ImportError as e:
        return _skip(f"Cannot import API services: {e}. Run from repo root.")

    import os
    chroma_host = os.getenv("CHROMADB_HOST", "localhost")

    # ── connectivity check ─────────────────────────────────────────────────
    chroma = ChromaClient(host=chroma_host)
    try:
        alive = await chroma.health_check()
        if not alive:
            raise RuntimeError("health_check returned False")
    except Exception as e:
        return _skip(f"ChromaDB not reachable at {chroma_host}: {e}")

    # ── check collections exist ────────────────────────────────────────────
    collections = await chroma.list_collections()
    if "eu_ai_act" not in collections:
        return _skip(
            "eu_ai_act collection not found in ChromaDB. "
            "Run ./run.sh setup to seed regulatory data."
        )

    if verbose:
        print(f"  ChromaDB OK — collections: {collections}")

    # Build BM25 index from ChromaDB corpus so the eval tests the full
    # hybrid pipeline (BM25 + vector + RRF + cross-encoder), matching
    # exactly what production does at app startup.
    await build_bm25_index(chroma)
    if verbose:
        print("  BM25 index built")

    emb_service = EmbeddingService()
    queries     = load_queries()

    hits_at: dict[int, int] = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks: list[float] = []
    precision_at_3s: list[float] = []
    precision_at_5s: list[float] = []
    ndcg_at_5s: list[float] = []
    per_query_results: list[dict] = []

    # Per-language recall@3 breakdown — corpus + embeddings are bilingual
    # (multilingual-e5-small; eu_ai_act passages exist in both en and de).
    lang_hits: dict[str, int] = {}
    lang_totals: dict[str, int] = {}

    for idx, q in enumerate(queries):
        query_text        = q["query"]
        expected_articles = set(q["expected_articles"])

        # Wrap query as a DocumentChunk with domain=None so retrieve_requirements
        # applies no article filter — same unfiltered path used when domain is
        # unknown at the start of the pipeline.
        chunk = DocumentChunk(
            chunk_id=f"eval-query-{idx:03d}",
            text=query_text,
            source_file="eval_gold_queries",
            chunk_index=idx,
            language="en",
            domain=None,
        )

        # Use the production retrieval path (eu_ai_act + compliance_checklist,
        # language-preferred, sorted by distance).
        retrieved = await retrieve_requirements(
            chunk, emb_service, chroma, top_k=max(top_k, 5)
        )

        results = [
            {
                "article_num": r["metadata"].get("article_num"),
                "distance":    round(r["distance"], 4),
                "text":        r["text"][:120],
            }
            for r in retrieved
        ]

        # Compute Recall@K and rank of first hit
        first_hit_rank: int | None = None
        for k in (1, 3, 5):
            top_articles = {r["article_num"] for r in results[:k]}
            if expected_articles & top_articles:
                hits_at[k] += 1
                if k == 1 and first_hit_rank is None:
                    first_hit_rank = 1

        if first_hit_rank is None:
            for rank, r in enumerate(results[:top_k], start=1):
                if r["article_num"] in expected_articles:
                    first_hit_rank = rank
                    break

        reciprocal_ranks.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

        # Precision@K: fraction of the top-K passages themselves that are
        # relevant, not just whether the right article showed up anywhere.
        relevance_flags = [int(r["article_num"] in expected_articles) for r in results]
        precision_at_3 = sum(relevance_flags[:3]) / 3
        precision_at_5 = sum(relevance_flags[:5]) / 5
        precision_at_3s.append(precision_at_3)
        precision_at_5s.append(precision_at_5)
        ndcg_at_5s.append(_ndcg_at_k(relevance_flags, 5))

        # Language breakdown — detect the query's own language (not the
        # retrieved passage's) to see if DE queries recall as well as EN ones.
        query_lang = await detect_language(query_text)
        lang_totals[query_lang] = lang_totals.get(query_lang, 0) + 1
        hit_at_3 = int(bool(expected_articles & {r["article_num"] for r in results[:3]}))
        if hit_at_3:
            lang_hits[query_lang] = lang_hits.get(query_lang, 0) + 1

        per_query_results.append({
            "query":             query_text,
            "expected_articles": list(expected_articles),
            "language":          query_lang,
            "precision@3":       round(precision_at_3, 4),
            "precision@5":       round(precision_at_5, 4),
            "ndcg@5":            ndcg_at_5s[-1],
            "hit@1":             int(bool(expected_articles & {r["article_num"] for r in results[:1]})),
            "hit@3":             hit_at_3,
            "hit@5":             int(bool(expected_articles & {r["article_num"] for r in results[:5]})),
            "first_hit_rank":    first_hit_rank,
            "top3_articles":     [r["article_num"] for r in results[:3]],
        })

        if verbose:
            hit_icon = "✓" if per_query_results[-1]["hit@3"] else "✗"
            print(f"  {hit_icon} [{per_query_results[-1]['top3_articles']}] ← {query_text[:60]}")

    n = len(queries)
    recall_at_1 = round(hits_at[1] / n, 4)
    recall_at_3 = round(hits_at[3] / n, 4)
    recall_at_5 = round(hits_at[5] / n, 4)
    mrr         = round(sum(reciprocal_ranks) / n, 4)
    precision_at_3 = round(sum(precision_at_3s) / n, 4)
    precision_at_5 = round(sum(precision_at_5s) / n, 4)
    ndcg_at_5      = round(sum(ndcg_at_5s) / n, 4)

    by_language = {
        lang: {
            "n_queries": lang_totals[lang],
            "recall@3":  round(lang_hits.get(lang, 0) / lang_totals[lang], 4),
        }
        for lang in lang_totals
    }

    # ── Negative-control queries: off-topic queries should score low ────────
    negative_queries = load_negative_queries()
    negative_results: list[dict] = []
    if negative_queries:
        encoder = _get_cross_encoder()
        for nq in negative_queries:
            nq_text = nq["query"]
            nq_chunk = DocumentChunk(
                chunk_id="eval-negative", text=nq_text, source_file="eval_negative_queries",
                chunk_index=0, language="en", domain=None,
            )
            nq_retrieved = await retrieve_requirements(nq_chunk, emb_service, chroma, top_k=3)
            if not nq_retrieved or encoder is None:
                negative_results.append({
                    "query": nq_text, "top_score": None, "correctly_rejected": nq_retrieved == [],
                })
                continue
            pairs = [(nq_text, r["text"]) for r in nq_retrieved]
            scores = encoder.predict(pairs)
            top_score = float(max(scores))
            negative_results.append({
                "query":              nq_text,
                "top_score":          round(top_score, 3),
                "correctly_rejected": top_score < _NEGATIVE_SCORE_THRESHOLD,
            })
            if verbose:
                icon = "✓" if negative_results[-1]["correctly_rejected"] else "✗"
                print(f"  {icon} [neg] score={top_score:.2f} ← {nq_text[:60]}")

    negative_rejection_rate = (
        round(sum(1 for r in negative_results if r["correctly_rejected"]) / len(negative_results), 4)
        if negative_results else None
    )

    status = "pass" if recall_at_3 >= 0.80 else "warn"
    if negative_rejection_rate is not None and negative_rejection_rate < 0.75:
        status = "warn" if status == "pass" else status

    results_dict = {
        "eval":                    "rag_retrieval",
        "status":                  status,
        "recall@1":                recall_at_1,
        "recall@3":                recall_at_3,
        "recall@5":                recall_at_5,
        "mrr":                     mrr,
        "precision@3":             precision_at_3,
        "precision@5":             precision_at_5,
        "ndcg@5":                  ndcg_at_5,
        "by_language":             by_language,
        "negative_rejection_rate": negative_rejection_rate,
        "negative_results":        negative_results,
        "n_queries":               n,
        "n_negative_queries":      len(negative_queries),
        "top_k":                   top_k,
        "per_query":               per_query_results,
        "threshold_recall@3":      0.80,
        "threshold_negative_rejection_rate": 0.75,
    }

    out_path = RESULTS_DIR / "rag.json"
    out_path.write_text(json.dumps(results_dict, indent=2), encoding="utf-8")
    return results_dict


def run(top_k: int = 5, verbose: bool = False) -> dict:
    return asyncio.run(_run_async(top_k=top_k, verbose=verbose))


def _skip(reason: str) -> dict:
    return {"eval": "rag_retrieval", "status": "skip", "reason": reason}


def print_report(r: dict) -> None:
    if r["status"] == "skip":
        print(f"  [SKIP] {r['reason']}")
        return

    status_icon = "✓" if r["status"] == "pass" else "✗"
    print(f"\n  {'─'*52}")
    print(f"  {status_icon} RAG Retrieval — Recall@K Evaluation")
    print(f"  {'─'*52}")
    print(f"  Recall@1     : {r['recall@1']*100:.1f}%")
    print(f"  Recall@3     : {r['recall@3']*100:.1f}%  (threshold ≥ 80%)")
    print(f"  Recall@5     : {r['recall@5']*100:.1f}%")
    print(f"  Precision@3  : {r.get('precision@3', 0)*100:.1f}%")
    print(f"  Precision@5  : {r.get('precision@5', 0)*100:.1f}%")
    print(f"  nDCG@5       : {r.get('ndcg@5', 0):.3f}")
    print(f"  MRR          : {r['mrr']:.3f}")
    print(f"  Queries      : {r['n_queries']}")

    by_lang = r.get("by_language", {})
    if by_lang:
        print("\n  Recall@3 by language:")
        for lang, info in by_lang.items():
            print(f"    {lang}: {info['recall@3']*100:.1f}%  ({info['n_queries']} queries)")

    if r.get("negative_rejection_rate") is not None:
        print(f"\n  Negative-query rejection rate: {r['negative_rejection_rate']*100:.1f}%  "
              f"(threshold ≥ 75%, {r.get('n_negative_queries', 0)} queries)")
        for nr in r.get("negative_results", []):
            if not nr["correctly_rejected"]:
                print(f"    ✗ FALSE POSITIVE: score={nr['top_score']} ← {nr['query'][:60]}")

    misses = [q for q in r.get("per_query", []) if not q["hit@3"]]
    if misses:
        print(f"\n  Recall@3 misses ({len(misses)}):")
        for m in misses[:5]:
            print(f"    • {m['query'][:60]}")
            print(f"      expected={m['expected_articles']}  got={m['top3_articles']}")


# ── pytest ─────────────────────────────────────────────────────────────────

def test_rag_recall_at_3() -> None:
    """pytest: RAG must achieve Recall@3 ≥ 80% on gold queries."""
    r = run(top_k=5)
    if r["status"] == "skip":
        import pytest
        pytest.skip(r.get("reason", ""))
    assert r["recall@3"] >= 0.80, (
        f"Recall@3 = {r['recall@3']:.3f} is below 0.80. "
        "Check ChromaDB seeding and embedding model."
    )


def test_rag_mrr() -> None:
    """pytest: MRR must be ≥ 0.65."""
    r = run(top_k=5)
    if r["status"] == "skip":
        import pytest
        pytest.skip(r.get("reason", ""))
    assert r["mrr"] >= 0.65, f"MRR {r['mrr']:.3f} is below 0.65"


def test_rag_negative_query_rejection() -> None:
    """pytest: off-topic queries must not be retrieved as confidently relevant."""
    r = run(top_k=5)
    if r["status"] == "skip":
        import pytest
        pytest.skip(r.get("reason", ""))
    rate = r.get("negative_rejection_rate")
    if rate is None:
        import pytest
        pytest.skip("No negative queries / cross-encoder unavailable")
    false_positives = [nr["query"] for nr in r.get("negative_results", []) if not nr["correctly_rejected"]]
    assert rate >= 0.75, (
        f"Negative-query rejection rate {rate:.2%} below 75%. "
        f"False positives: {false_positives}"
    )


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Recall@K evaluation")
    parser.add_argument("--top-k",   type=int,  default=5)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"Running RAG retrieval evaluation (top_k={args.top_k}) …")
    results = run(top_k=args.top_k, verbose=args.verbose)
    print_report(results)
    sys.exit(0 if results.get("status") != "fail" else 1)
