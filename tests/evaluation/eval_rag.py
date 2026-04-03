"""
Evaluation 2 — RAG Retrieval (Recall@K).

For each query in gold_rag_queries.jsonl, embeds the query and retrieves
the top K passages from ChromaDB.  Checks whether the expected article
number appears in the returned metadata.

Metrics produced:
  Recall@1, Recall@3, Recall@5  — fraction of queries where the correct
  article appeared in the top K results.
  MRR (Mean Reciprocal Rank)    — average of 1/rank of first correct hit.

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
import sys
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
GOLD_PATH   = Path(__file__).parent / "datasets" / "gold_rag_queries.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

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


async def _run_async(top_k: int, verbose: bool) -> dict:
    try:
        from services.embedding_service import EmbeddingService
        from services.chroma_client import ChromaClient
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

    emb_service = EmbeddingService()
    queries     = load_queries()

    hits_at: dict[int, int] = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks: list[float] = []
    per_query_results: list[dict] = []

    for q in queries:
        query_text        = q["query"]
        expected_articles = set(q["expected_articles"])

        # Embed query
        vectors = await emb_service.embed([query_text])

        # Query ChromaDB — search both collections
        results: list[dict] = []
        for col in ("eu_ai_act", "gdpr"):
            if col not in collections:
                continue
            raw = await chroma.query(
                collection_name=col,
                query_embeddings=vectors,
                n_results=min(top_k, 10),
            )
            ids       = raw.get("ids",       [[]])[0]
            docs      = raw.get("documents", [[]])[0]
            metas     = raw.get("metadatas", [[]])[0]
            distances = raw.get("distances", [[]])[0]
            for i, (doc_id, doc, meta, dist) in enumerate(
                zip(ids, docs, metas, distances)
            ):
                results.append({
                    "rank":        i + 1,
                    "id":          doc_id,
                    "text":        doc[:120],
                    "article_num": meta.get("article_num"),
                    "distance":    round(dist, 4),
                    "collection":  col,
                })

        # Sort merged results by distance (lower = more similar)
        results.sort(key=lambda x: x["distance"])

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

        per_query_results.append({
            "query":             query_text,
            "expected_articles": list(expected_articles),
            "hit@1":             int(bool(expected_articles & {r["article_num"] for r in results[:1]})),
            "hit@3":             int(bool(expected_articles & {r["article_num"] for r in results[:3]})),
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

    results_dict = {
        "eval":         "rag_retrieval",
        "status":       "pass" if recall_at_3 >= 0.80 else "warn",
        "recall@1":     recall_at_1,
        "recall@3":     recall_at_3,
        "recall@5":     recall_at_5,
        "mrr":          mrr,
        "n_queries":    n,
        "top_k":        top_k,
        "per_query":    per_query_results,
        "threshold_recall@3": 0.80,
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
    print(f"  Recall@1  : {r['recall@1']*100:.1f}%")
    print(f"  Recall@3  : {r['recall@3']*100:.1f}%  (threshold ≥ 80%)")
    print(f"  Recall@5  : {r['recall@5']*100:.1f}%")
    print(f"  MRR       : {r['mrr']:.3f}")
    print(f"  Queries   : {r['n_queries']}")

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
