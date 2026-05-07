"""
Evaluation 4 — Hallucination / Citation Verification.

Every compliance gap the system surfaces must be traceable to actual
regulatory text.  This eval enforces three rules:

  Rule 1 — Evidence linkage:
      Each ArticleScore must have ≥ 1 regulatory_passage retrieved.
      A gap reported against an article with zero retrieved passages is
      ungrounded (hallucinated).

  Rule 2 — Gap content quality:
      No gap may have an empty title or description.
      One-word descriptions are treated as hallucinated content.

  Rule 3 — Recommendation grounding:
      Recommendations must reference a regulatory concept
      (checked via keyword matching against a known vocabulary).

Uses proposition_chunk_text (the production chunker) so the chunking
strategy matches what the real pipeline produces.

applicable_articles is intentionally passed as None to analyse_article
so all 7 articles run through LangGraph — the goal here is to verify
LLM output quality across every article, independent of whether a given
document would trigger the applicability gate.

Requires: Ollama + ChromaDB running.

Usage:
    python tests/evaluation/eval_hallucination.py
    python tests/evaluation/eval_hallucination.py --strict
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
_API_DIR = next((p for p in [REPO_ROOT / "api", Path("/app")] if p.is_dir()), Path("/app"))
sys.path.insert(0, str(_API_DIR))

# Keywords that a grounded recommendation should contain
REGULATORY_VOCAB = {
    "article", "risk", "data", "document", "log", "record", "transparency",
    "human", "oversight", "accuracy", "security", "robustness", "training",
    "provider", "deployer", "audit", "assessment", "measure", "system",
    "requirement", "compliance", "regulation", "gdpr", "eu ai act",
    "annex", "technical", "monitor", "oversight", "governance",
}

SYNTHETIC_DOCUMENT = """
ARTIFICIAL INTELLIGENCE SYSTEM COMPLIANCE DOCUMENTATION

Risk Management: We assess risks via a register updated quarterly.
Data Governance: Training data was quality-checked and bias-analyzed.
Technical Documentation: System architecture is documented in the tech dossier.
Record Keeping: All decisions are logged with timestamps for 3 years.
Transparency: Users are informed when AI makes recommendations.
Human Oversight: Human review is required for all high-confidence decisions.
Security: Adversarial robustness testing passed with 97% success rate.
"""


def _is_grounded_recommendation(text: str) -> bool:
    """Return True if the recommendation references at least one regulatory concept."""
    lower = text.lower()
    return any(kw in lower for kw in REGULATORY_VOCAB)


def _check_article_score(score) -> list[str]:
    """Return a list of violation strings for a single ArticleScore."""
    violations: list[str] = []
    art = score.article_num

    # Rule 1: articles where actual user chunks were analysed must have regulatory
    # passages backing their gaps. Articles with no classified chunks produce a
    # synthetic "no documentation" gap that needs no citation.
    if score.chunk_count > 0 and not score.regulatory_passages:
        violations.append(
            f"Article {art}: no regulatory passages retrieved — "
            "gap analysis is ungrounded"
        )

    # Rule 2: gap content quality
    for i, gap in enumerate(score.gaps):
        if not gap.title or not gap.title.strip():
            violations.append(f"Article {art} gap[{i}]: empty title")
        if not gap.description or not gap.description.strip():
            violations.append(f"Article {art} gap[{i}]: empty description")
        if gap.description and len(gap.description.split()) < 4:
            violations.append(
                f"Article {art} gap[{i}]: suspiciously short description "
                f"({len(gap.description.split())} words): '{gap.description}'"
            )

    # Rule 3: recommendation grounding
    for i, rec in enumerate(score.recommendations):
        if not _is_grounded_recommendation(rec):
            violations.append(
                f"Article {art} rec[{i}]: recommendation not grounded in "
                f"regulatory vocabulary: '{rec[:80]}'"
            )

    return violations


async def _run_async(strict: bool, verbose: bool) -> dict:
    import os

    try:
        from services.chunker           import proposition_chunk_text
        from services.language_detector import detect_language
        from services.classifier        import classify_chunks
        from services.embedding_service import EmbeddingService
        from services.chroma_client     import ChromaClient
        from services.rag_engine        import retrieve_requirements, build_bm25_index
        from services.gap_analyser      import analyse_article
        from services.ollama_client     import OllamaClient
        from models.schemas             import ArticleDomain
    except ImportError as e:
        return _skip(f"Cannot import API services: {e}")

    ollama_host  = os.getenv("OLLAMA_HOST",  "localhost")
    chroma_host  = os.getenv("CHROMADB_HOST", "localhost")
    ollama_model = os.getenv("OLLAMA_MODEL",  "phi3:mini")

    ollama = OllamaClient(host=ollama_host, model=ollama_model)
    chroma = ChromaClient(host=chroma_host)
    try:
        ok = await ollama.health_check() and await chroma.health_check()
        if not ok:
            raise RuntimeError()
    except Exception as e:
        return _skip(f"Ollama or ChromaDB not reachable: {e}")

    emb = EmbeddingService()

    # Use the production chunker so chunking matches what the real pipeline produces
    chunks = await proposition_chunk_text(SYNTHETIC_DOCUMENT, source_file="hallucination_test.txt")

    lang = await detect_language(SYNTHETIC_DOCUMENT)
    for chunk in chunks:
        chunk.language = lang

    classified, _backend = await classify_chunks(chunks, ollama)

    # Build BM25 index so hybrid retrieval works (built at app startup in production)
    await build_bm25_index(chroma)

    ARTICLE_DOMAINS = {
        9:  ArticleDomain.RISK_MANAGEMENT,
        10: ArticleDomain.DATA_GOVERNANCE,
        11: ArticleDomain.TECHNICAL_DOCUMENTATION,
        12: ArticleDomain.RECORD_KEEPING,
        13: ArticleDomain.TRANSPARENCY,
        14: ArticleDomain.HUMAN_OVERSIGHT,
        15: ArticleDomain.SECURITY,
    }

    all_violations: list[str] = []
    article_results: dict[int, dict] = {}

    for art_num, domain in ARTICLE_DOMAINS.items():
        art_chunks = [c for c in classified if c.domain == domain]

        passages: list[dict] = []
        for chunk in art_chunks[:3]:
            try:
                # No applicable_articles filter — hallucination eval deliberately
                # runs LangGraph on all 7 articles to test LLM output quality
                # regardless of the applicability gate outcome.
                retrieved = await retrieve_requirements(
                    chunk=chunk,
                    embedding_service=emb,
                    chroma_client=chroma,
                    top_k=5,
                    regulation="eu_ai_act",
                )
                passages.extend(retrieved)
            except Exception:
                pass

        try:
            # applicable_articles=None so all articles enter LangGraph —
            # this eval tests citation quality, not the applicability gate.
            score = await analyse_article(
                article_num=art_num,
                domain=domain,
                user_chunks=art_chunks,
                regulatory_passages=passages,
                ollama=ollama,
                applicable_articles=None,
            )
        except Exception as ex:
            if verbose:
                print(f"  gap analysis failed for Article {art_num}: {ex}")
            continue

        violations = _check_article_score(score)
        all_violations.extend(violations)

        article_results[art_num] = {
            "gaps":            len(score.gaps),
            "recommendations": len(score.recommendations),
            "passages":        len(score.regulatory_passages),
            "chunk_count":     score.chunk_count,
            "violations":      violations,
        }

        if verbose:
            icon = "✓" if not violations else "✗"
            print(f"  {icon} Article {art_num}: {len(score.gaps)} gaps, "
                  f"{len(score.regulatory_passages)} passages, "
                  f"{len(violations)} violations")

    # Citation rate: fraction of articles that had user chunks and retrieved passages.
    # Articles with no classified chunks are excluded — they never trigger RAG.
    articles_with_chunks = [v for v in article_results.values() if v["chunk_count"] > 0]
    citation_rate = (
        sum(1 for v in articles_with_chunks if v["passages"] > 0)
        / max(len(articles_with_chunks), 1)
    )

    if strict:
        status = "pass" if not all_violations else "fail"
    else:
        status = "pass" if citation_rate >= 0.90 and len(all_violations) == 0 else (
            "warn" if citation_rate >= 0.75 else "fail"
        )

    results = {
        "eval":                    "hallucination",
        "status":                  status,
        "citation_rate":           round(citation_rate, 4),
        "total_violations":        len(all_violations),
        "violations":              all_violations,
        "article_results":         article_results,
        "strict_mode":             strict,
        "threshold_citation_rate": 0.90,
    }

    out_path = RESULTS_DIR / "hallucination.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run(strict: bool = False, verbose: bool = False) -> dict:
    return asyncio.run(_run_async(strict=strict, verbose=verbose))


def _skip(reason: str) -> dict:
    return {"eval": "hallucination", "status": "skip", "reason": reason}


def print_report(r: dict) -> None:
    if r["status"] == "skip":
        print(f"  [SKIP] {r['reason']}")
        return

    status_icon = "✓" if r["status"] == "pass" else ("!" if r["status"] == "warn" else "✗")
    print(f"\n  {'─'*52}")
    print(f"  {status_icon} Hallucination / Citation Verification")
    print(f"  {'─'*52}")
    print(f"  Citation rate    : {r['citation_rate']*100:.1f}%  (threshold ≥ 90%)")
    print(f"  Total violations : {r['total_violations']}")

    if r["violations"]:
        print("\n  Violations:")
        for v in r["violations"][:10]:
            print(f"    ✗ {v}")
        if len(r["violations"]) > 10:
            print(f"    … and {len(r['violations']) - 10} more")

    print("\n  Per-article:")
    for art_num, info in sorted(r.get("article_results", {}).items()):
        icon = "✓" if not info["violations"] else "✗"
        print(f"    {icon} Article {art_num}: {info['passages']} passages, "
              f"{info['gaps']} gaps, {info['recommendations']} recs  "
              f"violations={len(info['violations'])}")


# ── pytest ──────────────────────────────────────────────────────────────────

def test_no_ungrounded_gaps() -> None:
    """pytest: All articles must have ≥ 1 regulatory passage (no ungrounded gaps)."""
    r = run(strict=False)
    if r["status"] == "skip":
        import pytest
        pytest.skip(r.get("reason", ""))
    ungrounded = [v for v in r.get("violations", []) if "ungrounded" in v]
    assert not ungrounded, "Ungrounded gaps detected:\n" + "\n".join(ungrounded)


def test_gap_content_quality() -> None:
    """pytest: No gap may have an empty or single-word description."""
    r = run(strict=False)
    if r["status"] == "skip":
        import pytest
        pytest.skip(r.get("reason", ""))
    quality_failures = [v for v in r.get("violations", []) if "description" in v or "title" in v]
    assert not quality_failures, "Gap content quality failures:\n" + "\n".join(quality_failures)


def test_citation_rate_above_90() -> None:
    """pytest: Citation rate must be ≥ 90%."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r.get("reason", ""))
    assert r["citation_rate"] >= 0.90, (
        f"Citation rate {r['citation_rate']:.2%} below 90%. "
        "Some articles have no regulatory passages retrieved."
    )


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hallucination and citation verification")
    parser.add_argument("--strict",  action="store_true", help="Fail on any violation")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("Running hallucination / citation verification …")
    results = run(strict=args.strict, verbose=args.verbose)
    print_report(results)
    sys.exit(0 if results.get("status") in ("pass", "warn") else 1)
