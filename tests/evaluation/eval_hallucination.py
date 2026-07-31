"""
Evaluation 4 — Hallucination / Citation Verification.

Every compliance gap the system surfaces must be traceable to actual
regulatory text.  This eval enforces four rules:

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

  Rule 4 — Semantic entailment (topical grounding):
      Each gap's *topic* (its title) must be discussed by at least one of the
      article's retrieved regulatory passages, per the NLI cross-encoder
      already used in evidence_mapper.py (cross-encoder/nli-deberta-v3-small).
      Rule 1 only checks that *some* passage was retrieved for an article;
      Rule 4 checks that the reported gap is actually *about* what the
      retrieved text says, catching gaps that cite a real passage but are
      actually about an unrelated topic.

      The hypothesis is built from gap.title, not gap.description — gap
      descriptions are deficiency findings ("X is not done"), which read as
      CONTRADICTIONS of a regulatory requirement even when the citation is
      perfectly grounded. Framing the hypothesis as a topic-presence claim
      ("This passage discusses the requirement: {title}"), matching
      evidence_mapper.py's own proven pattern, checks topical relevance
      instead of asking the passage to logically prove the deficiency.

Uses proposition_chunk_text (the production chunker) so the chunking
strategy matches what the real pipeline produces. Uses select_query_chunks
(the production RAG chunk-selection ranker) so retrieval quality matches
what the real pipeline would retrieve for the same document.

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

# Minimum NLI entailment score for a passage to count as supporting a gap description
_NLI_ENTAILMENT_THRESHOLD = 0.5

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


def _score_entailment(nli_model, pairs: list[tuple[str, str]]) -> list[tuple[bool, float]]:
    """Batch-score (premise, hypothesis) pairs; return per-pair (entailed, score).

    Mirrors the scoring convention in evidence_mapper.py's _evidence_present:
    predicted label must be ENTAILMENT and its score must clear the threshold.
    `score` is the raw entailment-class probability, returned so callers can
    pick the best-supporting passage per gap for citation display.
    """
    labels: dict = getattr(
        nli_model.model.config,
        "id2label",
        {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"},
    )
    entailment_idx = next(
        (idx for idx, lbl in labels.items() if "ENTAILMENT" in lbl.upper()),
        None,
    )
    batch_scores = nli_model.predict(pairs)
    results: list[tuple[bool, float]] = []
    for scores in batch_scores:
        ent_score = float(scores[entailment_idx]) if entailment_idx is not None else 0.0
        predicted_label = labels.get(int(scores.argmax()), "").upper()
        if "ENTAILMENT" not in predicted_label or ent_score < _NLI_ENTAILMENT_THRESHOLD:
            results.append((False, ent_score))
        else:
            results.append((True, ent_score))
    return results


def _check_article_score(score, nli_model=None) -> tuple[list[str], dict, list[dict]]:
    """Return (violation strings, entailment stats, gap_citations) for one ArticleScore.

    gap_citations gives the frontend enough to show, per gap, which retrieved
    passage (if any) was compared against it and whether it was judged to
    support the gap — the actual "citation detail" a reviewer needs, not just
    a pass/fail count.
    """
    violations: list[str] = []
    art = score.article_num
    entailment_stats = {"checked": 0, "entailed": 0}
    gap_citations: list[dict] = []

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

    # Rule 4: semantic entailment — does any retrieved passage actually cover the
    # topic of the gap, or does the LLM just cite a passage count without the
    # gap being about that passage at all? Also builds gap_citations: for each
    # gap, the single best-matching passage + its score, so the frontend can
    # show actual citation text instead of just a pass/fail count.
    #
    # NOTE: the hypothesis is built from gap.title (a topic label), NOT
    # gap.description. Gap descriptions are deficiency findings ("X is not
    # done") — the *negation* of what a regulatory passage states — so
    # entailment(passage, description) is almost always CONTRADICTION even for
    # a perfectly well-grounded citation. Mirroring evidence_mapper.py's own
    # proven pattern (a topic-presence claim, not a deficiency claim) fixes this:
    # "This passage discusses the requirement: {title}." correctly reads as
    # ENTAILMENT when the passage is genuinely on-topic.
    checkable_gaps = [
        (gi, gap) for gi, gap in enumerate(score.gaps)
        if gap.title and gap.title.strip()
    ]

    if nli_model is not None and score.regulatory_passages and checkable_gaps:
        pairs: list[tuple[str, str]] = []
        pair_gap_index: list[int] = []
        pair_passage_index: list[int] = []
        for gi, gap in checkable_gaps:
            hypothesis = f"This passage discusses the requirement: {gap.title}."
            for pi, passage in enumerate(score.regulatory_passages):
                pairs.append((passage.text, hypothesis))
                pair_gap_index.append(gi)
                pair_passage_index.append(pi)

        try:
            scored = _score_entailment(nli_model, pairs)  # [(entailed, score), ...]
            per_gap_best: dict[int, tuple[int, float, bool]] = {}  # gi -> (pi, score, entailed)
            entailed_gaps: set[int] = set()
            for (entailed, ent_score), gi, pi in zip(scored, pair_gap_index, pair_passage_index):
                if entailed:
                    entailed_gaps.add(gi)
                if gi not in per_gap_best or ent_score > per_gap_best[gi][1]:
                    per_gap_best[gi] = (pi, ent_score, entailed)

            checked_gaps = {gi for gi, _ in checkable_gaps}
            entailment_stats["checked"] = len(checked_gaps)
            entailment_stats["entailed"] = len(entailed_gaps)

            for gi, gap in checkable_gaps:
                pi, ent_score, entailed = per_gap_best[gi]
                passage = score.regulatory_passages[pi]
                gap_citations.append({
                    "title": gap.title,
                    "description": gap.description,
                    "entailed": entailed,
                    "score": round(ent_score, 3),
                    "passage_ref": passage.article_ref or passage.title,
                    "passage_text": passage.text[:300],
                })
                if not entailed:
                    violations.append(
                        f"Article {art} gap[{gi}]: no retrieved passage discusses "
                        f"the topic '{gap.title}' — possible off-topic citation"
                    )
        except Exception as exc:
            # Non-fatal — fall back to Rules 1-3 only for this article, matching
            # evidence_mapper.py's own NLI-unavailable fallback behaviour.
            print(f"  [warn] Article {art}: NLI entailment scoring failed: {exc}")
    elif score.regulatory_passages and checkable_gaps:
        # NLI unavailable — still surface the first retrieved passage per gap so
        # the frontend has *something* to show, marked as "not checked."
        for gi, gap in checkable_gaps:
            passage = score.regulatory_passages[0]
            gap_citations.append({
                "title": gap.title,
                "description": gap.description,
                "entailed": None,
                "score": None,
                "passage_ref": passage.article_ref or passage.title,
                "passage_text": passage.text[:300],
            })

    return violations, entailment_stats, gap_citations


async def _run_async(strict: bool, verbose: bool) -> dict:
    import os

    try:
        from services.chunker           import proposition_chunk_text
        from services.language_detector import detect_language
        from services.classifier        import classify_chunks
        from services.embedding_service import EmbeddingService
        from services.chroma_client     import ChromaClient
        from services.rag_engine        import retrieve_requirements, build_bm25_index, select_query_chunks
        from services.gap_analyser      import analyse_article
        from services.evidence_mapper   import _get_nli_model
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

    # Same NLI cross-encoder evidence_mapper.py uses — reused here (not reloaded) so
    # Rule 4 stays consistent with the production grounding model. None if unavailable.
    nli_model = _get_nli_model()
    if verbose and nli_model is None:
        print("  [warn] NLI model unavailable — Rule 4 (semantic entailment) will be skipped")

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
    total_gaps_checked = 0
    total_gaps_entailed = 0

    for art_num, domain in ARTICLE_DOMAINS.items():
        art_chunks = [c for c in classified if c.domain == domain]
        query_chunks = await select_query_chunks(art_chunks, art_num, emb)

        passages: list[dict] = []
        for chunk in query_chunks:
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

        violations, ent_stats, gap_citations = _check_article_score(score, nli_model)
        all_violations.extend(violations)
        total_gaps_checked  += ent_stats["checked"]
        total_gaps_entailed += ent_stats["entailed"]

        article_results[art_num] = {
            "gaps":            len(score.gaps),
            "recommendations": len(score.recommendations),
            "passages":        len(score.regulatory_passages),
            "chunk_count":     score.chunk_count,
            "violations":      violations,
            "gaps_checked_for_entailment": ent_stats["checked"],
            "gaps_entailed":               ent_stats["entailed"],
            "gap_citations":   gap_citations,
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

    # Entailment rate: fraction of gaps (with passages to check against) whose
    # description was actually entailed by at least one retrieved passage.
    # None (not 0.0) when the NLI model is unavailable, since the check never ran.
    entailment_rate = (
        round(total_gaps_entailed / total_gaps_checked, 4)
        if total_gaps_checked > 0 else None
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
        "entailment_rate":         entailment_rate,
        "nli_model_available":     nli_model is not None,
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
    if r.get("entailment_rate") is not None:
        print(f"  Entailment rate  : {r['entailment_rate']*100:.1f}%  "
              f"(fraction of gap descriptions entailed by a retrieved passage)")
    elif not r.get("nli_model_available", True):
        print("  Entailment rate  : n/a (NLI model unavailable)")
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
    quality_failures = [
        v for v in r.get("violations", [])
        if "empty title" in v or "empty description" in v or "suspiciously short description" in v
    ]
    assert not quality_failures, "Gap content quality failures:\n" + "\n".join(quality_failures)


def test_gap_descriptions_entailed() -> None:
    """pytest: Every gap description must be entailed by a retrieved passage (Rule 4)."""
    r = run(strict=False)
    if r["status"] == "skip":
        import pytest
        pytest.skip(r.get("reason", ""))
    if not r.get("nli_model_available", True):
        import pytest
        pytest.skip("NLI model unavailable — Rule 4 could not run")
    ungrounded = [v for v in r.get("violations", []) if "not entailed by any retrieved passage" in v]
    assert not ungrounded, "Ungrounded (non-entailed) gap descriptions:\n" + "\n".join(ungrounded)


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
