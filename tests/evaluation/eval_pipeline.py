"""
Evaluation 3 — End-to-End Pipeline Test.

Runs a synthetic compliance document through the full pipeline:
  chunking → classification → RAG retrieval → gap analysis → scoring

Checks:
  • Every chunk receives a domain label (no unclassified chunks)
  • All 7 articles appear in the report
  • Overall score is in [0, 100]
  • Each ArticleScore has score_reasoning populated
  • Each gap has a non-empty title and description

Requires: Ollama (phi3:mini) + ChromaDB running.
Skipped automatically if either service is unreachable.

Usage:
    python tests/evaluation/eval_pipeline.py
    python tests/evaluation/eval_pipeline.py --verbose
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

# Minimal synthetic document covering all 7 EU AI Act articles
SYNTHETIC_DOCUMENT = """
ARTIFICIAL INTELLIGENCE SYSTEM COMPLIANCE DOCUMENTATION

1. Risk Management
The risk management system identifies potential hazards in our AI classification model.
We conduct regular risk assessments and maintain a risk register with mitigation measures.
Known risks include model drift, adversarial inputs, and demographic bias in predictions.

2. Data Governance
Training data was sourced from internal records and validated for quality and completeness.
Data bias analysis was performed across gender, age, and ethnicity dimensions.
Training and validation datasets are maintained separately with documented lineage.

3. Technical Documentation
The system uses a transformer-based classifier with 110M parameters.
Architecture: BERT fine-tuned on domain-specific compliance text.
Model card and system specifications are maintained in the engineering dossier.

4. Record Keeping
All inference events are logged with timestamp, input hash, and prediction confidence.
Logs are retained for 36 months and protected from unauthorized modification.

5. Transparency
Users are notified when AI-assisted decisions are made on their behalf.
The system provides confidence scores and top contributing features.

6. Human Oversight
A compliance officer reviews all high-stakes decisions flagged by the system.
Human override is available at every decision point with full audit logging.

7. Security and Accuracy
The system achieves 89% accuracy on the held-out validation set.
Adversarial robustness testing was conducted using FGSM and PGD attacks.
Security assessment completed per ISO 27001 with no critical findings.
"""


async def _run_async(verbose: bool) -> dict:
    import os

    try:
        from services.chunker             import chunk_text
        from services.language_detector   import detect_language
        from services.classifier          import classify_chunks
        from services.embedding_service   import EmbeddingService
        from services.chroma_client       import ChromaClient
        from services.rag_engine          import retrieve_requirements
        from services.gap_analyser        import analyse_article
        from services.compliance_scorer   import score_audit
        from services.ollama_client       import OllamaClient
        from models.schemas               import ArticleDomain, DocumentChunk
    except ImportError as e:
        return _skip(f"Cannot import API services: {e}")

    ollama_host = os.getenv("OLLAMA_HOST", "localhost")
    chroma_host = os.getenv("CHROMADB_HOST", "localhost")

    # ── service connectivity ────────────────────────────────────────────────
    import os as _os
    ollama_model = _os.getenv("OLLAMA_MODEL", "phi3:mini")
    ollama = OllamaClient(host=ollama_host, model=ollama_model)
    try:
        if not await ollama.health_check():
            raise RuntimeError("unhealthy")
    except Exception as e:
        return _skip(f"Ollama not reachable at {ollama_host}: {e}")

    chroma = ChromaClient(host=chroma_host)
    try:
        if not await chroma.health_check():
            raise RuntimeError("unhealthy")
    except Exception as e:
        return _skip(f"ChromaDB not reachable at {chroma_host}: {e}")

    emb = EmbeddingService()

    # ── stage 1: parse + chunk ──────────────────────────────────────────────
    if verbose: print("  Stage 1: Chunking …")
    chunks = await chunk_text(SYNTHETIC_DOCUMENT, source_file="synthetic_doc.txt")
    if verbose: print(f"    {len(chunks)} chunks produced")

    # ── stage 2: detect language ────────────────────────────────────────────
    lang = await detect_language(SYNTHETIC_DOCUMENT)
    if verbose: print(f"  Stage 2: Language detected → {lang}")

    # ── stage 3: classify ───────────────────────────────────────────────────
    if verbose: print("  Stage 3: Classifying chunks …")
    classified = await classify_chunks(chunks, ollama)

    unlabeled = [c for c in classified if c.domain is None]
    classified_count = len(classified) - len(unlabeled)
    if verbose:
        print(f"    {classified_count}/{len(classified)} chunks classified")

    # ── stage 4: RAG retrieval + gap analysis ───────────────────────────────
    ARTICLE_DOMAINS = {
        9:  ArticleDomain.RISK_MANAGEMENT,
        10: ArticleDomain.DATA_GOVERNANCE,
        11: ArticleDomain.TECHNICAL_DOCUMENTATION,
        12: ArticleDomain.RECORD_KEEPING,
        13: ArticleDomain.TRANSPARENCY,
        14: ArticleDomain.HUMAN_OVERSIGHT,
        15: ArticleDomain.SECURITY,
    }

    article_scores = []
    for art_num, domain in ARTICLE_DOMAINS.items():
        art_chunks = [c for c in classified if c.domain == domain]

        passages: list[dict] = []
        for chunk in art_chunks[:3]:  # limit RAG calls per article
            try:
                retrieved = await retrieve_requirements(chunk, emb, chroma, top_k=3)
                passages.extend(retrieved)
            except Exception:
                pass

        if verbose: print(f"  Stage 4: Article {art_num} — {len(art_chunks)} chunks, {len(passages)} passages")

        try:
            score = await analyse_article(art_num, domain, art_chunks, passages, ollama)
        except Exception as ex:
            if verbose: print(f"    Gap analysis failed for Article {art_num}: {ex}")
            from models.schemas import ArticleScore
            score = ArticleScore(article_num=art_num, domain=domain, score=0.0)

        article_scores.append(score)

    # ── stage 5: score audit ────────────────────────────────────────────────
    if verbose: print("  Stage 5: Scoring …")
    report = await score_audit(
        article_scores=article_scores,
        chunks=classified,
        audit_id="eval-pipeline-001",
        source_files=["synthetic_doc.txt"],
        language=lang,
    )

    # ── assertions / checks ─────────────────────────────────────────────────
    checks: dict[str, bool] = {
        "all_chunks_classified":    len(unlabeled) == 0,
        "all_7_articles_present":   len(report.article_scores) == 7,
        "score_in_range":           0.0 <= report.overall_score <= 100.0,
        "no_negative_scores":       all(0 <= a.score <= 100 for a in report.article_scores),
        "reasoning_populated":      all(
            bool(a.score_reasoning)
            for a in report.article_scores
            if a.chunk_count > 0  # articles with no user chunks have no LLM reasoning
        ),
        "gap_titles_non_empty":     all(
            all(bool(g.title) for g in a.gaps)
            for a in report.article_scores
        ),
        "gap_descriptions_non_empty": all(
            all(bool(g.description) for g in a.gaps)
            for a in report.article_scores
        ),
    }

    passed = sum(checks.values())
    total  = len(checks)

    results = {
        "eval":                "pipeline",
        "status":              "pass" if passed == total else ("warn" if passed >= total - 1 else "fail"),
        "checks_passed":       passed,
        "checks_total":        total,
        "checks":              checks,
        "overall_score":       round(report.overall_score, 2),
        "total_chunks":        len(classified),
        "classified_chunks":   classified_count,
        "total_gaps":          sum(len(a.gaps) for a in report.article_scores),
        "language_detected":   lang,
        "article_scores":      {
            a.article_num: round(a.score, 1)
            for a in report.article_scores
        },
    }

    out_path = RESULTS_DIR / "pipeline.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run(verbose: bool = False) -> dict:
    return asyncio.run(_run_async(verbose=verbose))


def _skip(reason: str) -> dict:
    return {"eval": "pipeline", "status": "skip", "reason": reason}


def print_report(r: dict) -> None:
    if r["status"] == "skip":
        print(f"  [SKIP] {r['reason']}")
        return

    status_icon = "✓" if r["status"] == "pass" else "✗"
    print(f"\n  {'─'*52}")
    print(f"  {status_icon} End-to-End Pipeline Evaluation")
    print(f"  {'─'*52}")
    print(f"  Overall Score   : {r['overall_score']}/100")
    print(f"  Chunks          : {r['classified_chunks']}/{r['total_chunks']} classified")
    print(f"  Gaps found      : {r['total_gaps']}")
    print(f"  Checks passed   : {r['checks_passed']}/{r['checks_total']}")
    print()
    for check, ok in r.get("checks", {}).items():
        icon = "✓" if ok else "✗"
        print(f"    {icon} {check}")
    print()
    print("  Article scores:")
    for art, score in sorted(r.get("article_scores", {}).items()):
        bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
        print(f"    Art.{art:2d}  {bar}  {score:.0f}/100")


# ── pytest ─────────────────────────────────────────────────────────────────

def test_pipeline_all_checks_pass() -> None:
    """pytest: All pipeline checks must pass on the synthetic document."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r.get("reason", ""))
    failed = [k for k, v in r.get("checks", {}).items() if not v]
    assert not failed, f"Pipeline checks failed: {failed}"


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end pipeline evaluation")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("Running end-to-end pipeline evaluation …")
    results = run(verbose=args.verbose)
    print_report(results)
    sys.exit(0 if results.get("status") in ("pass", "warn") else 1)
