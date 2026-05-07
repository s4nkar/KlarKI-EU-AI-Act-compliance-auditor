"""
Evaluation 3 — End-to-End Pipeline Test.

Mirrors _run_pipeline() in api/routers/audit.py exactly:

  proposition_chunk_text → detect_language
  → classify_actor (asyncio.to_thread) → check_applicability (asyncio.to_thread)
  → classify_chunks → enrich_chunks_with_ner (asyncio.to_thread)
  → build_bm25_index
  → [retrieve_requirements + analyse_article/LangGraph per article] (concurrent)
  → map_evidence (asyncio.to_thread)
  → check_emotion_recognition → score_audit

The synthetic document contains explicit "credit scoring" language so the
applicability engine flags it as ESSENTIAL_SERVICES (Annex III) → high-risk,
ensuring applicable_articles=[9..15] and all 7 articles enter LangGraph.

Checks:
  • Every chunk receives a domain label (no unclassified chunks)
  • All 7 articles appear in the report
  • Overall score is in [0, 100]
  • Each ArticleScore with user chunks has score_reasoning populated
  • Each gap has a non-empty title and description
  • Phase 3: actor, applicability, evidence_map are all populated in the report

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

# Includes "credit scoring" to trigger ESSENTIAL_SERVICES Annex III pattern so
# the applicability engine marks the document high-risk and all 7 articles enter
# the LangGraph gap analysis path (applicable_articles=[9..15]).
SYNTHETIC_DOCUMENT = """
ARTIFICIAL INTELLIGENCE SYSTEM COMPLIANCE DOCUMENTATION
System: AI-Powered Credit Scoring and Loan Assessment

This document describes the compliance posture of our AI-based credit scoring system
used in retail banking to assess loan applications and determine creditworthiness.
The creditworthiness assessment automation uses a gradient-boosted classifier.

1. Risk Management
The risk management system identifies potential hazards in our AI classification model.
We conduct regular risk assessments and maintain a risk register with mitigation measures.
Known risks include model drift, adversarial inputs, and demographic bias in predictions.

2. Data Governance
Training data was sourced from internal financial records and validated for quality.
Data bias analysis was performed across gender, age, and ethnicity dimensions.
Training and validation datasets are maintained separately with documented lineage.

3. Technical Documentation
The system uses a gradient-boosted classifier with 250 decision trees.
Architecture and model card specifications are documented in the engineering dossier.
System design rationale is recorded in the technical file per Annex IV requirements.

4. Record Keeping
All credit scoring decisions are logged with timestamp, input hash, and confidence score.
Logs are retained for 36 months and protected from unauthorized modification.

5. Transparency
Applicants are notified when automated credit scoring is used for their application.
The system provides a reason code and confidence breakdown for every credit decision.

6. Human Oversight
A credit analyst reviews all high-stakes decisions flagged by the system.
Human override is available at every decision point with full audit logging.

7. Security and Accuracy
The system achieves 91% accuracy on the held-out validation set.
Adversarial robustness testing was conducted and security assessment completed per ISO 27001.
"""


async def _run_async(verbose: bool) -> dict:
    import os

    try:
        from services.chunker              import proposition_chunk_text
        from services.language_detector    import detect_language
        from services.actor_classifier     import classify_actor
        from services.applicability_engine import check_applicability
        from services.classifier           import classify_chunks
        from services.ner_service          import enrich_chunks_with_ner
        from services.embedding_service    import EmbeddingService
        from services.chroma_client        import ChromaClient
        from services.rag_engine           import retrieve_requirements, build_bm25_index
        from services.gap_analyser         import analyse_article
        from services.evidence_mapper      import map_evidence
        from services.emotion_module       import check_emotion_recognition
        from services.compliance_scorer    import score_audit, ARTICLE_DOMAINS
        from services.ollama_client        import OllamaClient
        from models.schemas                import ArticleDomain
    except ImportError as e:
        return _skip(f"Cannot import API services: {e}")

    ollama_host  = os.getenv("OLLAMA_HOST",  "localhost")
    chroma_host  = os.getenv("CHROMADB_HOST", "localhost")
    ollama_model = os.getenv("OLLAMA_MODEL",  "phi3:mini")

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

    # ── stage 1: proposition chunking ──────────────────────────────────────
    if verbose:
        print("  Stage 1: Chunking (proposition_chunk_text) …")
    chunks = await proposition_chunk_text(SYNTHETIC_DOCUMENT, source_file="synthetic_doc.txt")
    if verbose:
        print(f"    {len(chunks)} chunks produced")

    # ── stage 2: language detection ────────────────────────────────────────
    lang = await detect_language(SYNTHETIC_DOCUMENT)
    for chunk in chunks:
        chunk.language = lang
    if verbose:
        print(f"  Stage 2: Language detected → {lang}")

    # ── stage 3: Phase 3 legal decision hierarchy ───────────────────────────
    # Mirrors audit.py: both run via asyncio.to_thread before classify_chunks.
    if verbose:
        print("  Stage 3: Actor classification + applicability gate …")
    actor_result        = await asyncio.to_thread(classify_actor, SYNTHETIC_DOCUMENT)
    applicability_result = await asyncio.to_thread(check_applicability, chunks)
    applicable_articles  = applicability_result.applicable_articles
    if verbose:
        print(f"    Actor: {actor_result.actor_type.value} "
              f"(conf={actor_result.confidence:.2f})")
        print(f"    Applicable articles: {applicable_articles}  "
              f"(prohibited={applicability_result.is_prohibited}, "
              f"high_risk={applicability_result.is_high_risk})")

    # ── stage 4: chunk classification ──────────────────────────────────────
    if verbose:
        print("  Stage 4: Classifying chunks …")
    chunks, _backend = await classify_chunks(chunks, ollama)
    unlabeled        = [c for c in chunks if c.domain is None]
    classified_count = len(chunks) - len(unlabeled)
    if verbose:
        print(f"    {classified_count}/{len(chunks)} chunks classified")

    # ── stage 5: NER enrichment ─────────────────────────────────────────────
    if verbose:
        print("  Stage 5: NER enrichment …")
    chunks = await asyncio.to_thread(enrich_chunks_with_ner, chunks)

    # ── stage 6: BM25 index (built at app startup in production) ────────────
    if verbose:
        print("  Stage 6: Building BM25 index …")
    await build_bm25_index(chroma)

    # ── stage 7: RAG + LangGraph gap analysis — all 7 articles concurrently ─
    if verbose:
        print("  Stage 7: RAG + LangGraph gap analysis (7 articles concurrently) …")

    domain_chunks: dict[ArticleDomain, list] = {d: [] for d in ArticleDomain}
    for chunk in chunks:
        if chunk.domain:
            domain_chunks[chunk.domain].append(chunk)

    async def _process_article(article_num: int, domain: ArticleDomain):
        art_chunks  = domain_chunks.get(domain, [])
        reg_passages: list[dict] = []

        for c in art_chunks[:3]:
            try:
                retrieved = await retrieve_requirements(
                    chunk=c,
                    embedding_service=emb,
                    chroma_client=chroma,
                    top_k=5,
                    applicable_articles=applicable_articles,
                    regulation="eu_ai_act",
                )
                reg_passages.extend(retrieved)
            except Exception:
                pass

        # Deduplicate passages (same logic as production process_article)
        seen: set[str] = set()
        unique: list[dict] = []
        for p in reg_passages:
            key = p.get("id") or p.get("text")
            if key not in seen:
                seen.add(key)
                unique.append(p)
        reg_passages = unique[:5]

        if verbose:
            applicable = (
                not applicable_articles or article_num in applicable_articles
            )
            print(f"    Article {article_num}: {len(art_chunks)} chunks, "
                  f"{len(reg_passages)} passages, "
                  f"{'applicable' if applicable else 'not applicable'}")

        return await analyse_article(
            article_num=article_num,
            domain=domain,
            user_chunks=art_chunks,
            regulatory_passages=reg_passages,
            ollama=ollama,
            applicable_articles=applicable_articles,
        )

    article_scores = list(await asyncio.gather(
        *[_process_article(num, dom) for num, dom in ARTICLE_DOMAINS.items()]
    ))

    # ── stage 8: evidence mapping (deterministic, no LLM) ───────────────────
    if verbose:
        print("  Stage 8: Evidence mapping …")
    evidence_map = await asyncio.to_thread(
        map_evidence,
        chunks,
        actor_result.actor_type,
        applicable_articles,
    )
    if verbose:
        print(f"    {evidence_map.total_obligations} obligations, "
              f"coverage={evidence_map.overall_coverage:.0%}")

    # ── stage 9: emotion scan + scoring ─────────────────────────────────────
    if verbose:
        print("  Stage 9: Scoring …")
    emotion_flag = await check_emotion_recognition(chunks)
    report = await score_audit(
        article_scores=article_scores,
        chunks=chunks,
        audit_id="eval-pipeline-001",
        source_files=["synthetic_doc.txt"],
        language=lang,
        emotion_flag=emotion_flag,
        actor=actor_result,
        applicability=applicability_result,
        evidence_map=evidence_map,
    )

    # ── assertions ───────────────────────────────────────────────────────────
    checks: dict[str, bool] = {
        "all_chunks_classified":      len(unlabeled) == 0,
        "all_7_articles_present":     len(report.article_scores) == 7,
        "score_in_range":             0.0 <= report.overall_score <= 100.0,
        "no_negative_scores":         all(0 <= a.score <= 100 for a in report.article_scores),
        "reasoning_populated":        all(
            bool(a.score_reasoning)
            for a in report.article_scores
            if a.chunk_count > 0
        ),
        "gap_titles_non_empty":       all(
            all(bool(g.title) for g in a.gaps)
            for a in report.article_scores
        ),
        "gap_descriptions_non_empty": all(
            all(bool(g.description) for g in a.gaps)
            for a in report.article_scores
        ),
        # Phase 3 checks
        "actor_populated":            report.actor is not None,
        "actor_type_valid":           (
            report.actor is not None and report.actor.actor_type is not None
        ),
        "applicability_populated":    report.applicability is not None,
        "evidence_map_populated":     report.evidence_map is not None,
        "confidence_score_is_float":  isinstance(report.confidence_score, float),
    }

    passed = sum(checks.values())
    total  = len(checks)

    results = {
        "eval":                  "pipeline",
        "status":                "pass" if passed == total else ("warn" if passed >= total - 1 else "fail"),
        "checks_passed":         passed,
        "checks_total":          total,
        "checks":                checks,
        "overall_score":         round(report.overall_score, 2),
        "total_chunks":          len(chunks),
        "classified_chunks":     classified_count,
        "total_gaps":            sum(len(a.gaps) for a in report.article_scores),
        "language_detected":     lang,
        "actor":                 report.actor.actor_type.value if report.actor else None,
        "actor_confidence":      round(report.actor.confidence, 3) if report.actor else None,
        "is_prohibited":         report.applicability.is_prohibited if report.applicability else None,
        "is_high_risk":          report.applicability.is_high_risk if report.applicability else None,
        "applicable_articles":   applicable_articles,
        "evidence_coverage":     round(evidence_map.overall_coverage, 3),
        "confidence_score":      round(report.confidence_score, 3),
        "requires_human_review": report.requires_human_review,
        "article_scores":        {
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
    print(f"  {status_icon} End-to-End Pipeline Evaluation (Phase 3)")
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
    if r.get("actor"):
        print(f"  Actor           : {r['actor']} (conf={r.get('actor_confidence', '?')})")
    if r.get("is_high_risk") is not None:
        tier = (
            "prohibited" if r.get("is_prohibited")
            else ("high-risk" if r["is_high_risk"] else "minimal")
        )
        print(f"  Risk tier       : {tier}")
    if r.get("applicable_articles") is not None:
        print(f"  Applicable arts : {r['applicable_articles']}")
    if r.get("evidence_coverage") is not None:
        print(f"  Evidence cov.   : {r['evidence_coverage']:.0%}")
    if r.get("confidence_score") is not None:
        print(f"  Confidence      : {r['confidence_score']:.0%}")
    print(f"  Human review    : {'required' if r.get('requires_human_review') else 'not required'}")

    print()
    print("  Article scores:")
    for art, score in sorted(r.get("article_scores", {}).items()):
        bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
        print(f"    Art.{art:2d}  {bar}  {score:.0f}/100")


# ── pytest ──────────────────────────────────────────────────────────────────

def test_pipeline_all_checks_pass() -> None:
    """pytest: All pipeline checks must pass on the synthetic document."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r.get("reason", ""))
    failed = [k for k, v in r.get("checks", {}).items() if not v]
    assert not failed, f"Pipeline checks failed: {failed}"


def test_pipeline_phase3_fields_present() -> None:
    """pytest: Phase 3 fields (actor, applicability, evidence_map) must be populated."""
    r = run()
    if r["status"] == "skip":
        import pytest
        pytest.skip(r.get("reason", ""))
    assert r.get("actor") is not None, "actor field is None — actor_classifier did not run"
    assert r.get("is_high_risk") is not None, "applicability field missing"
    assert r.get("evidence_coverage") is not None, "evidence_map missing"


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end pipeline evaluation (Phase 3)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("Running end-to-end pipeline evaluation …")
    results = run(verbose=args.verbose)
    print_report(results)
    sys.exit(0 if results.get("status") in ("pass", "warn") else 1)
