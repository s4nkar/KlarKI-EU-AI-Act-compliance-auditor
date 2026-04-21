"""Gap analysis service — per-article LLM structured analysis.

Concatenates user document chunks with retrieved regulatory passages,
sends to Ollama with the gap_analysis.txt prompt, parses the JSON response
into an ArticleScore with GapItems and recommendations.
"""

from pathlib import Path

import structlog

from models.schemas import (
    ArticleDomain,
    ArticleScore,
    DocumentChunk,
    GapItem,
    RegulatoryPassage,
    Severity,
)
from services.ollama_client import OllamaClient

logger = structlog.get_logger()

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "gap_analysis.txt"

_DOMAIN_LABELS: dict[ArticleDomain, str] = {
    ArticleDomain.RISK_MANAGEMENT:          "Risk Management System",
    ArticleDomain.DATA_GOVERNANCE:          "Data and Data Governance",
    ArticleDomain.TECHNICAL_DOCUMENTATION:  "Technical Documentation",
    ArticleDomain.RECORD_KEEPING:           "Record-Keeping and Logging",
    ArticleDomain.TRANSPARENCY:             "Transparency and User Information",
    ArticleDomain.HUMAN_OVERSIGHT:          "Human Oversight",
    ArticleDomain.SECURITY:                 "Accuracy, Robustness and Cybersecurity",
}

_SEVERITY_MAP = {
    "critical": Severity.CRITICAL,
    "major":    Severity.MAJOR,
    "minor":    Severity.MINOR,
}


def _parse_gap_item(raw: dict, article_num: int) -> GapItem | None:
    """Parse a single gap dict from LLM output into a GapItem."""
    try:
        severity_raw = str(raw.get("severity", "minor")).lower()
        severity = _SEVERITY_MAP.get(severity_raw, Severity.MINOR)
        return GapItem(
            title=str(raw.get("title", "Unnamed gap"))[:80],
            description=str(raw.get("description", "")),
            severity=severity,
            article_num=article_num,
        )
    except Exception as exc:
        logger.warning("gap_parse_failed", raw=raw, error=str(exc))
        return None


async def analyse_article(
    article_num: int,
    domain: ArticleDomain,
    user_chunks: list[DocumentChunk],
    regulatory_passages: list[dict],
    ollama: OllamaClient,
    applicable_articles: list[int] | None = None,
) -> ArticleScore:
    """Perform gap analysis for a single EU AI Act article.

    Builds a prompt from user documentation and retrieved regulatory text,
    calls Ollama with JSON output mode, and parses the result into ArticleScore.

    Args:
        article_num: EU AI Act article number (9–15).
        domain: ArticleDomain enum value for this article.
        user_chunks: Document chunks classified to this domain.
        regulatory_passages: Top-k passages from ChromaDB for this domain.
        ollama: OllamaClient instance.
        applicable_articles: Articles that apply per Article 6 gate. When provided
            and non-empty, articles outside this list are skipped (not applicable).

    Returns:
        ArticleScore with score, gaps, and recommendations populated.
    """
    # Phase 3: skip LLM for articles outside the applicability gate
    if applicable_articles and article_num not in applicable_articles:
        logger.info("gap_not_applicable", article_num=article_num)
        return ArticleScore(
            article_num=article_num,
            domain=domain,
            score=100.0,
            gaps=[],
            recommendations=[],
            chunk_count=0,
            score_reasoning=(
                "Article not applicable — Article 6 gate determined this system "
                "does not require compliance with this article."
            ),
        )

    # If no user chunks for this domain, return zero score with a single gap
    if not user_chunks:
        logger.info("gap_no_chunks", article_num=article_num, domain=domain.value)
        return ArticleScore(
            article_num=article_num,
            domain=domain,
            score=0.0,
            gaps=[GapItem(
                title="No documentation found for this article",
                description=(
                    f"No document sections were classified under "
                    f"'{_DOMAIN_LABELS.get(domain, domain.value)}'. "
                    f"This suggests the uploaded documentation does not address "
                    f"Article {article_num} requirements."
                ),
                severity=Severity.CRITICAL,
                article_num=article_num,
            )],
            recommendations=[
                f"Create dedicated documentation addressing Article {article_num} requirements.",
                "Review the EU AI Act Article requirements and map them to existing policies.",
            ],
            chunk_count=0,
        )

    # Filter out header/boilerplate chunks (< 80 chars) — they add noise without
    # contributing evidence. Fall back to the original list if filtering removes everything.
    meaningful_chunks = [c for c in user_chunks if len(c.text.strip()) >= 80]
    if not meaningful_chunks:
        meaningful_chunks = user_chunks

    # Phase 3 Option B: Invoke LangGraph Multi-Agent Workflow
    from services.agent_graph import build_audit_graph

    graph = build_audit_graph()

    initial_state = {
        "article_num": article_num,
        "domain": domain,
        "user_chunks": meaningful_chunks,
        "regulatory_passages": regulatory_passages,
        "ollama_client": ollama,
    }

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("gap_langgraph_failed", article_num=article_num, error=str(exc))
        # Fallback: return partial score without LLM analysis
        return ArticleScore(
            article_num=article_num,
            domain=domain,
            score=30.0,
            gaps=[GapItem(
                title="Analysis failed — review manually",
                description=f"LangGraph analysis could not be completed: {str(exc)[:200]}",
                severity=Severity.MAJOR,
                article_num=article_num,
            )],
            recommendations=["Retry the audit or review documentation manually."],
            chunk_count=len(user_chunks),
            score_reasoning="LangGraph analysis failed — manual review required.",
        )

    # Parse final_state output
    score = final_state.get("final_score", 50.0)
    score = max(0.0, min(100.0, float(score)))
    
    score_reasoning = str(final_state.get("reasoning", "")).strip()

    # Parse gaps — skip any gap with an empty title or description
    gaps: list[GapItem] = []
    for raw_gap in final_state.get("gaps", []):
        if isinstance(raw_gap, dict):
            gap = _parse_gap_item(raw_gap, article_num)
            if gap and gap.title.strip() and gap.description.strip():
                gaps.append(gap)

    # Parse recommendations
    recommendations: list[str] = [
        str(r) for r in final_state.get("recommendations", [])
        if isinstance(r, str) and r.strip()
    ]

    # Build regulatory passage objects from retrieved ChromaDB results
    passages: list[RegulatoryPassage] = []
    for p in regulatory_passages:
        meta = p.get("metadata", {})
        passages.append(RegulatoryPassage(
            title=meta.get("title", meta.get("requirement_id", meta.get("article", ""))),
            text=p.get("text", ""),
            regulation=meta.get("regulation", meta.get("source", "")),
            article_ref=meta.get("article_ref", meta.get("article", "")),
        ))

    logger.info(
        "gap_analysis_done",
        article_num=article_num,
        score=score,
        gaps=len(gaps),
        chunks=len(user_chunks),
    )

    return ArticleScore(
        article_num=article_num,
        domain=domain,
        score=score,
        gaps=gaps,
        recommendations=recommendations,
        chunk_count=len(user_chunks),
        score_reasoning=score_reasoning,
        regulatory_passages=passages,
    )
