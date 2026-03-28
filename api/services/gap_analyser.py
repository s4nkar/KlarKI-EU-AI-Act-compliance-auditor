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


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _build_prompt(
    article_num: int,
    domain: ArticleDomain,
    user_chunks: list[DocumentChunk],
    regulatory_passages: list[dict],
) -> str:
    domain_label = _DOMAIN_LABELS.get(domain, domain.value)

    user_text = "\n\n".join(c.text for c in user_chunks[:10]) or "(no relevant documentation found)"
    reg_text = "\n\n".join(
        f"[{p['metadata'].get('title', p['metadata'].get('requirement_id', ''))}]\n{p['text']}"
        for p in regulatory_passages[:8]
    ) or "(no regulatory passages retrieved)"

    template = _load_prompt()
    return (
        template
        .replace("{article_num}", str(article_num))
        .replace("{domain_label}", domain_label)
        .replace("{user_text}", user_text[:3000])
        .replace("{regulatory_text}", reg_text[:3000])
    )


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

    Returns:
        ArticleScore with score, gaps, and recommendations populated.
    """
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

    prompt = _build_prompt(article_num, domain, user_chunks, regulatory_passages)

    try:
        result = await ollama.generate_json(prompt)
    except Exception as exc:
        logger.error("gap_llm_failed", article_num=article_num, error=str(exc))
        # Fallback: return partial score without LLM analysis
        return ArticleScore(
            article_num=article_num,
            domain=domain,
            score=30.0,
            gaps=[GapItem(
                title="Analysis failed — review manually",
                description=f"LLM analysis could not be completed: {str(exc)[:200]}",
                severity=Severity.MAJOR,
                article_num=article_num,
            )],
            recommendations=["Retry the audit or review documentation manually."],
            chunk_count=len(user_chunks),
        )

    # Parse score
    try:
        score = float(result.get("score", 50))
        score = max(0.0, min(100.0, score))
    except (TypeError, ValueError):
        score = 50.0

    # Parse gaps
    gaps: list[GapItem] = []
    for raw_gap in result.get("gaps", []):
        if isinstance(raw_gap, dict):
            gap = _parse_gap_item(raw_gap, article_num)
            if gap:
                gaps.append(gap)

    # Parse recommendations
    recommendations: list[str] = [
        str(r) for r in result.get("recommendations", [])
        if isinstance(r, str) and r.strip()
    ]

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
    )
