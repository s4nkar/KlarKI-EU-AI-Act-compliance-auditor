"""Compliance scoring service — aggregates article scores into a full report.

Computes weighted overall score and applies rule-based Annex III risk tier
classification by scanning chunk text for high-risk keywords.
"""

import uuid
from datetime import datetime, timezone

import structlog

from models.schemas import (
    ArticleDomain,
    ArticleScore,
    ComplianceReport,
    DocumentChunk,
    EmotionFlag,
    RiskTier,
)

logger = structlog.get_logger()

# Article 9–15 domain map (used to ensure all 7 articles appear in output)
ARTICLE_DOMAINS: dict[int, ArticleDomain] = {
    9:  ArticleDomain.RISK_MANAGEMENT,
    10: ArticleDomain.DATA_GOVERNANCE,
    11: ArticleDomain.TECHNICAL_DOCUMENTATION,
    12: ArticleDomain.RECORD_KEEPING,
    13: ArticleDomain.TRANSPARENCY,
    14: ArticleDomain.HUMAN_OVERSIGHT,
    15: ArticleDomain.SECURITY,
}

# Equal weighting across all 7 articles
_ARTICLE_WEIGHT = 1.0 / 7.0

_PROHIBITED_KEYWORDS = [
    "social scoring", "soziales scoring",
    "real-time biometric", "echtzeit-biometrie",
    "subliminal manipulation", "unterschwellige manipulation",
    "emotion recognition workplace", "emotion recognition education",
    "gefühlserkennung arbeitsplatz", "emotionserkennung schule",
]

_HIGH_RISK_KEYWORDS = [
    "biometric", "biometrisch",
    "recruitment", "personalauswahl", "bewerbermanagement",
    "credit score", "kreditbewertung", "kreditwürdigkeit",
    "medical diagnosis", "medizinische diagnose",
    "critical infrastructure", "kritische infrastruktur",
    "law enforcement", "strafverfolgung",
    "border control", "grenzkontrolle",
    "education assessment", "bildungsbewertung",
    "employment decision", "beschäftigungsentscheidung",
]


async def score_audit(
    article_scores: list[ArticleScore],
    chunks: list[DocumentChunk],
    audit_id: str | None = None,
    source_files: list[str] | None = None,
    language: str = "en",
    emotion_flag: EmotionFlag | None = None,
) -> ComplianceReport:
    """Aggregate per-article scores into a full ComplianceReport.

    Overall score is the equally-weighted average across all 7 articles.
    Risk tier is derived from classify_risk_tier() keyword scan.

    Args:
        article_scores: List of ArticleScore objects (one per article 9–15).
        chunks: All document chunks (used for risk tier classification).
        audit_id: Unique audit identifier (generated if not provided).
        source_files: Filenames of uploaded documents.
        language: Primary detected language.

    Returns:
        ComplianceReport with overall_score, risk_tier, and all article detail.
    """
    # Ensure we have a score for all 7 articles
    scored_articles = {s.article_num: s for s in article_scores}
    for art_num, domain in ARTICLE_DOMAINS.items():
        if art_num not in scored_articles:
            scored_articles[art_num] = ArticleScore(
                article_num=art_num,
                domain=domain,
                score=0.0,
                chunk_count=0,
            )

    final_scores = list(scored_articles.values())
    overall = sum(s.score for s in final_scores) * _ARTICLE_WEIGHT

    risk_tier = classify_risk_tier(chunks)
    classified = sum(1 for c in chunks if c.domain is not None)

    report = ComplianceReport(
        audit_id=audit_id or str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        source_files=source_files or [],
        language=language,
        risk_tier=risk_tier,
        overall_score=round(overall, 1),
        article_scores=sorted(final_scores, key=lambda s: s.article_num),
        emotion_flag=emotion_flag or EmotionFlag(),
        total_chunks=len(chunks),
        classified_chunks=classified,
    )

    logger.info(
        "scoring_done",
        audit_id=report.audit_id,
        overall=report.overall_score,
        risk_tier=risk_tier.value,
        chunks=len(chunks),
    )
    return report


def classify_risk_tier(chunks: list[DocumentChunk]) -> RiskTier:
    """Rule-based Annex III risk tier classification.

    Scans all chunk text (lowercased) for prohibited and high-risk keywords.
    Returns PROHIBITED if any Article 5 keywords are found,
    HIGH if any Annex III keywords are found, otherwise MINIMAL.

    Args:
        chunks: All document chunks from the uploaded documents.

    Returns:
        RiskTier enum value.
    """
    combined = " ".join(c.text.lower() for c in chunks)

    for kw in _PROHIBITED_KEYWORDS:
        if kw in combined:
            logger.info("risk_tier_prohibited", keyword=kw)
            return RiskTier.PROHIBITED

    for kw in _HIGH_RISK_KEYWORDS:
        if kw in combined:
            logger.info("risk_tier_high", keyword=kw)
            return RiskTier.HIGH

    return RiskTier.MINIMAL
