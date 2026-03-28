"""Compliance scoring service — aggregates article scores into a report. (Phase 2)

Computes weighted overall score and applies rule-based Annex III risk tier
classification by scanning chunk text for high-risk keywords.
"""

import structlog
from models.schemas import ArticleScore, ComplianceReport, DocumentChunk, RiskTier

logger = structlog.get_logger()

# Annex III / Article 5 keyword lists for rule-based risk tier classification
_PROHIBITED_KEYWORDS = [
    "social scoring", "soziales scoring", "emotion recognition workplace",
    "emotion recognition education", "gefühlserkennung arbeitsplatz",
    "subliminal manipulation", "unterschwellige manipulation",
    "real-time biometric", "echtzeit-biometrie",
]
_HIGH_RISK_KEYWORDS = [
    "biometric", "biometrisch", "recruitment", "personalauswahl",
    "credit score", "kreditbewertung", "medical diagnosis", "medizinische diagnose",
    "critical infrastructure", "kritische infrastruktur",
    "law enforcement", "strafverfolgung", "border control", "grenzkontrolle",
    "education assessment", "bildungsbewertung",
]


async def score_audit(
    article_scores: list[ArticleScore],
    chunks: list[DocumentChunk],
) -> ComplianceReport:
    """Aggregate per-article scores into a full ComplianceReport.

    Overall score is weighted average (equal weight per article).
    Risk tier is derived from classify_risk_tier() keyword scan.

    Args:
        article_scores: List of ArticleScore objects (one per article 9–15).
        chunks: All document chunks (used for risk tier classification).

    Returns:
        ComplianceReport with overall_score, risk_tier, and all article detail.
    """
    raise NotImplementedError("compliance_scorer.score_audit — implemented in Phase 2")


def classify_risk_tier(chunks: list[DocumentChunk]) -> RiskTier:
    """Rule-based Annex III risk tier classification.

    Scans all chunk text for prohibited and high-risk keywords.
    Returns PROHIBITED if any Article 5 keywords are found,
    HIGH if any Annex III keywords are found, otherwise LIMITED/MINIMAL.

    Args:
        chunks: All document chunks from the uploaded documents.

    Returns:
        RiskTier enum value.
    """
    raise NotImplementedError("compliance_scorer.classify_risk_tier — implemented in Phase 2")
