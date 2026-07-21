"""Compliance scoring service — aggregates article scores into a full report.

Computes the weighted overall score. Risk tier is taken from the
applicability engine's result (services/applicability_engine.py), which is
the sole authority for Article 5/6/Annex III classification.
"""

import uuid
from datetime import datetime, timezone

import structlog

from models.schemas import (
    ActorClassification,
    ApplicabilityResult,
    ArticleDomain,
    ArticleScore,
    ComplianceReport,
    DocumentChunk,
    EmotionFlag,
    EvidenceMap,
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


async def score_audit(
    article_scores: list[ArticleScore],
    chunks: list[DocumentChunk],
    audit_id: str | None = None,
    source_files: list[str] | None = None,
    language: str = "en",
    emotion_flag: EmotionFlag | None = None,
    classifier_backend: str = "ollama/phi3:mini",
    wizard_risk_tier: RiskTier | None = None,
    actor: ActorClassification | None = None,
    applicability: ApplicabilityResult | None = None,
    evidence_map: EvidenceMap | None = None,
    model_versions: dict[str, str] | None = None,
) -> ComplianceReport:
    """Aggregate per-article scores into a full ComplianceReport.

    risk_tier is taken directly from the applicability engine result
    (Article 5/6 + Annex III pattern gate) — the sole authority for this
    determination. If applicability is not supplied (should only happen in
    isolated tests), risk_tier defaults to MINIMAL rather than re-deriving it
    from a second keyword scan, since that would just reintroduce a second
    source of truth that can drift from applicability_engine.py.
    wizard_risk_tier is stored separately as a user self-assessment for comparison.

    Args:
        article_scores: List of ArticleScore objects (one per article 9–15).
        chunks: All document chunks.
        audit_id: Unique audit identifier (generated if not provided).
        source_files: Filenames of uploaded documents.
        language: Primary detected language.
        actor: Article 3 actor classification result.
        applicability: Article 6 + Annex III applicability gate result.

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

    # Phase 3: average only applicable articles so minimal-risk systems are not
    # penalised for Articles 9–15 they do not need to satisfy.
    # Three cases:
    #   applicability is None          → no Phase 3 info, fall back to 7-article average
    #   applicable_articles is empty   → minimal risk, nothing applies → 100 %
    #   applicable_articles non-empty  → average only those articles
    if applicability is not None:
        if not applicability.applicable_articles:
            overall = 100.0
        else:
            scored_applicable = [
                s for s in final_scores
                if s.article_num in applicability.applicable_articles
            ]
            overall = (
                sum(s.score for s in scored_applicable) / len(scored_applicable)
                if scored_applicable else 100.0
            )
    else:
        overall = sum(s.score for s in final_scores) * _ARTICLE_WEIGHT

    # Phase 3: applicability engine is authoritative for risk_tier
    if applicability is not None:
        if applicability.is_prohibited:
            risk_tier = RiskTier.PROHIBITED
        elif applicability.is_high_risk:
            risk_tier = RiskTier.HIGH
        else:
            risk_tier = RiskTier.MINIMAL
    else:
        logger.warning("risk_tier_no_applicability", audit_id=audit_id)
        risk_tier = RiskTier.MINIMAL

    classified = sum(1 for c in chunks if c.domain is not None)

    # Phase 3: Calculate Confidence Score & Human Review Flag
    confidences = []
    
    # 1. Actor classification confidence
    if actor is not None:
        confidences.append(actor.confidence)
        
    # 2. Evidence map density
    if evidence_map is not None:
        if evidence_map.total_obligations > 0:
            confidences.append(0.5 + (evidence_map.overall_coverage / 2))
        elif applicability is not None and applicability.is_high_risk:
            # High-risk system but obligation files returned 0 matches —
            # treat as neutral rather than dropping the component entirely,
            # which would otherwise over-weight the actor confidence.
            confidences.append(0.5)
        
    # 3. Chunk classification ratio
    if chunks:
        confidences.append(classified / len(chunks))
        
    overall_confidence = sum(confidences) / len(confidences) if confidences else 1.0
    
    requires_human_review = overall_confidence < 0.70
    if actor is not None and actor.actor_type.value == "unknown":
        requires_human_review = True

    report = ComplianceReport(
        audit_id=audit_id or str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        source_files=source_files or [],
        language=language,
        risk_tier=risk_tier,
        wizard_risk_tier=wizard_risk_tier,
        overall_score=round(overall, 1),
        article_scores=sorted(final_scores, key=lambda s: s.article_num),
        emotion_flag=emotion_flag or EmotionFlag(),
        total_chunks=len(chunks),
        classified_chunks=classified,
        classifier_backend=classifier_backend,
        actor=actor,
        applicability=applicability,
        evidence_map=evidence_map,
        confidence_score=round(overall_confidence, 2),
        requires_human_review=requires_human_review,
        model_versions=model_versions or {},
    )

    logger.info(
        "scoring_done",
        audit_id=report.audit_id,
        overall=report.overall_score,
        risk_tier=risk_tier.value,
        chunks=len(chunks),
    )
    return report
