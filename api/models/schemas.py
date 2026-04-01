"""All Pydantic v2 data models for KlarKI.

These schemas are shared across the API layer, services, and tests.
Every public endpoint and service function uses these types exclusively.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class ArticleDomain(str, Enum):
    """Maps each EU AI Act article (9–15) to a human-readable domain label."""
    RISK_MANAGEMENT = "risk_management"          # Article 9
    DATA_GOVERNANCE = "data_governance"          # Article 10
    TECHNICAL_DOCUMENTATION = "technical_documentation"  # Article 11
    RECORD_KEEPING = "record_keeping"            # Article 12
    TRANSPARENCY = "transparency"                # Article 13
    HUMAN_OVERSIGHT = "human_oversight"          # Article 14
    SECURITY = "security"                        # Article 15
    UNRELATED = "unrelated"


class RiskTier(str, Enum):
    """EU AI Act risk classification tiers (Annex III + Article 5)."""
    PROHIBITED = "prohibited"   # Article 5 — absolute ban
    HIGH = "high"               # Annex III — conformity assessment required
    LIMITED = "limited"         # Article 52 — transparency obligations only
    MINIMAL = "minimal"         # No specific obligations


class Severity(str, Enum):
    """Gap severity — drives remediation priority in reports."""
    CRITICAL = "critical"   # Blocks deployment / legal exposure
    MAJOR = "major"         # Significant gap, must fix before go-live
    MINOR = "minor"         # Best-practice improvement


class AuditStatus(str, Enum):
    """Pipeline stage for async audit processing."""
    UPLOADING = "uploading"
    PARSING = "parsing"
    CLASSIFYING = "classifying"
    ANALYSING = "analysing"
    SCORING = "scoring"
    COMPLETE = "complete"
    FAILED = "failed"


# ── Document models ───────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """A single text chunk extracted from a user-uploaded document.

    Created by the chunker service and enriched by the classifier service.
    """
    chunk_id: str = Field(description="UUID4 assigned at chunk time")
    text: str = Field(description="Raw chunk text (max ~512 chars)")
    source_file: str = Field(description="Original filename")
    chunk_index: int = Field(description="0-based position within source file")
    language: str = Field(default="en", description="ISO-639-1 language code")
    domain: ArticleDomain | None = Field(
        default=None,
        description="Article domain assigned by classifier; None until classified",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Arbitrary extra metadata (page number, section heading, etc.)",
    )


# ── Report models ─────────────────────────────────────────────────────────────

class GapItem(BaseModel):
    """A single compliance gap identified by the gap analyser for an article."""
    title: str = Field(description="Short gap title (max 10 words)")
    description: str = Field(description="Detailed gap description referencing specific requirements")
    severity: Severity
    article_num: int = Field(description="EU AI Act article number (9–15)")


class RegulatoryPassage(BaseModel):
    """A single regulatory text passage retrieved from ChromaDB for gap analysis."""
    title: str = Field(description="Article/requirement identifier from metadata")
    text: str = Field(description="The regulatory passage text")
    regulation: str = Field(default="", description="Source regulation (eu_ai_act / gdpr)")
    article_ref: str = Field(default="", description="Human-readable article reference e.g. 'Art. 9 §1'")


class ArticleScore(BaseModel):
    """Compliance assessment for a single EU AI Act article."""
    article_num: int = Field(description="EU AI Act article number (9–15)")
    domain: ArticleDomain
    score: float = Field(ge=0, le=100, description="0=fully non-compliant, 100=fully compliant")
    gaps: list[GapItem] = Field(default_factory=list)
    recommendations: list[str] = Field(
        default_factory=list,
        description="Ordered, actionable remediation steps",
    )
    chunk_count: int = Field(
        default=0,
        description="Number of document chunks classified to this domain",
    )
    score_reasoning: str = Field(
        default="",
        description="LLM explanation of why this score was assigned",
    )
    regulatory_passages: list[RegulatoryPassage] = Field(
        default_factory=list,
        description="Regulatory text passages retrieved from ChromaDB and used in gap analysis",
    )


class EmotionFlag(BaseModel):
    """Result of the Article 5 emotion recognition scan (Phase 4)."""
    detected: bool = False
    context: str = Field(default="", description="Surrounding text that triggered detection")
    is_prohibited: bool = Field(
        default=False,
        description="True if use-case falls under Art. 5 prohibition (workplace/education)",
    )
    explanation: str = Field(default="", description="Human-readable explanation of the flag")


class ComplianceReport(BaseModel):
    """Full compliance report produced at the end of a successful audit pipeline."""
    audit_id: str
    created_at: datetime
    source_files: list[str] = Field(description="Filenames of uploaded documents")
    language: str = Field(description="Detected primary language ('de' or 'en')")
    risk_tier: RiskTier
    wizard_risk_tier: RiskTier | None = Field(
        default=None,
        description="Risk tier self-assessed via Annex III wizard before the audit, if completed",
    )
    overall_score: float = Field(ge=0, le=100, description="Weighted average across all articles")
    article_scores: list[ArticleScore] = Field(description="One entry per EU AI Act article 9–15")
    emotion_flag: EmotionFlag = Field(default_factory=EmotionFlag)
    total_chunks: int = Field(description="Total chunks extracted from all documents")
    classified_chunks: int = Field(description="Chunks successfully assigned a domain")
    classifier_backend: str = Field(
        default="ollama/phi3:mini",
        description="Classifier backend used: 'ollama/<model>' or 'triton/gbert-base'",
    )


# ── API envelope models ────────────────────────────────────────────────────────

class AuditResponse(BaseModel):
    """Response wrapper for audit endpoints — includes status and optional report."""
    audit_id: str
    status: AuditStatus
    report: ComplianceReport | None = None


class APIResponse(BaseModel):
    """Standard JSON envelope for all KlarKI API responses."""
    status: str = Field(default="success", description="'success' or 'error'")
    data: dict | None = None
    error: str | None = None
