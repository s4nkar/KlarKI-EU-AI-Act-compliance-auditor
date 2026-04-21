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


class ActorType(str, Enum):
    """EU AI Act Article 3 actor roles — determines applicable obligation tree."""
    PROVIDER = "provider"         # Art. 3(3) — develops + places on market
    DEPLOYER = "deployer"         # Art. 3(4) — uses under own authority
    IMPORTER = "importer"         # Art. 3(6) — places third-party system on EU market
    DISTRIBUTOR = "distributor"   # Art. 3(7) — makes available without modification
    UNKNOWN = "unknown"           # Could not determine from document


class AnnexIIICategory(int, Enum):
    """Annex III high-risk AI system categories (Article 6(2))."""
    BIOMETRIC = 1               # Biometric identification and categorisation
    CRITICAL_INFRASTRUCTURE = 2 # Critical infrastructure management
    EDUCATION = 3               # Education and vocational training
    EMPLOYMENT = 4              # Employment and workers management
    ESSENTIAL_SERVICES = 5      # Essential private/public services
    LAW_ENFORCEMENT = 6         # Law enforcement
    MIGRATION = 7               # Migration, asylum, border control
    JUSTICE = 8                 # Administration of justice and democratic processes


# ── Phase 3 — Applicability & obligation models ───────────────────────────────

class ObligationSchema(BaseModel):
    """Machine-readable legal obligation derived from EU AI Act / GDPR articles."""
    id: str = Field(description="Unique obligation ID, e.g. AIACT_ART9_PROVIDER_RISK_MGMT_001")
    article: str = Field(description="Source article, e.g. 'Article 9'")
    annex: str = Field(default="", description="Source annex if applicable, e.g. 'Annex III'")
    annex_category: int = Field(default=0, description="Annex III category number (0 = not Annex III)")
    actor: list[str] = Field(description="Applicable actor roles: provider / deployer / importer / distributor")
    trigger_keywords: list[str] = Field(description="Keywords that trigger this obligation")
    requirement: str = Field(description="Plain-language statement of the legal requirement")
    evidence_required: list[str] = Field(description="Document artefacts that satisfy this obligation")
    severity: str = Field(description="high / medium / low — reflects regulatory penalty exposure")
    linked_articles: list[str] = Field(default_factory=list, description="Cross-referenced articles")


class AnnexIIIMatch(BaseModel):
    """A single Annex III category matched against the uploaded document."""
    category: AnnexIIICategory
    category_name: str
    matched_keywords: list[str] = Field(description="Keywords found in the document that triggered this match")
    obligation_id: str = Field(description="Obligation schema ID from article_6_annex_iii.jsonl")
    evidence_required: list[str] = Field(description="Evidence the provider/deployer must produce")


class ApplicabilityResult(BaseModel):
    """Output of the Article 6 + Annex III applicability gate.

    Determines whether Articles 9–15 apply before any gap analysis runs.
    This is the legal decision hierarchy entry point.
    """
    is_high_risk: bool = Field(description="True if Article 6 applies (Annex I safety component or Annex III match)")
    is_prohibited: bool = Field(description="True if Article 5 prohibited practice detected")
    annex_iii_matches: list[AnnexIIIMatch] = Field(
        default_factory=list,
        description="All Annex III categories matched; empty means not high-risk via Annex III",
    )
    annex_i_triggered: bool = Field(
        default=False,
        description="True if Article 6(1) safety-component trigger detected (Annex I product)",
    )
    applicable_articles: list[int] = Field(
        default_factory=list,
        description="EU AI Act articles that apply given the risk tier (e.g. [9,10,11,12,13,14,15])",
    )
    reasoning: str = Field(
        default="",
        description="Plain-language explanation of why this applicability determination was reached",
    )


class ActorClassification(BaseModel):
    """Result of Article 3 actor role detection from document text."""
    actor_type: ActorType
    confidence: float = Field(ge=0.0, le=1.0, description="Pattern match confidence 0–1")
    matched_signals: list[str] = Field(
        default_factory=list,
        description="Text signals that led to this classification",
    )
    reasoning: str = Field(default="", description="Explanation of actor determination")


# ── Phase 3 — Evidence mapping models ────────────────────────────────────────

class EvidenceItem(BaseModel):
    """Coverage of a single legal obligation's evidence requirements.

    Produced by evidence_mapper.py — maps document sections to the specific
    artefacts each obligation demands (e.g. 'risk register', 'CE marking').
    """
    obligation_id: str = Field(description="Obligation schema ID from JSONL, e.g. AIACT_ART9_PROVIDER_RISK_MGMT_001")
    article: str = Field(description="Source article, e.g. 'Article 9'")
    requirement: str = Field(description="Plain-language obligation statement")
    evidence_required: list[str] = Field(description="All artefacts the obligation demands")
    satisfied_by_chunks: list[str] = Field(
        default_factory=list,
        description="chunk_ids whose text contains evidence of this artefact",
    )
    satisfied_evidence: list[str] = Field(
        default_factory=list,
        description="evidence_required items found in the document",
    )
    missing_evidence: list[str] = Field(
        default_factory=list,
        description="evidence_required items NOT found in any chunk",
    )
    coverage: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of evidence_required items found: satisfied / total",
    )


class EvidenceMap(BaseModel):
    """Aggregate evidence coverage across all applicable obligations.

    Produced once per audit after the applicability gate confirms high-risk.
    This is the Phase 3 'document section → legal obligation' mapping.
    """
    total_obligations: int
    fully_satisfied: int = Field(description="Obligations with coverage = 1.0")
    partially_satisfied: int = Field(description="Obligations with 0 < coverage < 1.0")
    missing: int = Field(description="Obligations with coverage = 0.0")
    overall_coverage: float = Field(ge=0.0, le=1.0, description="Mean coverage across all obligations")
    items: list[EvidenceItem] = Field(description="Per-obligation evidence detail")


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
    """Result of the Article 5 emotion recognition scan."""
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
    # Phase 3 — Actor + Applicability (legal decision hierarchy)
    actor: ActorClassification | None = Field(
        default=None,
        description="Detected Article 3 actor role (provider / deployer / importer / distributor)",
    )
    applicability: ApplicabilityResult | None = Field(
        default=None,
        description="Article 6 + Annex III gate result — determines which articles apply",
    )
    evidence_map: EvidenceMap | None = Field(
        default=None,
        description="[Phase 3] Document section → obligation evidence mapping (high-risk audits only)",
    )
    # Phase 3 — Production Hardening
    confidence_score: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Overall confidence in the audit results. Low confidence triggers human review.",
    )
    requires_human_review: bool = Field(
        default=False,
        description="True if the audit falls below the confidence threshold or encounters edge cases.",
    )
    model_versions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Active model versions used for this audit. "
            "Keys: bert, actor, risk, prohibited, ner, chunk_classifier. "
            "Values: version strings (e.g. 'v2') or backend identifiers (e.g. 'ollama/phi3:mini')."
        ),
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
