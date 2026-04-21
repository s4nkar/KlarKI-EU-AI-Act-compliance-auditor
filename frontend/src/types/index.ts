// TypeScript interfaces matching api/models/schemas.py Pydantic models exactly.
// Keep in sync with backend schemas.

export type ArticleDomain =
  | 'risk_management'
  | 'data_governance'
  | 'technical_documentation'
  | 'record_keeping'
  | 'transparency'
  | 'human_oversight'
  | 'security'
  | 'unrelated'

export type RiskTier = 'prohibited' | 'high' | 'limited' | 'minimal'

export type Severity = 'critical' | 'major' | 'minor'

export type AuditStatus =
  | 'uploading'
  | 'parsing'
  | 'classifying'
  | 'analysing'
  | 'scoring'
  | 'complete'
  | 'failed'

// ── Phase 3 — Actor & Applicability types ─────────────────────────────────────

export type ActorType = 'provider' | 'deployer' | 'importer' | 'distributor' | 'unknown'

export interface ActorClassification {
  actor_type: ActorType
  confidence: number
  matched_signals: string[]
  reasoning: string
}

export interface AnnexIIIMatch {
  category: number          // AnnexIIICategory int value 1–8
  category_name: string
  matched_keywords: string[]
  obligation_id: string
  evidence_required: string[]
}

export interface ApplicabilityResult {
  is_high_risk: boolean
  is_prohibited: boolean
  annex_iii_matches: AnnexIIIMatch[]
  annex_i_triggered: boolean
  applicable_articles: number[]
  reasoning: string
}

// ── Phase 3 — Evidence mapping types ─────────────────────────────────────────

export interface EvidenceItem {
  obligation_id: string
  article: string
  requirement: string
  evidence_required: string[]
  satisfied_by_chunks: string[]
  satisfied_evidence: string[]
  missing_evidence: string[]
  coverage: number
}

export interface EvidenceMap {
  total_obligations: number
  fully_satisfied: number
  partially_satisfied: number
  missing: number
  overall_coverage: number
  items: EvidenceItem[]
}

// ── Core report types ─────────────────────────────────────────────────────────

export interface GapItem {
  title: string
  description: string
  severity: Severity
  article_num: number
}

export interface RegulatoryPassage {
  title: string
  text: string
  regulation: string
  article_ref: string
}

export interface ArticleScore {
  article_num: number
  domain: ArticleDomain
  score: number
  gaps: GapItem[]
  recommendations: string[]
  chunk_count: number
  score_reasoning: string
  regulatory_passages: RegulatoryPassage[]
}

export interface EmotionFlag {
  detected: boolean
  context: string
  is_prohibited: boolean
  explanation: string
}

export interface ComplianceReport {
  audit_id: string
  created_at: string
  source_files: string[]
  language: string
  risk_tier: RiskTier
  wizard_risk_tier: RiskTier | null
  overall_score: number
  article_scores: ArticleScore[]
  emotion_flag: EmotionFlag
  total_chunks: number
  classified_chunks: number
  classifier_backend: string
  // Phase 3 fields
  actor: ActorClassification | null
  applicability: ApplicabilityResult | null
  evidence_map: EvidenceMap | null
  confidence_score: number
  requires_human_review: boolean
}

export interface AuditResponse {
  audit_id: string
  status: AuditStatus
  report: ComplianceReport | null
}

export interface APIResponse<T = Record<string, unknown>> {
  status: 'success' | 'error'
  data: T | null
  error: string | null
}
