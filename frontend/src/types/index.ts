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
