// Utility formatters for scores, severity labels, and dates.
// Used by ScoreRadial, GapCard, and ArticleCard components (Phase 3).

import type { RiskTier, Severity } from '../types'

/** Return Tailwind color class for a compliance score 0–100. */
export function scoreColor(score: number): string {
  if (score >= 70) return 'text-green-600'
  if (score >= 40) return 'text-amber-500'
  return 'text-red-600'
}

/** Return Tailwind background color class for a compliance score. */
export function scoreBgColor(score: number): string {
  if (score >= 70) return 'bg-green-100 text-green-800'
  if (score >= 40) return 'bg-amber-100 text-amber-800'
  return 'bg-red-100 text-red-800'
}

/** Human-readable severity label with color. */
export function severityLabel(severity: Severity): { label: string; className: string } {
  switch (severity) {
    case 'critical': return { label: 'Critical', className: 'bg-red-100 text-red-800' }
    case 'major':    return { label: 'Major',    className: 'bg-amber-100 text-amber-800' }
    case 'minor':    return { label: 'Minor',    className: 'bg-blue-100 text-blue-800' }
  }
}

/** Human-readable risk tier label and badge color. */
export function riskTierLabel(tier: RiskTier): { label: string; className: string } {
  switch (tier) {
    case 'prohibited': return { label: 'Prohibited',  className: 'bg-red-100 text-red-900' }
    case 'high':       return { label: 'High Risk',   className: 'bg-amber-100 text-amber-900' }
    case 'limited':    return { label: 'Limited Risk', className: 'bg-blue-100 text-blue-900' }
    case 'minimal':    return { label: 'Minimal Risk', className: 'bg-green-100 text-green-900' }
  }
}

/** Format ISO date string to localised human-readable date. */
export function formatDate(isoString: string): string {
  return new Date(isoString).toLocaleDateString('en-GB', {
    day: 'numeric', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

/** Article number to domain name map for display. */
export const ARTICLE_NAMES: Record<number, string> = {
  9:  'Risk Management',
  10: 'Data Governance',
  11: 'Technical Documentation',
  12: 'Record-Keeping',
  13: 'Transparency',
  14: 'Human Oversight',
  15: 'Accuracy & Security',
}
