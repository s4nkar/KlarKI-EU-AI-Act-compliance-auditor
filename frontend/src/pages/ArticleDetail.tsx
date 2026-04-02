// Per-article detail: score, gaps, recommendations, and three explainability panels.

import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import Layout from '../components/Layout'
import ScoreRadial from '../components/ScoreRadial'
import GapCard from '../components/GapCard'
import ReportDownload from '../components/ReportDownload'
import { fetchReport } from '../hooks/useReport'
import { ARTICLE_NAMES } from '../utils/formatters'
import type { ArticleScore, ComplianceReport, GapItem, RegulatoryPassage } from '../types'

const SEVERITY_ORDER = { critical: 0, major: 1, minor: 2 }

export default function ArticleDetail() {
  const { auditId, articleNum } = useParams<{ auditId: string; articleNum: string }>()
  const [report, setReport] = useState<ComplianceReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!auditId) return
    fetchReport(auditId)
      .then(setReport)
      .catch(err => setError(err?.response?.data?.error ?? err.message ?? 'Failed to load report.'))
      .finally(() => setLoading(false))
  }, [auditId])

  if (loading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <svg className="w-8 h-8 animate-spin text-brand-400" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
        </div>
      </Layout>
    )
  }

  if (error || !report) {
    return (
      <Layout>
        <div className="max-w-xl mx-auto text-center py-16">
          <p className="text-slate-500 mb-4">{error ?? 'Report not found.'}</p>
          <Link to="/" className="text-brand-600 hover:underline text-sm">← Start a new audit</Link>
        </div>
      </Layout>
    )
  }

  const num = parseInt(articleNum ?? '0', 10)
  const articleScore: ArticleScore | undefined = report.article_scores.find(
    a => a.article_num === num,
  )

  if (!articleScore) {
    return (
      <Layout>
        <div className="max-w-xl mx-auto text-center py-16">
          <p className="text-slate-500 mb-4">Article {num} not found in this report.</p>
          <Link to={`/audit/${auditId}`} className="text-brand-600 hover:underline text-sm">
            ← Back to dashboard
          </Link>
        </div>
      </Layout>
    )
  }

  const name = ARTICLE_NAMES[num] ?? articleScore.domain.replace(/_/g, ' ')
  const sortedGaps = [...articleScore.gaps].sort(
    (a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity],
  )

  return (
    <Layout>
      {/* Breadcrumb */}
      <nav className="mb-4 text-sm text-slate-400 flex items-center gap-1.5">
        <Link to={`/audit/${auditId}`} className="hover:text-brand-600 transition-colors">
          Dashboard
        </Link>
        <span>/</span>
        <span className="text-slate-700 font-medium">Article {num}: {name}</span>
      </nav>

      {/* Header card */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 mb-6 flex flex-col sm:flex-row items-center gap-6">
        <ScoreRadial score={articleScore.score} size={120} />
        <div className="flex-1">
          <p className="text-xs font-semibold text-brand-600 uppercase tracking-wide mb-1">
            Article {num}
          </p>
          <h1 className="text-2xl font-bold text-slate-800 mb-1">{name}</h1>
          <p className="text-sm text-slate-500">
            {articleScore.chunk_count} relevant chunks analysed ·{' '}
            {articleScore.gaps.length} gap{articleScore.gaps.length !== 1 ? 's' : ''} found
          </p>
        </div>
        <ReportDownload auditId={auditId!} />
      </div>

      {/* ── Explainability panel 1: Why this score? ─────────────────────────── */}
      <ScoreReasoningPanel
        reasoning={articleScore.score_reasoning}
        score={articleScore.score}
        gaps={articleScore.gaps}
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
        {/* Gaps column */}
        <div className="lg:col-span-2">
          <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">
            Compliance Gaps
          </h2>
          {sortedGaps.length === 0 ? (
            <div className="bg-green-50 border border-green-200 rounded-xl p-6 text-center">
              <p className="text-green-700 font-semibold">No gaps identified</p>
              <p className="text-green-600 text-sm mt-1">
                This article appears well-covered in your documentation.
              </p>
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              {sortedGaps.map((gap, i) => (
                <GapCard key={i} gap={gap} />
              ))}
            </div>
          )}
        </div>

        {/* Recommendations column */}
        <div>
          <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">
            Recommendations
          </h2>
          {articleScore.recommendations.length === 0 ? (
            <p className="text-slate-400 text-sm">No recommendations.</p>
          ) : (
            <ul className="flex flex-col gap-2">
              {articleScore.recommendations.map((rec, i) => (
                <li key={i} className="flex gap-2 text-sm text-slate-600 bg-white border border-slate-200 rounded-lg p-3">
                  <span className="text-brand-500 font-bold shrink-0 mt-0.5">→</span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* ── Explainability panel 2: Which regulation exactly? ───────────────── */}
      {articleScore.regulatory_passages.length > 0 && (
        <RegulatoryPassagesPanel passages={articleScore.regulatory_passages} articleNum={num} />
      )}

      {/* ── Explainability panel 3: Audit readiness checklist ───────────────── */}
      <AuditReadinessPanel gaps={sortedGaps} articleNum={num} articleName={name} />
    </Layout>
  )
}

// ── Panel 1: Why this score? ──────────────────────────────────────────────────

function ScoreReasoningPanel({
  reasoning,
  score,
  gaps,
}: {
  reasoning: string
  score: number
  gaps: GapItem[]
}) {
  if (!reasoning) return null

  const hasCritical = gaps.some(g => g.severity === 'critical')
  const hasMajor = gaps.some(g => g.severity === 'major')
  const hasMinor = gaps.some(g => g.severity === 'minor')

  // Color is driven by worst gap severity, NOT by the numeric score.
  // A high score with Critical gaps is a contradiction — show it as red.
  const band = hasCritical
    ? { label: 'Critical gaps present', color: 'bg-red-50 border-red-200 text-red-800' }
    : hasMajor
      ? { label: 'Major gaps present', color: 'bg-amber-50 border-amber-200 text-amber-800' }
      : hasMinor
        ? { label: 'Minor gaps only', color: 'bg-blue-50 border-blue-200 text-blue-800' }
        : { label: 'No gaps identified', color: 'bg-green-50 border-green-200 text-green-800' }

  // Detect LLM score/gap contradiction: score ≥ 70 but critical or major gaps present
  const contradiction = score >= 70 && (hasCritical || hasMajor)

  return (
    <div className={`rounded-xl border p-5 ${band.color}`}>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-sm font-semibold uppercase tracking-wide opacity-70">Why this score?</span>
        <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-white/50">{band.label}</span>
      </div>
      <p className="text-sm leading-relaxed">{reasoning}</p>
      {contradiction && (
        <p className="mt-3 text-xs opacity-80 border-t border-current/20 pt-2">
          Note: The AI assigned a high numeric score ({score}/100) while also identifying critical or major gaps.
          This inconsistency is a known limitation of LLM-based scoring — treat the gaps list as the authoritative signal, not the number.
        </p>
      )}
    </div>
  )
}

// ── Panel 2: Which regulation exactly? ───────────────────────────────────────

function RegulatoryPassagesPanel({
  passages,
  articleNum,
}: {
  passages: RegulatoryPassage[]
  articleNum: number
}) {
  const [open, setOpen] = useState(false)

  return (
    <div className="mt-6 bg-white border border-slate-200 rounded-2xl overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-slate-50 transition-colors"
      >
        <div>
          <p className="text-sm font-semibold text-slate-700">Which regulation exactly?</p>
          <p className="text-xs text-slate-400 mt-0.5">
            {passages.length} regulatory passage{passages.length !== 1 ? 's' : ''} used in the Article {articleNum} gap analysis
          </p>
        </div>
        <svg
          className={`w-5 h-5 text-slate-400 transition-transform ${open ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="border-t border-slate-100 divide-y divide-slate-100">
          {passages.map((p, i) => (
            <div key={i} className="px-5 py-4">
              <div className="flex flex-wrap items-center gap-2 mb-2">
                {p.title && (
                  <span className="text-xs font-semibold text-brand-700 bg-brand-50 px-2 py-0.5 rounded">
                    {p.title}
                  </span>
                )}
                {p.article_ref && (
                  <span className="text-xs text-slate-500">{p.article_ref}</span>
                )}
                {p.regulation && (
                  <span className="text-xs text-slate-400 uppercase tracking-wide">
                    {p.regulation.replace(/_/g, ' ')}
                  </span>
                )}
              </div>
              <p className="text-sm text-slate-600 leading-relaxed">{p.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Panel 3: Audit readiness checklist ───────────────────────────────────────

function AuditReadinessPanel({
  gaps,
  articleNum,
  articleName,
}: {
  gaps: GapItem[]
  articleNum: number
  articleName: string
}) {
  const critical = gaps.filter(g => g.severity === 'critical')
  const major = gaps.filter(g => g.severity === 'major')
  const minor = gaps.filter(g => g.severity === 'minor')

  const canDefend = critical.length === 0 && major.length === 0
  const borderColor = canDefend ? 'border-green-200' : critical.length > 0 ? 'border-red-200' : 'border-amber-200'
  const headerColor = canDefend ? 'bg-green-50' : critical.length > 0 ? 'bg-red-50' : 'bg-amber-50'
  const textColor = canDefend ? 'text-green-800' : critical.length > 0 ? 'text-red-800' : 'text-amber-800'

  return (
    <div className={`mt-6 rounded-2xl border ${borderColor} overflow-hidden`}>
      <div className={`px-5 py-4 ${headerColor}`}>
        <p className={`text-sm font-semibold ${textColor}`}>Can I defend this in an audit?</p>
        <p className={`text-xs mt-0.5 ${textColor} opacity-80`}>
          Article {articleNum} — {articleName}
        </p>
      </div>

      <div className="bg-white px-5 py-4">
        {/* Verdict */}
        <div className="flex items-start gap-3 mb-4 pb-4 border-b border-slate-100">
          <span className="text-2xl mt-0.5">{canDefend ? '✓' : critical.length > 0 ? '✗' : '⚠'}</span>
          <p className="text-sm text-slate-700 leading-relaxed">
            {canDefend
              ? `No critical or major gaps found. This article is likely defensible in a regulatory audit, provided the documentation accurately reflects your system's actual operation.`
              : critical.length > 0
                ? `${critical.length} critical gap${critical.length > 1 ? 's' : ''} must be resolved before this article can be defended. Critical gaps represent direct non-compliance with mandatory requirements.`
                : `No critical gaps, but ${major.length} major gap${major.length > 1 ? 's' : ''} remain. These should be addressed before submitting to a conformity assessment body.`
            }
          </p>
        </div>

        {/* Checklist */}
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">Remediation checklist</p>
        <div className="flex flex-col gap-2">
          {critical.length === 0 && major.length === 0 && minor.length === 0 && (
            <p className="text-sm text-green-600">All items resolved — no gaps to remediate.</p>
          )}
          {critical.map((g, i) => (
            <ChecklistItem key={`c${i}`} gap={g} />
          ))}
          {major.map((g, i) => (
            <ChecklistItem key={`m${i}`} gap={g} />
          ))}
          {minor.map((g, i) => (
            <ChecklistItem key={`n${i}`} gap={g} />
          ))}
        </div>

        <p className="text-xs text-slate-400 mt-4 pt-3 border-t border-slate-100">
          This checklist is generated from the automated gap analysis. Final audit readiness must be verified by a qualified legal or compliance professional.
        </p>
      </div>
    </div>
  )
}

function ChecklistItem({ gap }: { gap: GapItem }) {
  const [checked, setChecked] = useState(false)
  const severityColors = {
    critical: 'text-red-600 bg-red-50 border-red-200',
    major: 'text-amber-700 bg-amber-50 border-amber-200',
    minor: 'text-slate-600 bg-slate-50 border-slate-200',
  }
  return (
    <label className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-opacity ${severityColors[gap.severity]} ${checked ? 'opacity-50' : ''}`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={e => setChecked(e.target.checked)}
        className="mt-0.5 shrink-0 accent-brand-600"
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-xs font-bold uppercase tracking-wide opacity-70">{gap.severity}</span>
          <span className="text-sm font-medium">{gap.title}</span>
        </div>
        <p className="text-xs opacity-80 leading-snug">{gap.description}</p>
      </div>
    </label>
  )
}
