// Premium per-article detail view: score, gaps, recommendations, and three explainability panels.

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
  const [report, setReport]   = useState<ComplianceReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState<string | null>(null)

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
        <div className="flex flex-col items-center justify-center h-64 gap-3">
          <svg className="w-8 h-8 animate-spin text-brand-500" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
          <p className="text-sm text-slate-400">Loading article detail…</p>
        </div>
      </Layout>
    )
  }

  if (error || !report) {
    return (
      <Layout>
        <div className="max-w-xl mx-auto text-center py-20">
          <p className="text-slate-500 mb-5">{error ?? 'Report not found.'}</p>
          <Link to="/" className="btn-primary">← Start a new audit</Link>
        </div>
      </Layout>
    )
  }

  const num = parseInt(articleNum ?? '0', 10)
  const articleScore: ArticleScore | undefined = report.article_scores.find(a => a.article_num === num)

  if (!articleScore) {
    return (
      <Layout>
        <div className="max-w-xl mx-auto text-center py-20">
          <p className="text-slate-500 mb-5">Article {num} not found in this report.</p>
          <Link to={`/audit/${auditId}`} className="btn-secondary">← Back to dashboard</Link>
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
      <nav className="mb-5 flex items-center gap-1.5 text-sm">
        <Link to={`/audit/${auditId}`} className="text-slate-400 hover:text-brand-600 transition-colors flex items-center gap-1">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
          </svg>
          Dashboard
        </Link>
        <svg className="w-4 h-4 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        <span className="text-slate-700 font-medium">Article {num}: {name}</span>
      </nav>

      {/* Header card */}
      <div className="card p-6 mb-6">
        <div className="flex flex-col sm:flex-row items-center gap-6">
          <ScoreRadial score={articleScore.score} size={120} />
          <div className="flex-1 text-center sm:text-left">
            <p className="text-xs font-bold text-brand-600 uppercase tracking-widest mb-1">
              EU AI Act — Article {num}
            </p>
            <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight mb-2">{name}</h1>
            <p className="text-sm text-slate-500">
              {articleScore.chunk_count} relevant chunks analysed
              {' · '}
              <span className={articleScore.gaps.length > 0 ? 'text-red-600 font-semibold' : 'text-emerald-600 font-semibold'}>
                {articleScore.gaps.length} gap{articleScore.gaps.length !== 1 ? 's' : ''} found
              </span>
            </p>
          </div>
          <ReportDownload auditId={auditId!} />
        </div>
      </div>

      {/* Explainability panel 1: Why this score? */}
      <ScoreReasoningPanel reasoning={articleScore.score_reasoning} score={articleScore.score} gaps={articleScore.gaps} />

      {/* Gaps + Recommendations */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
        {/* Gaps */}
        <div className="lg:col-span-2">
          <h2 className="section-label">Compliance Gaps</h2>
          {sortedGaps.length === 0 ? (
            <div className="card border-emerald-200 bg-emerald-50 p-8 text-center">
              <svg className="w-10 h-10 text-emerald-400 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-emerald-700 font-semibold">No gaps identified</p>
              <p className="text-emerald-600 text-sm mt-1">
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

        {/* Recommendations */}
        <div>
          <h2 className="section-label">Recommendations</h2>
          {articleScore.recommendations.length === 0 ? (
            <p className="text-slate-400 text-sm">No recommendations.</p>
          ) : (
            <ul className="flex flex-col gap-2">
              {articleScore.recommendations.map((rec, i) => (
                <li key={i} className="flex gap-3 text-sm bg-white border border-slate-200 rounded-xl p-3.5 hover:border-brand-200 hover:bg-brand-50 transition-colors">
                  <svg className="w-4 h-4 text-brand-500 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                  <span className="text-slate-700 leading-snug">{rec}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* Explainability panel 2: Which regulation? */}
      {articleScore.regulatory_passages.length > 0 && (
        <RegulatoryPassagesPanel passages={articleScore.regulatory_passages} articleNum={num} />
      )}

      {/* Explainability panel 3: Audit readiness checklist */}
      <AuditReadinessPanel gaps={sortedGaps} articleNum={num} articleName={name} />
    </Layout>
  )
}

// ── Panel 1: Why this score? ─────────────────────────────────────────────────

function ScoreReasoningPanel({ reasoning, score, gaps }: { reasoning: string; score: number; gaps: GapItem[] }) {
  if (!reasoning) return null

  const hasCritical = gaps.some(g => g.severity === 'critical')
  const hasMajor    = gaps.some(g => g.severity === 'major')
  const hasMinor    = gaps.some(g => g.severity === 'minor')
  const contradiction = score >= 70 && (hasCritical || hasMajor)

  const band = hasCritical
    ? { label: 'Critical gaps present', bg: 'bg-red-50',   border: 'border-red-200',   text: 'text-red-800',   badge: 'bg-red-100 text-red-700' }
    : hasMajor
      ? { label: 'Major gaps present',  bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-800', badge: 'bg-amber-100 text-amber-700' }
      : hasMinor
        ? { label: 'Minor gaps only',   bg: 'bg-blue-50',  border: 'border-blue-200',  text: 'text-blue-800',  badge: 'bg-blue-100 text-blue-700' }
        : { label: 'No gaps',           bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-800', badge: 'bg-emerald-100 text-emerald-700' }

  return (
    <div className={`rounded-xl border p-5 ${band.bg} ${band.border}`}>
      <div className="flex items-center gap-2.5 mb-3">
        <p className={`text-sm font-bold uppercase tracking-widest ${band.text} opacity-70`}>Why this score?</p>
        <span className={`badge ${band.badge}`}>{band.label}</span>
      </div>
      <p className={`text-sm leading-relaxed ${band.text}`}>{reasoning}</p>
      {contradiction && (
        <div className={`mt-3 pt-3 border-t ${band.border} text-xs ${band.text} opacity-75`}>
          Note: The AI assigned {score}/100 while identifying critical or major gaps. Treat the gaps list as the authoritative signal — LLM-based numeric scores can be optimistic.
        </div>
      )}
    </div>
  )
}

// ── Panel 2: Which regulation? ───────────────────────────────────────────────

function RegulatoryPassagesPanel({ passages, articleNum }: { passages: RegulatoryPassage[]; articleNum: number }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="mt-6 card overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-slate-50 transition-colors"
      >
        <div>
          <p className="text-sm font-semibold text-slate-800">Which regulation exactly?</p>
          <p className="text-xs text-slate-400 mt-0.5">
            {passages.length} regulatory passage{passages.length !== 1 ? 's' : ''} used in the Article {articleNum} gap analysis
          </p>
        </div>
        <svg
          className={`w-5 h-5 text-slate-400 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
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
                  <span className="badge bg-brand-100 text-brand-700 border border-brand-200">{p.title}</span>
                )}
                {p.article_ref && (
                  <span className="text-xs text-slate-500">{p.article_ref}</span>
                )}
                {p.regulation && (
                  <span className="badge bg-slate-100 text-slate-600">{p.regulation.replace(/_/g, ' ').toUpperCase()}</span>
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

// ── Panel 3: Audit readiness checklist ──────────────────────────────────────

function AuditReadinessPanel({ gaps, articleNum, articleName }: { gaps: GapItem[]; articleNum: number; articleName: string }) {
  const critical = gaps.filter(g => g.severity === 'critical')
  const major    = gaps.filter(g => g.severity === 'major')
  const minor    = gaps.filter(g => g.severity === 'minor')
  const canDefend = critical.length === 0 && major.length === 0

  const cfg = canDefend
    ? { border: 'border-emerald-200', header: 'bg-emerald-50', text: 'text-emerald-800' }
    : critical.length > 0
      ? { border: 'border-red-200', header: 'bg-red-50', text: 'text-red-800' }
      : { border: 'border-amber-200', header: 'bg-amber-50', text: 'text-amber-800' }

  return (
    <div className={`mt-6 rounded-2xl border overflow-hidden ${cfg.border}`}>
      <div className={`px-5 py-4 ${cfg.header} flex items-center gap-3`}>
        {canDefend ? (
          <svg className="w-5 h-5 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
        ) : (
          <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
        )}
        <div>
          <p className={`text-sm font-bold ${cfg.text}`}>Can I defend this in an audit?</p>
          <p className={`text-xs ${cfg.text} opacity-70`}>Article {articleNum} — {articleName}</p>
        </div>
      </div>

      <div className="bg-white px-5 py-5">
        {/* Verdict */}
        <div className="flex items-start gap-3 mb-5 pb-5 border-b border-slate-100">
          <div className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 mt-0.5 text-white text-base ${
            canDefend ? 'bg-emerald-500' : critical.length > 0 ? 'bg-red-500' : 'bg-amber-500'
          }`}>
            {canDefend ? '✓' : critical.length > 0 ? '✗' : '!'}
          </div>
          <p className="text-sm text-slate-700 leading-relaxed">
            {canDefend
              ? 'No critical or major gaps found. This article is likely defensible in a regulatory audit, provided the documentation accurately reflects your system\'s actual operation.'
              : critical.length > 0
                ? `${critical.length} critical gap${critical.length > 1 ? 's' : ''} must be resolved before this article can be defended. Critical gaps represent direct non-compliance with mandatory requirements.`
                : `No critical gaps, but ${major.length} major gap${major.length > 1 ? 's' : ''} remain. Address these before submitting to a conformity assessment body.`
            }
          </p>
        </div>

        {/* Checklist */}
        <p className="section-label">Remediation checklist</p>
        <div className="flex flex-col gap-2">
          {critical.length === 0 && major.length === 0 && minor.length === 0 && (
            <p className="text-sm text-emerald-600 flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              All items resolved — no gaps to remediate.
            </p>
          )}
          {[...critical, ...major, ...minor].map((g, i) => (
            <ChecklistItem key={i} gap={g} />
          ))}
        </div>

        <p className="text-xs text-slate-400 mt-5 pt-4 border-t border-slate-100">
          This checklist is generated from the automated gap analysis. Final audit readiness must be verified by a qualified legal or compliance professional.
        </p>
      </div>
    </div>
  )
}

function ChecklistItem({ gap }: { gap: GapItem }) {
  const [checked, setChecked] = useState(false)

  const cfg = {
    critical: 'border-red-200 bg-red-50 text-red-700',
    major:    'border-amber-200 bg-amber-50 text-amber-700',
    minor:    'border-slate-200 bg-slate-50 text-slate-600',
  }[gap.severity]

  return (
    <label className={`flex items-start gap-3 p-3.5 rounded-xl border cursor-pointer transition-opacity ${cfg} ${checked ? 'opacity-40' : ''}`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={e => setChecked(e.target.checked)}
        className="mt-0.5 shrink-0 w-4 h-4 accent-brand-600"
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5 flex-wrap">
          <span className="badge bg-white/60 border border-current/20 uppercase text-[10px]">{gap.severity}</span>
          <span className="text-sm font-semibold">{gap.title}</span>
        </div>
        <p className="text-xs opacity-80 leading-snug">{gap.description}</p>
      </div>
    </label>
  )
}
