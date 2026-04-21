// Premium compliance dashboard: overall score, risk tier, article grid, PDF download.

import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import Layout from '../components/Layout'
import ScoreRadial from '../components/ScoreRadial'
import ArticleCard from '../components/ArticleCard'
import ReportDownload from '../components/ReportDownload'
import EmotionWarning from '../components/EmotionWarning'
import { fetchReport } from '../hooks/useReport'
import { riskTierLabel, formatDate } from '../utils/formatters'
import type { ActorClassification, ApplicabilityResult, ComplianceReport, EvidenceMap, RiskTier } from '../types'

export default function Dashboard() {
  const { auditId } = useParams<{ auditId: string }>()
  const [report, setReport] = useState<ComplianceReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]   = useState<string | null>(null)

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
          <p className="text-sm text-slate-400">Loading compliance report…</p>
        </div>
      </Layout>
    )
  }

  if (error || !report) {
    return (
      <Layout>
        <div className="max-w-xl mx-auto text-center py-20">
          <div className="w-14 h-14 rounded-2xl bg-red-50 flex items-center justify-center mx-auto mb-4">
            <svg className="w-7 h-7 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <p className="text-slate-600 mb-5">{error ?? 'Report not found.'}</p>
          <Link to="/" className="btn-primary">← Start a new audit</Link>
        </div>
      </Layout>
    )
  }

  const { label: tierLabel, className: tierClass } = riskTierLabel(report.risk_tier)
  const sorted = [...report.article_scores].sort((a, b) => a.article_num - b.article_num)

  const totalGaps    = sorted.reduce((s, a) => s + a.gaps.length, 0)
  const criticalGaps = sorted.reduce((s, a) => s + a.gaps.filter(g => g.severity === 'critical').length, 0)
  const majorGaps    = sorted.reduce((s, a) => s + a.gaps.filter(g => g.severity === 'major').length, 0)

  return (
    <Layout>
      <EmotionWarning flag={report.emotion_flag} />

      {/* ── Human Review Warning ──────────────────────────────────────────── */}
      {report.requires_human_review && (
        <div className="mb-6 rounded-xl border border-amber-300 bg-amber-50 p-4">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div>
              <p className="text-sm font-bold text-amber-800">Manual Human Review Required</p>
              <p className="text-sm text-amber-700 mt-0.5">
                Audit confidence is {Math.round((report.confidence_score ?? 1) * 100)}%. A compliance expert should verify these findings before relying on them for regulatory decisions.
              </p>
            </div>
          </div>
        </div>
      )}

      {report.wizard_risk_tier && (
        <RiskTierComparison
          wizardTier={report.wizard_risk_tier}
          documentTier={report.risk_tier}
        />
      )}

      {/* ── Hero section ─────────────────────────────────────────────────── */}
      <div className="card p-6 mb-6">
        <div className="flex flex-col sm:flex-row items-center gap-6">
          <ScoreRadial score={report.overall_score} size={140} label="Overall Score" />

          <div className="flex-1 min-w-0 text-center sm:text-left">
            <div className="flex flex-wrap items-center gap-2 mb-1 justify-center sm:justify-start">
              <h1 className="text-xl font-bold text-slate-900">Compliance Report</h1>
              <span className={`badge ${tierClass}`}>{tierLabel}</span>
            </div>
            <p className="text-sm text-slate-400 mb-4">{formatDate(report.created_at)}</p>

            {/* Stats row */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
              <StatBox label="Total Chunks"    value={report.total_chunks} />
              <StatBox label="Classified"      value={report.classified_chunks} />
              <StatBox label="Language"        value={report.language.toUpperCase()} />
              <StatBox label="Source Files"    value={report.source_files.length} />
              <StatBox label="Classifier"      value={report.classifier_backend} mono />
              <StatBox label="Confidence"      value={`${Math.round((report.confidence_score ?? 1) * 100)}%`} />
            </div>
          </div>

          <div className="shrink-0">
            <ReportDownload auditId={report.audit_id} />
          </div>
        </div>
      </div>

      {/* ── Gap summary bar ───────────────────────────────────────────────── */}
      {totalGaps > 0 && (
        <div className="flex flex-wrap gap-3 mb-6">
          {criticalGaps > 0 && (
            <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-red-50 border border-red-200">
              <span className="w-2 h-2 rounded-full bg-red-500 shrink-0" />
              <span className="text-sm font-semibold text-red-700">{criticalGaps} Critical Gap{criticalGaps !== 1 ? 's' : ''}</span>
            </div>
          )}
          {majorGaps > 0 && (
            <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-amber-50 border border-amber-200">
              <span className="w-2 h-2 rounded-full bg-amber-500 shrink-0" />
              <span className="text-sm font-semibold text-amber-700">{majorGaps} Major Gap{majorGaps !== 1 ? 's' : ''}</span>
            </div>
          )}
          <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-100 border border-slate-200">
            <span className="text-sm text-slate-600">{totalGaps} total gaps across {sorted.length} articles</span>
          </div>
        </div>
      )}

      {/* ── Phase 3 panels ───────────────────────────────────────────────── */}
      {(report.actor || report.applicability || report.evidence_map) && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {report.actor && <ActorPanel actor={report.actor} />}
          {report.applicability && <ApplicabilityPanel applicability={report.applicability} />}
          {report.evidence_map && <EvidencePanel evidence={report.evidence_map} />}
        </div>
      )}

      {/* ── Article grid ─────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="section-label mb-0">Article Scores — EU AI Act Art. 9–15</h2>
        <span className="text-xs text-slate-400">{sorted.length} articles analysed</span>
      </div>

      {sorted.length === 0 ? (
        <div className="card p-12 text-center text-slate-400">
          No article scores found in this report.
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {sorted.map(article => (
            <ArticleCard key={article.article_num} score={article} auditId={report.audit_id} />
          ))}
        </div>
      )}
    </Layout>
  )
}

const ACTOR_LABELS: Record<string, string> = {
  provider: 'Provider', deployer: 'Deployer', importer: 'Importer',
  distributor: 'Distributor', unknown: 'Unknown',
}

function ActorPanel({ actor }: { actor: ActorClassification }) {
  const confidencePct = Math.round(actor.confidence * 100)
  return (
    <div className="card p-4">
      <p className="text-xs font-bold text-slate-400 uppercase tracking-wide mb-2">Article 3 — Actor Role</p>
      <p className="text-base font-bold text-slate-900 mb-1">{ACTOR_LABELS[actor.actor_type] ?? actor.actor_type}</p>
      <div className="flex items-center gap-2 mb-2">
        <div className="flex-1 h-1.5 rounded-full bg-slate-100">
          <div className="h-1.5 rounded-full bg-blue-500" style={{ width: `${confidencePct}%` }} />
        </div>
        <span className="text-xs text-slate-500 shrink-0">{confidencePct}%</span>
      </div>
      {actor.reasoning && <p className="text-xs text-slate-500 leading-relaxed">{actor.reasoning}</p>}
    </div>
  )
}

function ApplicabilityPanel({ applicability }: { applicability: ApplicabilityResult }) {
  const categoryNames = applicability.annex_iii_matches.map(m => m.category_name)
  return (
    <div className="card p-4">
      <p className="text-xs font-bold text-slate-400 uppercase tracking-wide mb-2">Article 6 — Applicability Gate</p>
      {applicability.is_prohibited ? (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold bg-red-100 text-red-700 mb-2">Prohibited Use</span>
      ) : applicability.is_high_risk ? (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold bg-amber-100 text-amber-700 mb-2">High-Risk — Art. 9–15 Apply</span>
      ) : (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700 mb-2">Minimal Risk</span>
      )}
      {categoryNames.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-2">
          {categoryNames.map(name => (
            <span key={name} className="px-1.5 py-0.5 rounded bg-slate-100 text-xs text-slate-600">{name}</span>
          ))}
        </div>
      )}
      {applicability.reasoning && <p className="text-xs text-slate-500 leading-relaxed">{applicability.reasoning}</p>}
    </div>
  )
}

function EvidencePanel({ evidence }: { evidence: EvidenceMap }) {
  const coveragePct = Math.round(evidence.overall_coverage * 100)
  return (
    <div className="card p-4">
      <p className="text-xs font-bold text-slate-400 uppercase tracking-wide mb-2">Evidence Coverage</p>
      <p className="text-2xl font-bold text-slate-900 mb-1">{coveragePct}%</p>
      <div className="h-1.5 rounded-full bg-slate-100 mb-2">
        <div
          className={`h-1.5 rounded-full ${coveragePct >= 70 ? 'bg-emerald-500' : coveragePct >= 40 ? 'bg-amber-400' : 'bg-red-500'}`}
          style={{ width: `${coveragePct}%` }}
        />
      </div>
      <div className="grid grid-cols-3 gap-1 text-center">
        <div className="rounded bg-emerald-50 px-1 py-1">
          <p className="text-xs font-bold text-emerald-700">{evidence.fully_satisfied}</p>
          <p className="text-[10px] text-emerald-600">Satisfied</p>
        </div>
        <div className="rounded bg-amber-50 px-1 py-1">
          <p className="text-xs font-bold text-amber-700">{evidence.partially_satisfied}</p>
          <p className="text-[10px] text-amber-600">Partial</p>
        </div>
        <div className="rounded bg-red-50 px-1 py-1">
          <p className="text-xs font-bold text-red-700">{evidence.missing}</p>
          <p className="text-[10px] text-red-600">Missing</p>
        </div>
      </div>
    </div>
  )
}

function StatBox({ label, value, mono }: { label: string; value: string | number; mono?: boolean }) {
  return (
    <div className="rounded-xl bg-slate-50 border border-slate-100 px-3 py-2.5">
      <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide mb-0.5">{label}</p>
      <p className={`text-sm font-bold text-slate-800 truncate ${mono ? 'font-mono' : ''}`}>{value}</p>
    </div>
  )
}

const TIER_RANK: Record<RiskTier, number> = { prohibited: 3, high: 2, limited: 1, minimal: 0 }

function RiskTierComparison({ wizardTier, documentTier }: { wizardTier: RiskTier; documentTier: RiskTier }) {
  const { label: wizLabel, className: wizClass } = riskTierLabel(wizardTier)
  const { label: docLabel, className: docClass } = riskTierLabel(documentTier)
  const wizRank = TIER_RANK[wizardTier]
  const docRank = TIER_RANK[documentTier]

  let message: string
  let cfg: { bg: string; border: string; text: string; icon: React.ReactNode }

  if (wizardTier === documentTier) {
    message = 'The document-derived risk tier matches your self-assessment. Your audit is internally consistent.'
    cfg = {
      bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-800',
      icon: <svg className="w-4 h-4 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
    }
  } else if (docRank > wizRank) {
    message = 'The document audit found a higher risk tier than your self-assessment. The document contains keywords associated with higher-risk AI use cases. Review the flagged areas carefully before deployment.'
    cfg = {
      bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-800',
      icon: <svg className="w-4 h-4 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>,
    }
  } else {
    message = `The document audit derived a lower risk tier than your self-assessment. The wizard result (${riskTierLabel(wizardTier).label}) remains the primary signal — a qualified assessor should confirm the final tier.`
    cfg = {
      bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-800',
      icon: <svg className="w-4 h-4 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
    }
  }

  return (
    <div className={`mb-6 rounded-xl border p-4 ${cfg.bg} ${cfg.border}`}>
      <div className="flex items-center gap-2 mb-2.5">
        {cfg.icon}
        <p className={`text-xs font-bold uppercase tracking-widest ${cfg.text} opacity-70`}>
          Risk Tier Comparison
        </p>
        <span className={`text-xs ml-auto italic ${cfg.text} opacity-50`}>
          Wizard result is informational only
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-3 mb-2">
        <div className="flex items-center gap-2 text-sm">
          <span className={`opacity-70 ${cfg.text}`}>Self-assessment:</span>
          <span className={`badge ${wizClass}`}>{wizLabel}</span>
        </div>
        <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
        </svg>
        <div className="flex items-center gap-2 text-sm">
          <span className={`opacity-70 ${cfg.text}`}>Document audit:</span>
          <span className={`badge ${docClass}`}>{docLabel}</span>
        </div>
      </div>
      <p className={`text-sm ${cfg.text}`}>{message}</p>
    </div>
  )
}
