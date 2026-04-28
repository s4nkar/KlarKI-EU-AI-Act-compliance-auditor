import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import Layout from '../components/Layout'
import ScoreRadial from '../components/ScoreRadial'
import ArticleCard from '../components/ArticleCard'
import ReportDownload from '../components/ReportDownload'
import EmotionWarning from '../components/EmotionWarning'
import { fetchReport } from '../hooks/useReport'
import { riskTierLabel, formatDate } from '../utils/formatters'
import type {
  ActorClassification, ApplicabilityResult, ArticleScore,
  ComplianceReport, EvidenceItem, EvidenceMap, RiskTier,
} from '../types'

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

  const criticalGaps = sorted.reduce((s, a) => s + a.gaps.filter(g => g.severity === 'critical').length, 0)
  const majorGaps    = sorted.reduce((s, a) => s + a.gaps.filter(g => g.severity === 'major').length, 0)
  const minorGaps    = sorted.reduce((s, a) => s + a.gaps.filter(g => g.severity === 'minor').length, 0)
  const totalGaps    = criticalGaps + majorGaps + minorGaps

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
        <RiskTierComparison wizardTier={report.wizard_risk_tier} documentTier={report.risk_tier} />
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

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
              <StatBox label="Total Chunks"  value={report.total_chunks} />
              <StatBox label="Classified"    value={report.classified_chunks} />
              <StatBox label="Language"      value={report.language.toUpperCase()} />
              <StatBox label="Confidence"    value={`${Math.round((report.confidence_score ?? 1) * 100)}%`} />
              <StatBox label="Classifier"    value={report.classifier_backend} mono />
              <StatBox label="Source Files"  value={report.source_files.length} />
            </div>

            {/* Source file names */}
            {report.source_files.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {report.source_files.map((f, i) => (
                  <span key={i} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-100 text-[10px] font-mono text-slate-500 border border-slate-200">
                    <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    {f}
                  </span>
                ))}
              </div>
            )}
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
          {minorGaps > 0 && (
            <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-blue-50 border border-blue-200">
              <span className="w-2 h-2 rounded-full bg-blue-400 shrink-0" />
              <span className="text-sm font-semibold text-blue-700">{minorGaps} Minor Gap{minorGaps !== 1 ? 's' : ''}</span>
            </div>
          )}
          <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-100 border border-slate-200">
            <span className="text-sm text-slate-600">{totalGaps} total gaps across {sorted.length} articles</span>
          </div>
        </div>
      )}

      {/* ── Priority action items ─────────────────────────────────────────── */}
      <PrioritiesSection articleScores={sorted} auditId={report.audit_id} />

      {/* ── Phase 3 panels ───────────────────────────────────────────────── */}
      {(report.actor || report.applicability || report.evidence_map) && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {report.actor        && <ActorPanel actor={report.actor} />}
          {report.applicability && <ApplicabilityPanel applicability={report.applicability} />}
          {report.evidence_map  && <EvidenceSummaryPanel evidence={report.evidence_map} />}
        </div>
      )}

      {/* ── Legal obligation coverage drill-down ─────────────────────────── */}
      {report.evidence_map && report.evidence_map.items.length > 0 && (
        <EvidenceObligationsSection evidence={report.evidence_map} />
      )}

      {/* ── Article grid ─────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="section-label mb-0">Article Scores — EU AI Act Art. 9–15</h2>
        <span className="text-xs text-slate-400">{sorted.length} articles analysed</span>
      </div>
      {sorted.length === 0 ? (
        <div className="card p-12 text-center text-slate-400">No article scores found in this report.</div>
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

// ── Priority action items ────────────────────────────────────────────────────

function PrioritiesSection({ articleScores, auditId }: { articleScores: ArticleScore[]; auditId: string }) {
  const allCritical = articleScores.flatMap(a =>
    a.gaps.filter(g => g.severity === 'critical').map(g => ({ ...g, article_num: a.article_num }))
  )
  const allMajor = articleScores.flatMap(a =>
    a.gaps.filter(g => g.severity === 'major').map(g => ({ ...g, article_num: a.article_num }))
  )

  // Show all critical + up to 3 major (or more major if no critical), cap at 8
  const displayed = [...allCritical, ...allMajor.slice(0, Math.max(3, 8 - allCritical.length))].slice(0, 8)
  if (displayed.length === 0) return null

  const remaining = allCritical.length + allMajor.length - displayed.length

  return (
    <div className="card p-5 mb-6">
      <div className="flex items-center justify-between mb-3">
        <h2 className="section-label mb-0">Priority Action Items</h2>
        <span className="text-xs text-slate-400">{allCritical.length} critical · {allMajor.length} major</span>
      </div>
      <div className="flex flex-col gap-2">
        {displayed.map((gap, i) => (
          <Link
            key={i}
            to={`/audit/${auditId}/article/${gap.article_num}`}
            className="flex items-start gap-3 p-3 rounded-xl border border-slate-100 hover:border-brand-200 hover:bg-brand-50 transition-colors group"
          >
            <div className="flex flex-col items-center gap-1 shrink-0 w-16">
              <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold w-full text-center ${
                gap.severity === 'critical' ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'
              }`}>{gap.severity}</span>
              <span className="text-[10px] text-slate-400 font-semibold">Art. {gap.article_num}</span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-slate-800 group-hover:text-brand-700 transition-colors leading-snug">{gap.title}</p>
              <p className="text-xs text-slate-500 mt-0.5 line-clamp-2 leading-relaxed">{gap.description}</p>
            </div>
            <svg className="w-4 h-4 text-slate-300 group-hover:text-brand-400 shrink-0 mt-1 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </Link>
        ))}
      </div>
      {remaining > 0 && (
        <p className="text-xs text-slate-400 mt-3 text-center">
          +{remaining} more gaps — click individual article cards below for full details
        </p>
      )}
    </div>
  )
}

// ── Actor panel ──────────────────────────────────────────────────────────────

const ACTOR_LABELS: Record<string, string> = {
  provider: 'Provider', deployer: 'Deployer', importer: 'Importer',
  distributor: 'Distributor', unknown: 'Unknown',
}

function ActorPanel({ actor }: { actor: ActorClassification }) {
  const confidencePct = Math.round(actor.confidence * 100)
  const isLow = actor.confidence < 0.60

  return (
    <div className={`card p-4 ${isLow ? 'border-amber-300' : ''}`}>
      <p className="text-xs font-bold text-slate-400 uppercase tracking-wide mb-2">Article 3 — Actor Role</p>

      {isLow && (
        <div className="flex items-center gap-1.5 mb-2 px-2.5 py-1.5 bg-amber-50 rounded-lg border border-amber-200">
          <svg className="w-3.5 h-3.5 text-amber-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="text-xs font-semibold text-amber-700">Low confidence — verify actor role manually</span>
        </div>
      )}

      <p className="text-base font-bold text-slate-900 mb-1">{ACTOR_LABELS[actor.actor_type] ?? actor.actor_type}</p>
      <div className="flex items-center gap-2 mb-2">
        <div className="flex-1 h-1.5 rounded-full bg-slate-100">
          <div
            className={`h-1.5 rounded-full ${isLow ? 'bg-amber-400' : 'bg-blue-500'}`}
            style={{ width: `${confidencePct}%` }}
          />
        </div>
        <span className="text-xs text-slate-500 shrink-0">{confidencePct}%</span>
      </div>

      {actor.matched_signals.length > 0 && (
        <div className="mb-2">
          <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide mb-1">Detected signals</p>
          <div className="flex flex-wrap gap-1">
            {actor.matched_signals.map((s, i) => (
              <span key={i} className="px-1.5 py-0.5 rounded-full bg-blue-50 text-[10px] text-blue-700 border border-blue-100">{s}</span>
            ))}
          </div>
        </div>
      )}

      {actor.reasoning && <p className="text-xs text-slate-500 leading-relaxed">{actor.reasoning}</p>}
    </div>
  )
}

// ── Applicability panel ──────────────────────────────────────────────────────

function ApplicabilityPanel({ applicability }: { applicability: ApplicabilityResult }) {
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

      {/* Annex I safety-component signal */}
      {applicability.annex_i_triggered && (
        <div className="flex items-center gap-1.5 mb-2 px-2 py-1 bg-orange-50 rounded-lg border border-orange-200">
          <svg className="w-3 h-3 text-orange-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          <span className="text-[10px] font-semibold text-orange-700">Article 6(1) Annex I safety-component signals detected</span>
        </div>
      )}

      {/* Annex III category matches with keywords */}
      {applicability.annex_iii_matches.length > 0 && (
        <div className="mb-2 space-y-1.5">
          {applicability.annex_iii_matches.map(match => (
            <div key={match.category} className="rounded-lg bg-slate-50 p-2 border border-slate-100">
              <p className="text-[10px] font-bold text-slate-600 mb-1">Annex III — {match.category_name}</p>
              {match.matched_keywords.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {match.matched_keywords.map((kw, i) => (
                    <span key={i} className="px-1.5 py-0.5 rounded bg-amber-50 text-[10px] text-amber-700 border border-amber-100">{kw}</span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Applicable EU AI Act articles */}
      {applicability.applicable_articles.length > 0 && (
        <div className="mb-2">
          <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide mb-1">EU AI Act obligations</p>
          <div className="flex flex-wrap gap-1">
            {applicability.applicable_articles.map(n => (
              <span key={n} className="px-1.5 py-0.5 rounded bg-blue-50 text-[10px] font-bold text-blue-700 border border-blue-100">Art. {n}</span>
            ))}
          </div>
        </div>
      )}

      {/* Applicable GDPR articles */}
      {applicability.gdpr_applicable_articles.length > 0 && (
        <div className="mb-2">
          <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide mb-1">GDPR obligations</p>
          <div className="flex flex-wrap gap-1">
            {applicability.gdpr_applicable_articles.map(n => (
              <span key={n} className="px-1.5 py-0.5 rounded bg-purple-50 text-[10px] font-bold text-purple-700 border border-purple-100">Art. {n}</span>
            ))}
          </div>
        </div>
      )}

      {applicability.reasoning && (
        <p className="text-xs text-slate-500 leading-relaxed mt-1">{applicability.reasoning}</p>
      )}
    </div>
  )
}

// ── Evidence summary panel (in the 3-col grid) ───────────────────────────────

function EvidenceCoverageBar({ items, label, color }: { items: EvidenceItem[]; label: string; color: string }) {
  if (items.length === 0) return null
  const total   = items.length
  const fully   = items.filter(i => i.coverage >= 1.0).length
  const partial = items.filter(i => i.coverage > 0 && i.coverage < 1.0).length
  const missing = items.filter(i => i.coverage === 0).length
  const pct     = Math.round((items.reduce((s, i) => s + i.coverage, 0) / total) * 100)

  return (
    <div className="mt-2.5">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide">{label}</span>
        <span className={`text-[10px] font-bold ${color}`}>{pct}%</span>
      </div>
      <div className="h-1 rounded-full bg-slate-100 mb-1.5">
        <div
          className={`h-1 rounded-full ${pct >= 70 ? 'bg-emerald-500' : pct >= 40 ? 'bg-amber-400' : 'bg-red-500'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex gap-1.5 text-[10px]">
        <span className="text-emerald-600">{fully} satisfied</span>
        <span className="text-slate-300">·</span>
        <span className="text-amber-600">{partial} partial</span>
        <span className="text-slate-300">·</span>
        <span className="text-red-600">{missing} missing</span>
      </div>
    </div>
  )
}

function EvidenceSummaryPanel({ evidence }: { evidence: EvidenceMap }) {
  const coveragePct  = Math.round(evidence.overall_coverage * 100)
  const aiActItems   = evidence.items.filter(i => i.regulation === 'eu_ai_act')
  const gdprItems    = evidence.items.filter(i => i.regulation === 'gdpr')

  return (
    <div className="card p-4">
      <p className="text-xs font-bold text-slate-400 uppercase tracking-wide mb-2">Evidence Coverage</p>
      <p className="text-2xl font-bold text-slate-900 mb-1">{coveragePct}%</p>
      <div className="h-1.5 rounded-full bg-slate-100 mb-3">
        <div
          className={`h-1.5 rounded-full ${coveragePct >= 70 ? 'bg-emerald-500' : coveragePct >= 40 ? 'bg-amber-400' : 'bg-red-500'}`}
          style={{ width: `${coveragePct}%` }}
        />
      </div>

      <div className="grid grid-cols-3 gap-1 text-center mb-1">
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

      <p className="text-[10px] text-slate-400 text-center mb-1">{evidence.total_obligations} total obligations checked</p>

      {(aiActItems.length > 0 || gdprItems.length > 0) && (
        <div className="border-t border-slate-100 pt-2 mt-2 space-y-0.5">
          <EvidenceCoverageBar items={aiActItems} label="EU AI Act" color="text-blue-600" />
          <EvidenceCoverageBar items={gdprItems}  label="GDPR"      color="text-purple-600" />
        </div>
      )}
    </div>
  )
}

// ── Legal obligation coverage drill-down (full width) ────────────────────────

function EvidenceObligationsSection({ evidence }: { evidence: EvidenceMap }) {
  const [open, setOpen] = useState(true)

  const aiActItems = evidence.items.filter(i => i.regulation === 'eu_ai_act')
  const gdprItems  = evidence.items.filter(i => i.regulation === 'gdpr')
  const totalMissingArtefacts = evidence.items.reduce((s, i) => s + i.missing_evidence.length, 0)

  return (
    <div className="card mb-6 overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-5 py-4 hover:bg-slate-50 transition-colors text-left"
      >
        <div className="flex items-center gap-3 flex-wrap">
          <p className="text-sm font-bold text-slate-800">Legal Obligation Coverage</p>
          <span className="text-xs text-slate-400">{evidence.total_obligations} obligations</span>
          {totalMissingArtefacts > 0 && (
            <span className="px-2 py-0.5 rounded-full bg-red-100 text-red-700 text-[10px] font-bold">
              {totalMissingArtefacts} artefact{totalMissingArtefacts !== 1 ? 's' : ''} missing — action required
            </span>
          )}
        </div>
        <svg
          className={`w-5 h-5 text-slate-400 transition-transform duration-200 shrink-0 ml-3 ${open ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="border-t border-slate-100">
          {aiActItems.length > 0 && (
            <ObligationGroup title="EU AI Act" items={aiActItems} accentColor="blue" />
          )}
          {gdprItems.length > 0 && (
            <ObligationGroup title="GDPR" items={gdprItems} accentColor="purple" />
          )}
        </div>
      )}
    </div>
  )
}

function ObligationGroup({ title, items, accentColor }: { title: string; items: EvidenceItem[]; accentColor: 'blue' | 'purple' }) {
  const fully   = items.filter(i => i.coverage >= 1.0).length
  const partial = items.filter(i => i.coverage > 0 && i.coverage < 1.0).length
  const missing = items.filter(i => i.coverage === 0.0).length

  const bg   = accentColor === 'blue' ? 'bg-blue-50'   : 'bg-purple-50'
  const text = accentColor === 'blue' ? 'text-blue-700' : 'text-purple-700'

  return (
    <div>
      <div className={`px-5 py-2 ${bg} flex items-center justify-between`}>
        <span className={`text-xs font-bold ${text}`}>{title}</span>
        <div className="flex gap-4 text-[10px]">
          <span className="text-emerald-600 font-semibold">{fully} fully satisfied</span>
          <span className="text-amber-600 font-semibold">{partial} partial</span>
          {missing > 0 && <span className="text-red-600 font-semibold">{missing} missing</span>}
        </div>
      </div>
      <div className="divide-y divide-slate-100">
        {items.map(item => <ObligationRow key={item.obligation_id} item={item} />)}
      </div>
    </div>
  )
}

function ObligationRow({ item }: { item: EvidenceItem }) {
  const pct = Math.round(item.coverage * 100)
  const isFullyCovered = item.coverage >= 1.0
  const isPartial      = item.coverage > 0 && item.coverage < 1.0

  const rowBg     = isFullyCovered ? 'bg-emerald-50/30' : isPartial ? 'bg-amber-50/30' : 'bg-red-50/40'
  const pctColor  = isFullyCovered ? 'text-emerald-600' : isPartial ? 'text-amber-600' : 'text-red-600'
  const barColor  = isFullyCovered ? 'bg-emerald-500'   : isPartial ? 'bg-amber-400'   : 'bg-red-400'

  return (
    <div className={`px-5 py-3.5 ${rowBg}`}>
      {/* Header row: ID, article, coverage % */}
      <div className="flex items-center gap-2 mb-1.5 flex-wrap">
        <span className="font-mono text-[10px] text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded">{item.obligation_id}</span>
        {item.article && (
          <span className="text-[10px] font-semibold text-slate-600 bg-white border border-slate-200 px-1.5 py-0.5 rounded">{item.article}</span>
        )}
        <div className="ml-auto flex items-center gap-2">
          <div className="w-16 h-1 rounded-full bg-slate-200">
            <div className={`h-1 rounded-full ${barColor}`} style={{ width: `${pct}%` }} />
          </div>
          <span className={`text-[10px] font-bold ${pctColor}`}>{pct}%</span>
        </div>
      </div>

      {/* Requirement text */}
      {item.requirement && (
        <p className="text-xs text-slate-600 mb-2 leading-relaxed">
          {item.requirement.length > 180 ? item.requirement.slice(0, 180) + '…' : item.requirement}
        </p>
      )}

      {/* Evidence artefacts: satisfied (green) + missing (red) */}
      {(item.satisfied_evidence.length > 0 || item.missing_evidence.length > 0) && (
        <div className="flex flex-wrap gap-1">
          {item.satisfied_evidence.map((e, i) => (
            <span key={i} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-50 text-[10px] text-emerald-700 border border-emerald-200">
              <svg className="w-2.5 h-2.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
              {e}
            </span>
          ))}
          {item.missing_evidence.map((e, i) => (
            <span key={i} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-50 text-[10px] text-red-700 border border-red-200 font-medium">
              <svg className="w-2.5 h-2.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
              </svg>
              {e}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Shared helpers ────────────────────────────────────────────────────────────

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
        <p className={`text-xs font-bold uppercase tracking-widest ${cfg.text} opacity-70`}>Risk Tier Comparison</p>
        <span className={`text-xs ml-auto italic ${cfg.text} opacity-50`}>Wizard result is informational only</span>
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
