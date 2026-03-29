// Compliance dashboard: overall score, risk tier, 7 article cards, PDF download.

import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import Layout from '../components/Layout'
import ScoreRadial from '../components/ScoreRadial'
import ArticleCard from '../components/ArticleCard'
import ReportDownload from '../components/ReportDownload'
import { fetchReport } from '../hooks/useReport'
import { riskTierLabel, formatDate } from '../utils/formatters'
import type { ComplianceReport } from '../types'

export default function Dashboard() {
  const { auditId } = useParams<{ auditId: string }>()
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

  const { label: tierLabel, className: tierClass } = riskTierLabel(report.risk_tier)
  const sorted = [...report.article_scores].sort((a, b) => a.article_num - b.article_num)

  return (
    <Layout>
      {/* Summary header */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 mb-6 flex flex-col sm:flex-row items-center gap-6">
        <ScoreRadial score={report.overall_score} size={140} label="Overall Score" />

        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2 mb-1">
            <h1 className="text-xl font-bold text-slate-800">Compliance Report</h1>
            <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${tierClass}`}>
              {tierLabel}
            </span>
          </div>
          <p className="text-sm text-slate-500 mb-3">{formatDate(report.created_at)}</p>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
            <StatBox label="Total Chunks" value={report.total_chunks} />
            <StatBox label="Classified" value={report.classified_chunks} />
            <StatBox label="Language" value={report.language.toUpperCase()} />
            <StatBox label="Source Files" value={report.source_files.length} />
          </div>
        </div>

        <div className="shrink-0">
          <ReportDownload auditId={report.audit_id} />
        </div>
      </div>

      {/* Article grid */}
      <h2 className="text-base font-semibold text-slate-600 mb-3 uppercase tracking-wide text-xs">
        Article Scores (EU AI Act Articles 9–15)
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {sorted.map(article => (
          <ArticleCard key={article.article_num} score={article} auditId={report.audit_id} />
        ))}
      </div>

      {sorted.length === 0 && (
        <div className="text-center py-12 text-slate-400">
          No article scores found in this report.
        </div>
      )}
    </Layout>
  )
}

function StatBox({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-slate-50 rounded-lg px-3 py-2">
      <p className="text-xs text-slate-400">{label}</p>
      <p className="font-semibold text-slate-700">{value}</p>
    </div>
  )
}
