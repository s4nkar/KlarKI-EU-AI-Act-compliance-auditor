// Per-article detail: score, gaps sorted by severity, recommendations.

import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import Layout from '../components/Layout'
import ScoreRadial from '../components/ScoreRadial'
import GapCard from '../components/GapCard'
import ReportDownload from '../components/ReportDownload'
import { fetchReport } from '../hooks/useReport'
import { ARTICLE_NAMES } from '../utils/formatters'
import type { ArticleScore, ComplianceReport } from '../types'

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

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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
    </Layout>
  )
}
