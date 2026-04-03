// Premium article summary card for the dashboard grid.

import { Link } from 'react-router-dom'
import type { ArticleScore } from '../types'
import { ARTICLE_NAMES } from '../utils/formatters'

interface ArticleCardProps {
  score: ArticleScore
  auditId: string
}

const SEVERITY_ORDER = { critical: 0, major: 1, minor: 2 }

const ARTICLE_ICONS: Record<number, string> = {
  9: 'M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z',
  10: 'M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4',
  11: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z',
  12: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2',
  13: 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
  14: 'M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z',
  15: 'M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z',
}

function gapTheme(criticalCount: number, majorCount: number, minorCount: number) {
  if (criticalCount > 0) return {
    bar: 'bg-red-500', num: 'text-red-600', badge: 'bg-red-50 text-red-700 border-red-200',
    dot: 'bg-red-500', border: 'hover:border-red-200',
  }
  if (majorCount > 0) return {
    bar: 'bg-amber-500', num: 'text-amber-600', badge: 'bg-amber-50 text-amber-700 border-amber-200',
    dot: 'bg-amber-500', border: 'hover:border-amber-200',
  }
  if (minorCount > 0) return {
    bar: 'bg-blue-400', num: 'text-blue-600', badge: 'bg-blue-50 text-blue-700 border-blue-200',
    dot: 'bg-blue-400', border: 'hover:border-blue-200',
  }
  return {
    bar: 'bg-emerald-500', num: 'text-emerald-600', badge: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    dot: 'bg-emerald-500', border: 'hover:border-emerald-200',
  }
}

export default function ArticleCard({ score, auditId }: ArticleCardProps) {
  const name = ARTICLE_NAMES[score.article_num] ?? score.domain.replace(/_/g, ' ')
  const criticalCount = score.gaps.filter(g => g.severity === 'critical').length
  const majorCount = score.gaps.filter(g => g.severity === 'major').length
  const minorCount = score.gaps.filter(g => g.severity === 'minor').length
  const theme = gapTheme(criticalCount, majorCount, minorCount)
  const iconPath = ARTICLE_ICONS[score.article_num] ?? ARTICLE_ICONS[9]

  const worstGap = [...score.gaps].sort(
    (a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity],
  )[0]

  return (
    <Link
      to={`/audit/${auditId}/article/${score.article_num}`}
      className={`group block card card-hover p-5 ${theme.border}`}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0 transition-colors">
            <svg className="w-4.5 h-4.5 text-brand-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
              <path strokeLinecap="round" strokeLinejoin="round" d={iconPath} />
            </svg>
          </div>
          <div className="min-w-0">
            <p className="text-xs font-semibold text-brand-600 uppercase tracking-wide leading-none mb-1">
              Art. {score.article_num}
            </p>
            <h3 className="text-sm font-semibold text-slate-800 leading-tight group-hover:text-brand-700 transition-colors">
              {name}
            </h3>
          </div>
        </div>
        <span className={`text-2xl font-extrabold tabular-nums shrink-0 ${theme.num}`}>
          {Math.round(score.score)}
        </span>
      </div>

      {/* Score bar */}
      <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden mb-3">
        <div
          className={`h-full rounded-full transition-all duration-500 ${theme.bar}`}
          style={{ width: `${score.score}%` }}
        />
      </div>

      {/* Gap counts */}
      <div className="flex items-center gap-2.5 text-xs">
        {criticalCount > 0 && (
          <span className="flex items-center gap-1 font-medium text-red-600">
            <span className="w-1.5 h-1.5 rounded-full bg-red-500 inline-block" />
            {criticalCount} critical
          </span>
        )}
        {majorCount > 0 && (
          <span className="flex items-center gap-1 font-medium text-amber-600">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500 inline-block" />
            {majorCount} major
          </span>
        )}
        {minorCount > 0 && (
          <span className="flex items-center gap-1 font-medium text-blue-500">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 inline-block" />
            {minorCount} minor
          </span>
        )}
        {score.gaps.length === 0 && (
          <span className="flex items-center gap-1 font-medium text-emerald-600">
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
            </svg>
            No gaps
          </span>
        )}
        <span className="ml-auto text-slate-400">{score.chunk_count} chunks</span>
      </div>

      {/* Worst gap preview */}
      {worstGap && (
        <p className="mt-3 text-xs text-slate-400 truncate border-t border-slate-100 pt-2.5">
          {worstGap.title}
        </p>
      )}
    </Link>
  )
}
