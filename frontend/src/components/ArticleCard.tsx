// Summary card per article on the dashboard.

import { Link } from 'react-router-dom'
import type { ArticleScore } from '../types'
import { ARTICLE_NAMES } from '../utils/formatters'

interface ArticleCardProps {
  score: ArticleScore
  auditId: string
}

const SEVERITY_ORDER = { critical: 0, major: 1, minor: 2 }

export default function ArticleCard({ score, auditId }: ArticleCardProps) {
  const name = ARTICLE_NAMES[score.article_num] ?? score.domain.replace(/_/g, ' ')
  const criticalCount = score.gaps.filter(g => g.severity === 'critical').length
  const majorCount = score.gaps.filter(g => g.severity === 'major').length
  const minorCount = score.gaps.filter(g => g.severity === 'minor').length

  // Color driven by worst gap severity, not numeric score.
  // A 75 with a Critical gap must show red, not green.
  const gapColor =
    criticalCount > 0 ? { bar: 'bg-red-500',   num: 'text-red-600' }
    : majorCount > 0  ? { bar: 'bg-amber-500', num: 'text-amber-600' }
    : minorCount > 0  ? { bar: 'bg-blue-400',  num: 'text-blue-600' }
    :                   { bar: 'bg-green-500', num: 'text-green-600' }

  // Sort gaps to show worst first
  const worstGap = [...score.gaps].sort(
    (a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity],
  )[0]

  return (
    <Link
      to={`/audit/${auditId}/article/${score.article_num}`}
      className="block bg-white border border-slate-200 rounded-xl p-5 hover:border-brand-300 hover:shadow-md transition-all group"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div>
          <span className="text-xs font-semibold text-brand-600 uppercase tracking-wide">
            Article {score.article_num}
          </span>
          <h3 className="font-semibold text-slate-800 leading-tight mt-0.5 group-hover:text-brand-700 transition-colors">
            {name}
          </h3>
        </div>
        <span className={`text-2xl font-bold ${gapColor.num}`}>
          {Math.round(score.score)}
        </span>
      </div>

      {/* Score bar — width from score, color from worst gap severity */}
      <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden mb-3">
        <div
          className={`h-full rounded-full transition-all ${gapColor.bar}`}
          style={{ width: `${score.score}%` }}
        />
      </div>

      {/* Counts */}
      <div className="flex items-center gap-3 text-xs text-slate-500">
        {criticalCount > 0 && (
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-red-500 inline-block" />
            {criticalCount} critical
          </span>
        )}
        {majorCount > 0 && (
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-amber-500 inline-block" />
            {majorCount} major
          </span>
        )}
        {minorCount > 0 && (
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-blue-400 inline-block" />
            {minorCount} minor
          </span>
        )}
        {score.gaps.length === 0 && (
          <span className="text-green-600 font-medium">No gaps found</span>
        )}
        <span className="ml-auto">{score.chunk_count} chunks</span>
      </div>

      {/* Worst gap preview */}
      {worstGap && (
        <p className="mt-3 text-xs text-slate-400 truncate border-t border-slate-100 pt-2">
          {worstGap.title}
        </p>
      )}
    </Link>
  )
}
