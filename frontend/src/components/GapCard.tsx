// Single gap: severity badge, title, description.

import type { GapItem } from '../types'
import { severityLabel } from '../utils/formatters'

interface GapCardProps {
  gap: GapItem
}

export default function GapCard({ gap }: GapCardProps) {
  const { label, className } = severityLabel(gap.severity)

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4 flex gap-3">
      <div className="pt-0.5 shrink-0">
        <span className={`inline-block px-2 py-0.5 rounded text-xs font-semibold ${className}`}>
          {label}
        </span>
      </div>
      <div className="min-w-0">
        <p className="font-semibold text-slate-800 text-sm leading-snug">{gap.title}</p>
        <p className="text-slate-500 text-sm mt-1 leading-relaxed">{gap.description}</p>
      </div>
    </div>
  )
}
