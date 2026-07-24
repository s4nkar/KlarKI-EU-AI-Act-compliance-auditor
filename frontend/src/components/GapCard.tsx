// Premium gap card: severity badge, title, description.

import type { GapItem } from '../types'

interface GapCardProps {
  gap: GapItem
}

const SEVERITY_CONFIG = {
  critical: {
    badge: 'bg-red-500/10 text-red-400',
    border: 'border-line hover:border-red-500/30',
    icon: (
      <svg className="w-4 h-4 text-red-400 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    ),
  },
  major: {
    badge: 'bg-amber-500/10 text-amber-400',
    border: 'border-line hover:border-amber-500/30',
    icon: (
      <svg className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  },
  minor: {
    badge: 'bg-blue-500/10 text-blue-400',
    border: 'border-line hover:border-blue-500/30',
    icon: (
      <svg className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
      </svg>
    ),
  },
}

export default function GapCard({ gap }: GapCardProps) {
  const cfg = SEVERITY_CONFIG[gap.severity]
  const label = gap.severity.charAt(0).toUpperCase() + gap.severity.slice(1)

  return (
    <div className={`bg-surface border rounded-xl p-4 flex gap-3 transition-colors ${cfg.border}`}>
      {cfg.icon}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 mb-1.5 flex-wrap">
          <span className={`badge ${cfg.badge}`}>{label}</span>
          <p className="font-semibold text-slate-100 text-sm leading-snug">{gap.title}</p>
        </div>
        <p className="text-slate-400 text-sm leading-relaxed">{gap.description}</p>
      </div>
    </div>
  )
}
