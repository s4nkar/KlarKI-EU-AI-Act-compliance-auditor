// Pipeline stage progress indicator shown during audit processing.

import type { AuditStatus } from '../types'

const STAGES: { status: AuditStatus; label: string }[] = [
  { status: 'uploading',   label: 'Uploading document' },
  { status: 'parsing',     label: 'Parsing text' },
  { status: 'classifying', label: 'Classifying content' },
  { status: 'analysing',   label: 'Analysing compliance gaps' },
  { status: 'scoring',     label: 'Computing scores' },
  { status: 'complete',    label: 'Complete' },
]

const STATUS_ORDER: Record<AuditStatus, number> = {
  uploading:   0,
  parsing:     1,
  classifying: 2,
  analysing:   3,
  scoring:     4,
  complete:    5,
  failed:      6,
}

interface LoadingSpinnerProps {
  status: AuditStatus
}

export default function LoadingSpinner({ status }: LoadingSpinnerProps) {
  const current = STATUS_ORDER[status] ?? 0

  return (
    <div className="flex flex-col gap-3 w-full max-w-sm mx-auto">
      {STAGES.map((stage, i) => {
        const done = STATUS_ORDER[stage.status] < current
        const active = stage.status === status

        return (
          <div key={stage.status} className="flex items-center gap-3">
            <div className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 transition-colors
              ${done    ? 'bg-green-500'
              : active  ? 'bg-brand-600'
              :           'bg-slate-200'}`}
            >
              {done ? (
                <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                </svg>
              ) : active ? (
                <svg className="w-4 h-4 text-white animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
              ) : (
                <span className="text-xs font-semibold text-slate-400">{i + 1}</span>
              )}
            </div>
            <span className={`text-sm transition-colors
              ${done   ? 'text-green-600 font-medium'
              : active ? 'text-brand-700 font-semibold'
              :          'text-slate-400'}`}
            >
              {stage.label}
            </span>
          </div>
        )
      })}
    </div>
  )
}
