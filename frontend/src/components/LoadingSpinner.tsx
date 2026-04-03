// Premium pipeline stage progress indicator shown during audit processing.

import type { AuditStatus } from '../types'

const STAGES: { status: AuditStatus; label: string; desc: string }[] = [
  { status: 'uploading',   label: 'Uploading',    desc: 'Sending your document to the pipeline' },
  { status: 'parsing',     label: 'Parsing',      desc: 'Extracting text from the document'      },
  { status: 'classifying', label: 'Classifying',  desc: 'Categorising content with BERT'         },
  { status: 'analysing',   label: 'Analysing',    desc: 'Identifying compliance gaps via LLM'    },
  { status: 'scoring',     label: 'Scoring',      desc: 'Computing article compliance scores'    },
  { status: 'complete',    label: 'Complete',     desc: 'Audit finished successfully'            },
]

const STATUS_ORDER: Record<AuditStatus, number> = {
  uploading: 0, parsing: 1, classifying: 2, analysing: 3, scoring: 4, complete: 5, failed: 6,
}

export default function LoadingSpinner({ status }: { status: AuditStatus }) {
  const current = STATUS_ORDER[status] ?? 0

  return (
    <div className="w-full max-w-sm mx-auto flex flex-col gap-2">
      {STAGES.map((stage, i) => {
        const done   = STATUS_ORDER[stage.status] < current
        const active = stage.status === status

        return (
          <div key={stage.status} className="flex items-center gap-4">
            {/* Step dot */}
            <div className="relative shrink-0">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center transition-all duration-300 ${
                done   ? 'bg-emerald-500 shadow-sm'
                : active ? 'bg-brand-600 shadow-sm shadow-brand-200'
                :          'bg-slate-100'
              }`}>
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
                  <span className="text-xs font-bold text-slate-400">{i + 1}</span>
                )}
              </div>
              {/* Vertical connector */}
              {i < STAGES.length - 1 && (
                <div className={`absolute top-8 left-1/2 -translate-x-1/2 w-0.5 h-2 transition-colors duration-300 ${
                  done ? 'bg-emerald-300' : 'bg-slate-200'
                }`} />
              )}
            </div>

            {/* Label */}
            <div className="flex-1 min-w-0">
              <p className={`text-sm font-semibold transition-colors ${
                done   ? 'text-emerald-600'
                : active ? 'text-brand-700'
                :          'text-slate-400'
              }`}>
                {stage.label}
              </p>
              {active && (
                <p className="text-xs text-slate-400 mt-0.5">{stage.desc}</p>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
