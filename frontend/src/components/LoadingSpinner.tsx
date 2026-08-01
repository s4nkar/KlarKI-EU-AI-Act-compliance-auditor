// Premium single-stage-focus progress indicator shown during audit processing.
// Shows one large card for the current pipeline stage (not a 9-item checklist),
// with a slim overall progress bar up top and real backend-reported sub-progress
// (files parsed, articles analysed) where the backend actually tracks it.
// gap_analyser.py's _LLM_SEMAPHORE serializes article analysis one at a time in
// practice, so "article N of 7" is real progress, not a simulated animation.

import type { AuditProgress, AuditStatus } from '../types'

interface Stage {
  status: AuditStatus
  label: string
  desc: string
  icon: string // SVG path `d` attribute
}

const STAGES: Stage[] = [
  {
    status: 'uploading', label: 'Uploading', desc: 'Sending your documents to the pipeline',
    icon: 'M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12',
  },
  {
    status: 'parsing', label: 'Parsing', desc: 'Extracting text and splitting into propositions',
    icon: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z',
  },
  {
    status: 'extracting_entities', label: 'Extracting Entities', desc: 'Detecting legal and risk entities via NER',
    icon: 'M21 21l-4.35-4.35M11 19a8 8 0 100-16 8 8 0 000 16z',
  },
  {
    status: 'classifying_risk', label: 'Assessing Risk', desc: 'Determining actor role and Art. 5/6 applicability',
    icon: 'M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z',
  },
  {
    status: 'classifying_chunks', label: 'Classifying Content', desc: 'Categorising each chunk by EU AI Act article',
    icon: 'M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z',
  },
  {
    status: 'analysing', label: 'Analysing', desc: 'Running gap analysis per article (RAG + LLM)',
    icon: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2',
  },
  {
    status: 'mapping_evidence', label: 'Mapping Evidence', desc: 'Matching obligations to documentation evidence',
    icon: 'M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1',
  },
  {
    status: 'scoring', label: 'Scoring', desc: 'Computing article compliance scores',
    icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
  },
  {
    status: 'complete', label: 'Complete', desc: 'Audit finished successfully',
    icon: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
  },
]

const FAILED_ICON = 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z'

const STATUS_ORDER: Record<AuditStatus, number> = {
  uploading: 0,
  parsing: 1,
  extracting_entities: 2,
  classifying_risk: 3,
  classifying_chunks: 4,
  analysing: 5,
  mapping_evidence: 6,
  scoring: 7,
  complete: 8,
  failed: 9,
}

// Stages shown as the minimal dot-trail at the bottom (excludes the terminal
// 'complete' stage — that's the destination, not something to wait through).
const DOT_STAGES = STAGES.filter(s => s.status !== 'complete')

function formatElapsed(totalSeconds: number): string {
  const m = Math.floor(totalSeconds / 60)
  const s = totalSeconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

function formatEta(seconds: number): string {
  if (seconds <= 0) return 'Finishing up…'
  if (seconds < 60) return '< 1 min remaining'
  return `~${Math.round(seconds / 60)} min remaining`
}

// 'pips' style (discrete segments) reads well for small, bounded counts
// (<=5 files, exactly 7 articles). 'bar' is used for chunk counts, which can
// run into the hundreds for a large real document — individual pips would
// be unreadably tiny/cluttered at that scale.
type SubProgress = { done: number; total: number; label: string; style: 'pips' | 'bar' }

function subProgressFor(
  stageStatus: AuditStatus,
  progress?: AuditProgress | null,
): SubProgress | null {
  if (!progress) return null
  if (stageStatus === 'parsing' && progress.files_total != null && progress.files_total > 1) {
    return {
      done: progress.files_done ?? 0,
      total: progress.files_total,
      label: `File ${progress.files_done ?? 0} of ${progress.files_total}`,
      style: 'pips',
    }
  }
  if (stageStatus === 'classifying_chunks' && progress.chunks_total != null && progress.chunks_total > 0) {
    return {
      done: progress.chunks_done ?? 0,
      total: progress.chunks_total,
      label: `Chunk ${progress.chunks_done ?? 0} of ${progress.chunks_total} classified`,
      style: 'bar',
    }
  }
  if (stageStatus === 'analysing' && progress.articles_total != null) {
    return {
      done: progress.articles_done ?? 0,
      total: progress.articles_total,
      label: `Article ${progress.articles_done ?? 0} of ${progress.articles_total} analysed`,
      style: 'pips',
    }
  }
  return null
}

// ETA is only meaningful on the two stages that can actually take a while.
const ETA_STAGES: AuditStatus[] = ['classifying_chunks', 'analysing']

export default function LoadingSpinner({
  status,
  progress,
  elapsedSeconds = 0,
  documentCount,
}: {
  status: AuditStatus
  progress?: AuditProgress | null
  elapsedSeconds?: number
  documentCount?: number
}) {
  const stepIndex = STATUS_ORDER[status] ?? 0
  const failed = status === 'failed'
  const stage = STAGES.find(s => s.status === status) ?? STAGES[0]
  const sub = subProgressFor(status, progress)
  const overallPct = Math.min(100, Math.round(((stepIndex + 1) / STAGES.length) * 100))

  return (
    <div className="w-full max-w-sm mx-auto">
      {/* Overall progress bar */}
      <div className="flex items-center justify-between text-xs text-slate-500 mb-1.5">
        <span>Step {Math.min(stepIndex + 1, STAGES.length)} of {STAGES.length}</span>
        <span className="flex items-center gap-1.5">
          <span className="tabular-nums font-medium text-slate-400">{formatElapsed(elapsedSeconds)}</span>
          <span>elapsed</span>
          {documentCount != null && documentCount > 1 && (
            <>
              <span className="text-slate-600">·</span>
              <span>{documentCount} documents</span>
            </>
          )}
        </span>
      </div>
      <div className="h-1 w-full bg-surface-raised rounded-full overflow-hidden mb-9">
        <div
          className={`h-full rounded-full transition-all duration-700 ${failed ? 'bg-red-500' : 'bg-gradient-brand'}`}
          style={{ width: `${failed ? 100 : overallPct}%` }}
        />
      </div>

      {/* Current-stage focus card */}
      <div className="flex flex-col items-center text-center">
        <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-4 relative ${
          failed ? 'bg-red-500/10' : 'bg-brand-500/10'
        }`}>
          <svg className={`w-8 h-8 ${failed ? 'text-red-400' : 'text-brand-400'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={failed ? FAILED_ICON : stage.icon} />
          </svg>
          {!failed && status !== 'complete' && (
            <span className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full bg-brand-500 flex items-center justify-center shadow-glow">
              <svg className="w-3 h-3 text-white animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
            </span>
          )}
        </div>

        <h2 className="text-lg font-bold text-slate-100 mb-1">
          {failed ? 'Audit Failed' : stage.label}
        </h2>
        <p className="text-sm text-slate-500 mb-1 max-w-xs">
          {failed ? 'Something went wrong during processing.' : (sub ? sub.label : stage.desc)}
        </p>

        {sub && sub.style === 'pips' && (
          <div className="flex items-center gap-1.5 mt-3">
            {Array.from({ length: sub.total }).map((_, i) => (
              <div
                key={i}
                className={`h-1.5 rounded-full transition-all duration-300 ${
                  i < sub.done ? 'w-6 bg-gradient-brand' : 'w-3 bg-surface-raised'
                }`}
              />
            ))}
          </div>
        )}

        {sub && sub.style === 'bar' && (
          <div className="w-full h-1.5 bg-surface-raised rounded-full overflow-hidden mt-3">
            <div
              className="h-full bg-gradient-brand rounded-full transition-all duration-300"
              style={{ width: `${sub.total > 0 ? Math.min(100, (sub.done / sub.total) * 100) : 0}%` }}
            />
          </div>
        )}

        {ETA_STAGES.includes(status) && progress?.estimated_seconds_remaining != null && (
          <p className="text-xs text-slate-500 mt-2.5">
            {formatEta(progress.estimated_seconds_remaining)}
          </p>
        )}
      </div>

      {/* Minimal stage dot-trail */}
      {!failed && (
        <div className="flex items-center justify-center gap-1.5 mt-9">
          {DOT_STAGES.map(s => {
            const isDone   = STATUS_ORDER[s.status] < stepIndex
            const isActive = s.status === status
            return (
              <div
                key={s.status}
                title={s.label}
                className={`rounded-full transition-all duration-300 ${
                  isActive ? 'w-2.5 h-2.5 bg-brand-400 shadow-glow'
                    : isDone ? 'w-1.5 h-1.5 bg-emerald-500'
                    : 'w-1.5 h-1.5 bg-surface-raised'
                }`}
              />
            )
          })}
        </div>
      )}
    </div>
  )
}
