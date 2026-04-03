// Upload page: drag-drop file or paste raw text → start audit → show progress → redirect to dashboard.

import { useState } from 'react'
import { useNavigate, useSearchParams, Link } from 'react-router-dom'
import Layout from '../components/Layout'
import FileDropzone from '../components/FileDropzone'
import TextPasteArea from '../components/TextPasteArea'
import LoadingSpinner from '../components/LoadingSpinner'
import { startAudit, pollAudit } from '../hooks/useAudit'
import { riskTierLabel } from '../utils/formatters'
import type { AuditStatus, RiskTier } from '../types'

type InputMode = 'file' | 'text'

const VALID_TIERS: RiskTier[] = ['prohibited', 'high', 'limited', 'minimal']

export default function Upload() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const tierParam = searchParams.get('tier') as RiskTier | null
  const wizardTier: RiskTier | null =
    tierParam && VALID_TIERS.includes(tierParam) ? tierParam : null

  const [mode, setMode]     = useState<InputMode>('file')
  const [files, setFiles]   = useState<File[]>([])
  const [rawText, setRawText] = useState('')
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState<AuditStatus | null>(null)
  const [error, setError]   = useState<string | null>(null)

  const canSubmit = mode === 'file' ? files.length > 0 : rawText.trim().length > 0

  const handleSubmit = async () => {
    if (!canSubmit) return
    setRunning(true)
    setError(null)
    setStatus('uploading')
    try {
      const auditId = await startAudit(
        mode === 'file' ? files : [],
        mode === 'text' ? rawText : undefined,
        wizardTier ?? undefined,
      )
      const audit = await pollAudit(auditId, s => setStatus(s))
      if (audit.status === 'failed') {
        setError('Audit failed. Please try again with a different document.')
        setRunning(false)
        return
      }
      navigate(`/audit/${auditId}`)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unexpected error.')
      setRunning(false)
      setStatus(null)
    }
  }

  return (
    <Layout>
      <div className="max-w-2xl mx-auto">
        {/* Page header */}
        <div className="mb-8 text-center">
          <div className="flex items-center justify-center gap-2 mb-3">
            <span className="badge bg-brand-100 text-brand-700 border border-brand-200">Step 2 of 2</span>
          </div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
            Upload Documentation
          </h1>
          <p className="mt-2 text-slate-500 max-w-lg mx-auto leading-relaxed">
            Upload your AI system documentation to analyse compliance against EU AI Act Articles 9–15 and GDPR.
          </p>
        </div>

        {/* Risk tier banner */}
        {wizardTier ? (
          <WizardTierBanner tier={wizardTier} />
        ) : (
          <div className="mb-5 flex items-center gap-3 p-4 bg-amber-50 border border-amber-200 rounded-xl text-sm">
            <svg className="w-4 h-4 text-amber-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="text-amber-700 flex-1">
              No risk tier selected. Complete the Annex III wizard first for a more accurate audit.
            </span>
            <Link to="/" className="shrink-0 text-sm font-semibold text-amber-800 hover:text-amber-900 underline">
              Go to Step 1 →
            </Link>
          </div>
        )}

        {running ? (
          /* Running state */
          <div className="card p-10 text-center">
            <div className="mb-8">
              <div className="w-16 h-16 rounded-2xl bg-brand-50 flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-brand-600 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <h2 className="text-lg font-bold text-slate-800 mb-1">Analysing your document</h2>
              <p className="text-sm text-slate-400">This typically takes 2–5 minutes depending on length.</p>
            </div>
            <LoadingSpinner status={status!} />
          </div>
        ) : (
          /* Input form */
          <div className="card shadow-card">
            {/* Mode tabs */}
            <div className="p-5 border-b border-slate-100">
              <div className="flex gap-1 bg-slate-100 rounded-xl p-1 w-fit">
                {(['file', 'text'] as InputMode[]).map(m => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`px-5 py-2 rounded-lg text-sm font-semibold transition-all duration-150 ${
                      mode === m
                        ? 'bg-white text-brand-700 shadow-sm'
                        : 'text-slate-500 hover:text-slate-700'
                    }`}
                  >
                    {m === 'file' ? (
                      <span className="flex items-center gap-2">
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        Upload File
                      </span>
                    ) : (
                      <span className="flex items-center gap-2">
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                            d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2" />
                        </svg>
                        Paste Text
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>

            <div className="p-5">
              {mode === 'file' ? (
                <FileDropzone files={files} onChange={setFiles} />
              ) : (
                <TextPasteArea value={rawText} onChange={setRawText} />
              )}

              {error && (
                <div className="mt-4 flex items-start gap-2.5 p-3.5 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
                  <svg className="w-4 h-4 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  {error}
                </div>
              )}
            </div>

            <div className="px-5 pb-5 flex items-center justify-between gap-4 border-t border-slate-100 pt-4">
              <p className="text-xs text-slate-400 flex items-center gap-1.5">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
                All processing is local — no data leaves your machine
              </p>
              <button
                onClick={handleSubmit}
                disabled={!canSubmit}
                className="btn-primary"
              >
                Start Audit
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </button>
            </div>
          </div>
        )}
      </div>
    </Layout>
  )
}

function WizardTierBanner({ tier }: { tier: RiskTier }) {
  const { label, className } = riskTierLabel(tier)

  return (
    <div className="mb-5 flex items-center gap-3 p-4 bg-slate-50 border border-slate-200 rounded-xl text-sm">
      <svg className="w-4 h-4 text-slate-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
      <span className="text-slate-600">Risk tier from Step 1:</span>
      <span className={`badge ${className}`}>{label}</span>
      <span className="text-slate-400 text-xs ml-auto">Recorded alongside your audit results</span>
    </div>
  )
}
