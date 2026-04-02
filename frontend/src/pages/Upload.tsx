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
  const wizardTier: RiskTier | null = tierParam && VALID_TIERS.includes(tierParam) ? tierParam : null

  const [mode, setMode] = useState<InputMode>('file')
  const [files, setFiles] = useState<File[]>([])
  const [rawText, setRawText] = useState('')
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState<AuditStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

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
      const msg = err instanceof Error ? err.message : 'Unexpected error.'
      setError(msg)
      setRunning(false)
      setStatus(null)
    }
  }

  return (
    <Layout>
      <div className="max-w-2xl mx-auto">
        <div className="mb-8 text-center">
          <p className="text-xs font-semibold text-brand-600 uppercase tracking-wide mb-1">Step 2 of 2</p>
          <h1 className="text-3xl font-bold text-slate-800">Upload Documentation</h1>
          <p className="mt-2 text-slate-500">
            Upload your AI system documentation to analyse compliance against EU AI Act Articles 9–15 and GDPR.
          </p>
        </div>

        {wizardTier ? (
          <WizardTierBanner tier={wizardTier} />
        ) : (
          <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-700 flex items-center justify-between gap-3">
            <span>No risk tier selected. Complete the Annex III wizard first for a more trustworthy audit.</span>
            <Link to="/" className="shrink-0 font-semibold underline hover:text-amber-900">
              Go to Step 1 →
            </Link>
          </div>
        )}

        {running ? (
          <div className="bg-white border border-slate-200 rounded-2xl p-10 text-center">
            <p className="text-lg font-semibold text-slate-700 mb-6">Analysing your document…</p>
            <LoadingSpinner status={status!} />
            <p className="mt-6 text-sm text-slate-400">
              This may take 2–5 minutes depending on document length.
            </p>
          </div>
        ) : (
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            {/* Mode tabs */}
            <div className="flex border border-slate-200 rounded-lg p-1 mb-6 w-fit">
              <button
                onClick={() => setMode('file')}
                className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors
                  ${mode === 'file'
                    ? 'bg-brand-600 text-white'
                    : 'text-slate-500 hover:text-slate-700'}`}
              >
                Upload File
              </button>
              <button
                onClick={() => setMode('text')}
                className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors
                  ${mode === 'text'
                    ? 'bg-brand-600 text-white'
                    : 'text-slate-500 hover:text-slate-700'}`}
              >
                Paste Text
              </button>
            </div>

            {mode === 'file' ? (
              <FileDropzone files={files} onChange={setFiles} />
            ) : (
              <TextPasteArea value={rawText} onChange={setRawText} />
            )}

            {error && (
              <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
                {error}
              </div>
            )}

            <div className="mt-6 flex items-center justify-between">
              <p className="text-xs text-slate-400">
                All processing is local — no data leaves your machine.
              </p>
              <button
                onClick={handleSubmit}
                disabled={!canSubmit}
                className="px-6 py-2.5 rounded-lg bg-brand-600 text-white font-semibold text-sm
                  hover:bg-brand-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                Start Audit →
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
    <div className="mb-4 p-3 bg-slate-50 border border-slate-200 rounded-lg flex items-center gap-3 text-sm">
      <span className="text-slate-500">Risk tier from Step 1:</span>
      <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${className}`}>{label}</span>
      <span className="text-slate-400 text-xs ml-auto">
        This will be recorded alongside your document audit results.
      </span>
    </div>
  )
}
