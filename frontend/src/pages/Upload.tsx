// Upload page: drag-drop file or paste raw text → start audit → show progress → redirect to dashboard.

import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Layout from '../components/Layout'
import FileDropzone from '../components/FileDropzone'
import TextPasteArea from '../components/TextPasteArea'
import LoadingSpinner from '../components/LoadingSpinner'
import { startAudit, pollAudit } from '../hooks/useAudit'
import type { AuditStatus } from '../types'

type InputMode = 'file' | 'text'

export default function Upload() {
  const navigate = useNavigate()
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
      const auditId = await startAudit(mode === 'file' ? files : [], mode === 'text' ? rawText : undefined)
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
          <h1 className="text-3xl font-bold text-slate-800">Start Compliance Audit</h1>
          <p className="mt-2 text-slate-500">
            Upload your AI system documentation to analyse compliance against EU AI Act Articles 9–15 and GDPR.
          </p>
        </div>

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
