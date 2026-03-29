// PDF download button for a completed audit.

import { useState } from 'react'
import { downloadPdf } from '../hooks/useReport'

interface ReportDownloadProps {
  auditId: string
}

export default function ReportDownload({ auditId }: ReportDownloadProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleDownload = async () => {
    setLoading(true)
    setError(null)
    try {
      await downloadPdf(auditId)
    } catch {
      setError('Download failed. Try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <button
        onClick={handleDownload}
        disabled={loading}
        className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-brand-600 text-white
          text-sm font-medium hover:bg-brand-700 disabled:opacity-60 disabled:cursor-not-allowed
          transition-colors"
      >
        {loading ? (
          <>
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
            Generating PDF…
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Download PDF Report
          </>
        )}
      </button>
      {error && <p className="mt-1 text-sm text-red-600">{error}</p>}
    </div>
  )
}
