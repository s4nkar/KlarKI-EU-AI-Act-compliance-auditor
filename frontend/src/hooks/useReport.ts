// Fetch report data and trigger PDF download.

import apiClient from '../api/client'
import type { ComplianceReport } from '../types'

/** Fetch the JSON report for a completed audit. */
export async function fetchReport(auditId: string): Promise<ComplianceReport> {
  const resp = await apiClient.get<{ status: string; data: ComplianceReport }>(
    `/api/v1/reports/${auditId}/json`,
  )
  return resp.data.data
}

/** Trigger browser PDF download for a completed audit. */
export async function downloadPdf(auditId: string): Promise<void> {
  const resp = await apiClient.get(`/api/v1/reports/${auditId}/pdf`, {
    responseType: 'blob',
  })
  const url = URL.createObjectURL(new Blob([resp.data], { type: 'application/pdf' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `klarki-report-${auditId.slice(0, 8)}.pdf`
  a.click()
  URL.revokeObjectURL(url)
}
