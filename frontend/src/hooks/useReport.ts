// Fetch report data and trigger PDF download. (Phase 3)

import type { ComplianceReport } from '../types'

/** Fetch the JSON report for a completed audit. (Phase 3) */
export async function fetchReport(_auditId: string): Promise<ComplianceReport> {
  throw new Error('useReport.fetchReport — implemented in Phase 3')
}

/** Trigger PDF download for a completed audit. (Phase 3) */
export async function downloadPdf(_auditId: string): Promise<void> {
  throw new Error('useReport.downloadPdf — implemented in Phase 3')
}
