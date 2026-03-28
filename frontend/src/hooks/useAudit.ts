// Upload documents and poll audit status. (Phase 3)

import type { AuditResponse } from '../types'

/** Upload files or raw text and start an audit. Returns audit_id. (Phase 3) */
export async function startAudit(_files: File[], _rawText?: string): Promise<string> {
  throw new Error('useAudit.startAudit — implemented in Phase 3')
}

/** Poll audit status until COMPLETE or FAILED. (Phase 3) */
export async function pollAudit(_auditId: string): Promise<AuditResponse> {
  throw new Error('useAudit.pollAudit — implemented in Phase 3')
}
