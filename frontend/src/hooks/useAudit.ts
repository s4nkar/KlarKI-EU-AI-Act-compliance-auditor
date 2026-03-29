// Upload documents and poll audit status.

import apiClient from '../api/client'
import type { AuditResponse, AuditStatus } from '../types'

const TERMINAL_STATUSES: AuditStatus[] = ['complete', 'failed']
const POLL_INTERVAL_MS = 2000

/** Upload files or raw text and start an audit. Returns audit_id. */
export async function startAudit(files: File[], rawText?: string): Promise<string> {
  const form = new FormData()

  if (files.length > 0) {
    form.append('file', files[0])
  } else if (rawText) {
    form.append('raw_text', rawText)
  } else {
    throw new Error('Provide at least one file or raw text.')
  }

  const resp = await apiClient.post<{ status: string; data: { audit_id: string } }>(
    '/api/v1/audit/upload',
    form,
    { headers: { 'Content-Type': 'multipart/form-data' } },
  )
  return resp.data.data.audit_id
}

/** Poll audit status every 2 s until COMPLETE or FAILED. */
export async function pollAudit(
  auditId: string,
  onStatusChange?: (status: AuditStatus) => void,
): Promise<AuditResponse> {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const resp = await apiClient.get<AuditResponse>(
          `/api/v1/audit/${auditId}`,
        )
        const audit = resp.data
        onStatusChange?.(audit.status)

        if (TERMINAL_STATUSES.includes(audit.status)) {
          resolve(audit)
        } else {
          setTimeout(tick, POLL_INTERVAL_MS)
        }
      } catch (err) {
        reject(err)
      }
    }
    tick()
  })
}
