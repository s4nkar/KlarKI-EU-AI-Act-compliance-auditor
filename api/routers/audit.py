"""Audit router — document upload and status polling endpoints. (Phase 2)

Endpoints:
    POST /api/v1/audit/upload       — Upload files or raw text, start audit pipeline
    GET  /api/v1/audit/{audit_id}   — Fetch full AuditResponse (status + report)
    GET  /api/v1/audit/{audit_id}/status — Lightweight status poll
"""

from fastapi import APIRouter
from models.schemas import APIResponse, AuditResponse, AuditStatus

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])

# In-memory audit state store (Phase 2 — replaced with Redis/DB in production)
_audits: dict[str, AuditResponse] = {}


# @router.post("/upload")
# async def upload_document(...) -> APIResponse:
#     """Phase 2: Accept UploadFile + optional raw_text Form field.
#     Starts audit pipeline in BackgroundTasks. Returns audit_id immediately."""
#     raise NotImplementedError("audit.upload_document — implemented in Phase 2")


# @router.get("/{audit_id}")
# async def get_audit(audit_id: str) -> AuditResponse:
#     """Phase 2: Return full AuditResponse including ComplianceReport when COMPLETE."""
#     raise NotImplementedError("audit.get_audit — implemented in Phase 2")


# @router.get("/{audit_id}/status")
# async def get_audit_status(audit_id: str) -> APIResponse:
#     """Phase 2: Return current AuditStatus for polling."""
#     raise NotImplementedError("audit.get_audit_status — implemented in Phase 2")
