"""Reports router — PDF and JSON report download endpoints.

Endpoints:
    GET /api/v1/reports/{audit_id}/pdf   — Stream PDF compliance report
    GET /api/v1/reports/{audit_id}/json  — Return ComplianceReport as JSON
"""

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import io

from models.schemas import APIResponse, AuditStatus, ComplianceReport
from routers.audit import _audits
from services.report_generator import generate_pdf

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


def _get_completed_report(audit_id: str) -> ComplianceReport:
    """Fetch a completed report or raise appropriate HTTP errors."""
    audit = _audits.get(audit_id)
    if audit is None:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")
    if audit.status != AuditStatus.COMPLETE:
        raise HTTPException(
            status_code=409,
            detail=f"Audit status is '{audit.status.value}', not complete yet.",
        )
    if audit.report is None:
        raise HTTPException(status_code=500, detail="Report data missing for completed audit.")
    return audit.report


@router.get("/{audit_id}/pdf")
async def download_pdf(audit_id: str) -> StreamingResponse:
    """Generate and stream the PDF compliance report.

    Args:
        audit_id: Completed audit identifier.

    Returns:
        StreamingResponse with application/pdf content type.
    """
    report = _get_completed_report(audit_id)
    pdf_bytes = await generate_pdf(report)

    filename = f"klarki_report_{audit_id[:8]}.pdf"
    logger.info("report_pdf_downloaded", audit_id=audit_id)

    return StreamingResponse(
        content=io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{audit_id}/json", response_model=APIResponse)
async def download_json(audit_id: str) -> APIResponse:
    """Return the full ComplianceReport as JSON.

    Args:
        audit_id: Completed audit identifier.

    Returns:
        APIResponse with the ComplianceReport data.
    """
    report = _get_completed_report(audit_id)
    logger.info("report_json_downloaded", audit_id=audit_id)
    return APIResponse(status="success", data=report.model_dump(mode="json"))
