"""Reports router — PDF and JSON report download endpoints. (Phase 2)

Endpoints:
    GET /api/v1/reports/{audit_id}/pdf   — Stream PDF compliance report
    GET /api/v1/reports/{audit_id}/json  — Return ComplianceReport as JSON
"""

from fastapi import APIRouter
from models.schemas import ComplianceReport

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


# @router.get("/{audit_id}/pdf")
# async def download_pdf(audit_id: str) -> StreamingResponse:
#     """Phase 2: Generate and stream PDF report via WeasyPrint."""
#     raise NotImplementedError("reports.download_pdf — implemented in Phase 2")


# @router.get("/{audit_id}/json")
# async def download_json(audit_id: str) -> ComplianceReport:
#     """Phase 2: Return the raw ComplianceReport JSON for the given audit."""
#     raise NotImplementedError("reports.download_json — implemented in Phase 2")
