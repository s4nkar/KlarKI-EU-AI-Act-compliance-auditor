"""PDF report generation via Jinja2 + WeasyPrint. (Phase 2)

Renders templates/report.html with the ComplianceReport data,
then converts to PDF bytes using WeasyPrint.
"""

import structlog
from models.schemas import ComplianceReport

logger = structlog.get_logger()


async def generate_pdf(report: ComplianceReport) -> bytes:
    """Render a compliance report to PDF bytes.

    Loads templates/report.html as a Jinja2 template, renders it with the
    ComplianceReport data, then passes the HTML through WeasyPrint.

    Args:
        report: Fully populated ComplianceReport object.

    Returns:
        PDF as raw bytes, ready to stream as application/pdf.
    """
    raise NotImplementedError("report_generator.generate_pdf — implemented in Phase 2")
