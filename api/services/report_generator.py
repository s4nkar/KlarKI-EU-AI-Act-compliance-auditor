"""PDF report generation via Jinja2 + WeasyPrint.

Renders templates/report.html with the ComplianceReport data,
then converts to PDF bytes using WeasyPrint.
Runs in a thread pool to avoid blocking the async event loop.
"""

import asyncio
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader

from models.schemas import ComplianceReport

logger = structlog.get_logger()

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


async def generate_pdf(report: ComplianceReport) -> bytes:
    """Render a compliance report to PDF bytes.

    Loads templates/report.html as a Jinja2 template, renders it with the
    ComplianceReport data, then passes the HTML through WeasyPrint.

    Args:
        report: Fully populated ComplianceReport object.

    Returns:
        PDF as raw bytes, ready to stream as application/pdf.
    """
    pdf_bytes = await asyncio.to_thread(_render_pdf, report)
    logger.info("report_pdf_generated", audit_id=report.audit_id, size=len(pdf_bytes))
    return pdf_bytes


def _render_pdf(report: ComplianceReport) -> bytes:
    """Blocking render: Jinja2 → HTML string → WeasyPrint → PDF bytes."""
    from weasyprint import HTML

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html")
    html_str = template.render(report=report)

    pdf = HTML(string=html_str, base_url=str(_TEMPLATES_DIR)).write_pdf()
    return pdf
