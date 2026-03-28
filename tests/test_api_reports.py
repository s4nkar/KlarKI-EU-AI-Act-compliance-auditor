"""Integration tests for reports API endpoints."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.mark.asyncio
async def test_report_unknown_audit_returns_404(test_client):
    """Requesting report for unknown audit_id returns 404."""
    resp = await test_client.get("/api/v1/reports/nonexistent-123/json")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_report_pdf_unknown_returns_404(test_client):
    """Requesting PDF for unknown audit_id returns 404."""
    resp = await test_client.get("/api/v1/reports/nonexistent-123/pdf")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_report_incomplete_audit_returns_409(test_client):
    """Requesting report for an in-progress audit returns 409."""
    from routers.audit import _audits
    from models.schemas import AuditResponse, AuditStatus

    # Inject a fake in-progress audit
    fake_id = "test-in-progress-audit-id"
    _audits[fake_id] = AuditResponse(audit_id=fake_id, status=AuditStatus.ANALYSING)

    resp = await test_client.get(f"/api/v1/reports/{fake_id}/json")
    assert resp.status_code == 409

    # Cleanup
    del _audits[fake_id]
