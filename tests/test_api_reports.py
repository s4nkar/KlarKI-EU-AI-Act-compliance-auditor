"""Integration tests for reports API endpoints. (Phase 2)"""

import pytest


@pytest.mark.asyncio
async def test_download_pdf(test_client):
    """GET /api/v1/reports/{id}/pdf returns application/pdf stream."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_download_json(test_client):
    """GET /api/v1/reports/{id}/json returns valid ComplianceReport."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_report_unknown_audit_returns_404(test_client):
    """Requesting report for unknown audit_id returns 404."""
    pytest.skip("Implemented in Phase 2")
