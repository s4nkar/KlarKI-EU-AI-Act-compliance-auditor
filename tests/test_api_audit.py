"""Integration tests for audit API endpoints. (Phase 2)"""

import pytest


@pytest.mark.asyncio
async def test_upload_txt_returns_audit_id(test_client):
    """POST /api/v1/audit/upload with TXT file returns audit_id."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_upload_starts_pipeline(test_client):
    """Upload triggers async pipeline and status transitions to PARSING."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_get_audit_complete(test_client):
    """GET /api/v1/audit/{id} returns ComplianceReport when COMPLETE."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_upload_oversized_file_rejected(test_client):
    """Files over 10MB are rejected with 413."""
    pytest.skip("Implemented in Phase 2")
