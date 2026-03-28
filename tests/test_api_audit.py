"""Integration tests for audit API endpoints."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.mark.asyncio
async def test_health_endpoint(test_client):
    """GET /api/v1/health returns valid response structure."""
    resp = await test_client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "services" in body["data"]
    assert "chromadb" in body["data"]["services"]
    assert "ollama" in body["data"]["services"]


@pytest.mark.asyncio
async def test_upload_no_file_returns_400(test_client):
    """POST /upload with neither file nor raw_text returns 400."""
    resp = await test_client.post("/api/v1/audit/upload")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_upload_unsupported_extension_returns_415(test_client):
    """POST /upload with unsupported file type returns 415."""
    resp = await test_client.post(
        "/api/v1/audit/upload",
        files={"file": ("report.csv", b"col1,col2\n1,2", "text/csv")},
    )
    assert resp.status_code == 415


@pytest.mark.asyncio
async def test_upload_raw_text_returns_audit_id(test_client):
    """POST /upload with raw_text returns audit_id."""
    resp = await test_client.post(
        "/api/v1/audit/upload",
        data={"raw_text": "The system has a risk management process in place."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "audit_id" in body["data"]
    assert len(body["data"]["audit_id"]) == 36  # UUID4


@pytest.mark.asyncio
async def test_get_unknown_audit_returns_404(test_client):
    """GET /audit/{unknown_id} returns 404."""
    resp = await test_client.get("/api/v1/audit/nonexistent-id-1234")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_upload_txt_file_returns_audit_id(test_client):
    """POST /upload with .txt file returns audit_id."""
    content = b"Our AI system monitors data governance practices and maintains logs."
    resp = await test_client.post(
        "/api/v1/audit/upload",
        files={"file": ("policy.txt", content, "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "audit_id" in body["data"]
