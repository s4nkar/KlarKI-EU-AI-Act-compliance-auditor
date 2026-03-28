"""Tests for document_parser service. (Phase 2)"""

import pytest


@pytest.mark.asyncio
async def test_parse_txt(tmp_path):
    """Parse a plain text file and return its content."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_parse_pdf(tmp_path):
    """Parse a PDF file and return extracted text."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_parse_docx(tmp_path):
    """Parse a DOCX file including German characters."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_unsupported_extension(tmp_path):
    """Raise ValueError for unsupported file types."""
    pytest.skip("Implemented in Phase 2")
