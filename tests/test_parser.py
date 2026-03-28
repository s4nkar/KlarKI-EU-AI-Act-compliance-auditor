"""Tests for document_parser service."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.mark.asyncio
async def test_parse_txt():
    """Parse a plain text file and return its content."""
    from services.document_parser import parse_document

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as f:
        f.write("This is a test document about risk management.")
        path = f.name

    try:
        text = await parse_document(path, "test.txt")
        assert "risk management" in text
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_parse_txt_german_chars():
    """German characters are preserved in plain text parsing."""
    from services.document_parser import parse_document

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as f:
        f.write("Das System analysiert die Gefühlserkennung der Mitarbeiter.")
        path = f.name

    try:
        text = await parse_document(path, "test.txt")
        assert "Gefühlserkennung" in text
        assert "Mitarbeiter" in text
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_parse_md():
    """Markdown files are parsed as plain text."""
    from services.document_parser import parse_document

    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", encoding="utf-8", delete=False) as f:
        f.write("# AI Policy\n\nThis policy covers data governance.")
        path = f.name

    try:
        text = await parse_document(path, "policy.md")
        assert "data governance" in text
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_unsupported_extension():
    """Raise ValueError for unsupported file types."""
    from services.document_parser import parse_document

    with pytest.raises(ValueError, match="Unsupported file type"):
        await parse_document("/tmp/fake.csv", "fake.csv")
