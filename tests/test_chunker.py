"""Tests for chunker service."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.mark.asyncio
async def test_chunk_assigns_uuid():
    """Each chunk receives a unique UUID4 chunk_id."""
    from services.chunker import chunk_text

    chunks = await chunk_text("Hello world. " * 100, source_file="test.txt")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique"


@pytest.mark.asyncio
async def test_chunk_size_respected():
    """Chunks do not exceed chunk_size characters (with small tolerance for splitter)."""
    from services.chunker import chunk_text

    long_text = "The system performs data governance analysis. " * 200
    chunks = await chunk_text(long_text, source_file="test.txt", chunk_size=256)
    for chunk in chunks:
        assert len(chunk.text) <= 300, f"Chunk too large: {len(chunk.text)}"


@pytest.mark.asyncio
async def test_chunk_source_file_preserved():
    """Each chunk retains the source_file name."""
    from services.chunker import chunk_text

    chunks = await chunk_text("Test content. " * 50, source_file="policy.pdf")
    for chunk in chunks:
        assert chunk.source_file == "policy.pdf"


@pytest.mark.asyncio
async def test_chunk_index_sequential():
    """chunk_index is sequential starting from 0."""
    from services.chunker import chunk_text

    chunks = await chunk_text("Word. " * 300, source_file="test.txt")
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


@pytest.mark.asyncio
async def test_empty_text_returns_empty():
    """Empty or whitespace-only text returns empty list."""
    from services.chunker import chunk_text

    chunks = await chunk_text("   ", source_file="empty.txt")
    assert chunks == []
