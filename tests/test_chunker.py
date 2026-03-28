"""Tests for chunker service. (Phase 2)"""

import pytest


@pytest.mark.asyncio
async def test_chunk_assigns_uuid():
    """Each chunk receives a unique UUID4 chunk_id."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_chunk_size_respected():
    """Chunks do not exceed chunk_size characters."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_chunk_overlap():
    """Adjacent chunks share at least chunk_overlap characters."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_chunk_source_file_preserved():
    """Each chunk retains the source_file name."""
    pytest.skip("Implemented in Phase 2")
