"""Tests for language_detector service. (Phase 2)"""

import pytest


@pytest.mark.asyncio
async def test_detect_german():
    """Detect German text as 'de'."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_detect_english():
    """Detect English text as 'en'."""
    pytest.skip("Implemented in Phase 2")


@pytest.mark.asyncio
async def test_fallback_on_short_text():
    """Very short text defaults to 'en'."""
    pytest.skip("Implemented in Phase 2")
