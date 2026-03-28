"""Tests for Annex III risk wizard. (Phase 4)"""

import pytest


@pytest.mark.asyncio
async def test_q9_workplace_yields_prohibited():
    """Q9 answered yes → PROHIBITED tier."""
    pytest.skip("Implemented in Phase 4")


@pytest.mark.asyncio
async def test_any_yes_yields_high():
    """Any other yes answer → HIGH tier."""
    pytest.skip("Implemented in Phase 4")


@pytest.mark.asyncio
async def test_all_no_yields_minimal():
    """All no answers → MINIMAL tier."""
    pytest.skip("Implemented in Phase 4")
