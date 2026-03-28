"""Tests for compliance scorer. (Phase 2)"""

import pytest


@pytest.mark.asyncio
async def test_overall_score_weighted_average():
    """Overall score is the correct weighted average of article scores."""
    pytest.skip("Implemented in Phase 2")


def test_classify_risk_tier_prohibited():
    """Biometric real-time keyword yields PROHIBITED tier."""
    pytest.skip("Implemented in Phase 2")


def test_classify_risk_tier_high():
    """Recruitment keyword yields HIGH tier."""
    pytest.skip("Implemented in Phase 2")


def test_classify_risk_tier_minimal():
    """No keywords yields MINIMAL tier."""
    pytest.skip("Implemented in Phase 2")
