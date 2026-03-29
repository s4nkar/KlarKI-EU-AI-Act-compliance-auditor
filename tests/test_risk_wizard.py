"""Tests for Annex III risk wizard."""

import sys
sys.path.insert(0, "/app")

import pytest
from models.schemas import RiskTier
from services.risk_wizard import guided_risk_classification

_ALL_NO = {f"q{i}": False for i in range(1, 10)}


@pytest.mark.asyncio
async def test_q9_workplace_yields_prohibited():
    """Q9 answered yes → PROHIBITED tier."""
    answers = {**_ALL_NO, "q9": True}
    tier = await guided_risk_classification(answers)
    assert tier == RiskTier.PROHIBITED


@pytest.mark.asyncio
async def test_any_yes_yields_high():
    """Any q1–q8 yes → HIGH tier."""
    for i in range(1, 9):
        answers = {**_ALL_NO, f"q{i}": True}
        tier = await guided_risk_classification(answers)
        assert tier == RiskTier.HIGH, f"q{i}=True should yield HIGH, got {tier}"


@pytest.mark.asyncio
async def test_all_no_yields_minimal():
    """All no answers → MINIMAL tier."""
    tier = await guided_risk_classification(_ALL_NO)
    assert tier == RiskTier.MINIMAL
