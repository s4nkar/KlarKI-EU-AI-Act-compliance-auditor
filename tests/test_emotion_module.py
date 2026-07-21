"""Tests for emotion recognition module."""

import sys
sys.path.insert(0, "/app")

import pytest
from models.schemas import DocumentChunk, ArticleDomain
from services.emotion_module import check_emotion_recognition


def _chunk(text: str) -> DocumentChunk:
    return DocumentChunk(chunk_id="test", text=text, source_file="test.txt", chunk_index=0)


@pytest.mark.asyncio
async def test_emotion_workplace_is_prohibited():
    """Emotion recognition in workplace context → is_prohibited = True."""
    chunks = [_chunk("We use emotion recognition to monitor employee wellbeing in the workplace.")]
    flag = await check_emotion_recognition(chunks)
    assert flag.detected is True
    assert flag.is_prohibited is True
    assert flag.context == "workplace"


@pytest.mark.asyncio
async def test_emotion_commercial_not_prohibited():
    """Emotion recognition in commercial context → detected, not prohibited."""
    chunks = [_chunk("Our sentiment analysis helps understand customer satisfaction in retail.")]
    flag = await check_emotion_recognition(chunks)
    assert flag.detected is True
    assert flag.is_prohibited is False
    assert flag.context == "commercial"


@pytest.mark.asyncio
async def test_no_emotion_no_flag():
    """Documents without emotion keywords → EmotionFlag.detected = False."""
    chunks = [_chunk("This system processes invoices and generates financial reports.")]
    flag = await check_emotion_recognition(chunks)
    assert flag.detected is False


# ── Consistency with applicability_engine.py (authoritative gate) ────────────


def _applicability(is_prohibited: bool, prohibited_signals: list[str]):
    from models.schemas import ApplicabilityResult
    return ApplicabilityResult(
        is_high_risk=False,
        is_prohibited=is_prohibited,
        prohibited_signals=prohibited_signals,
        applicable_articles=[5] if is_prohibited else [],
        reasoning="test",
    )


@pytest.mark.asyncio
async def test_emotion_flag_anchored_to_applicability_emotion_signal():
    """applicability.is_prohibited=True via an emotion signal → emotion_flag agrees."""
    chunks = [_chunk("We use emotion recognition to monitor employee wellbeing in the workplace.")]
    applicability = _applicability(
        is_prohibited=True,
        prohibited_signals=["emotion recognition (emotion recognition) in workplace/education context"],
    )
    flag = await check_emotion_recognition(chunks, applicability)
    assert flag.is_prohibited is True
    assert flag.context == "workplace"


@pytest.mark.asyncio
async def test_emotion_flag_not_prohibited_when_applicability_prohibits_for_other_reason():
    """applicability.is_prohibited=True for an unrelated signal (e.g. social scoring) must
    not make emotion_flag.is_prohibited True just because emotion+workplace text is also
    present — the two flags must never disagree on *why* a document is prohibited."""
    chunks = [_chunk(
        "We use emotion recognition to monitor employee wellbeing in the workplace, "
        "and also perform social scoring of citizens."
    )]
    applicability = _applicability(is_prohibited=True, prohibited_signals=["social scoring"])
    flag = await check_emotion_recognition(chunks, applicability)
    assert flag.is_prohibited is False
