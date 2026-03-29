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
