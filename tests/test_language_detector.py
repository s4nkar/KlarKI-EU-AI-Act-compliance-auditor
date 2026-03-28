"""Tests for language_detector service."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.mark.asyncio
async def test_detect_english():
    """Detect English text as 'en'."""
    from services.language_detector import detect_language

    text = "This document describes the risk management system for the AI application."
    assert await detect_language(text) == "en"


@pytest.mark.asyncio
async def test_detect_german():
    """Detect German text as 'de'."""
    from services.language_detector import detect_language

    text = (
        "Dieses Dokument beschreibt das Risikomanagementsystem für das KI-System. "
        "Die technische Dokumentation umfasst alle relevanten Anforderungen gemäß "
        "Artikel 9 des EU KI-Gesetzes."
    )
    assert await detect_language(text) == "de"


@pytest.mark.asyncio
async def test_fallback_on_short_text():
    """Very short text defaults to 'en'."""
    from services.language_detector import detect_language

    assert await detect_language("Hi") == "en"
    assert await detect_language("") == "en"


@pytest.mark.asyncio
async def test_non_german_defaults_to_en():
    """Non-German, non-English text maps to 'en'."""
    from services.language_detector import detect_language

    french = "Ce document décrit le système de gestion des risques pour l'IA."
    result = await detect_language(french)
    assert result == "en"  # Only 'de' maps to 'de'; all others → 'en'
