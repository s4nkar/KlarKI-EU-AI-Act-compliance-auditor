"""Tests for the chunk classifier service (Ollama and Triton backends).

All tests use mocked backends — no live Ollama or Triton required.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


# ── _parse_label (pure function) ──────────────────────────────────────────────

def test_parse_label_known_labels():
    """_parse_label maps every valid label string to the correct ArticleDomain."""
    from services.classifier import _parse_label
    from models.schemas import ArticleDomain

    cases = {
        "risk_management":         ArticleDomain.RISK_MANAGEMENT,
        "data_governance":         ArticleDomain.DATA_GOVERNANCE,
        "technical_documentation": ArticleDomain.TECHNICAL_DOCUMENTATION,
        "record_keeping":          ArticleDomain.RECORD_KEEPING,
        "transparency":            ArticleDomain.TRANSPARENCY,
        "human_oversight":         ArticleDomain.HUMAN_OVERSIGHT,
        "security":                ArticleDomain.SECURITY,
        "unrelated":               ArticleDomain.UNRELATED,
    }
    for raw, expected in cases.items():
        assert _parse_label(raw) == expected, f"Failed for label: {raw}"


def test_parse_label_unknown_maps_to_unrelated():
    """Unrecognised or garbage LLM output maps to ArticleDomain.UNRELATED."""
    from services.classifier import _parse_label
    from models.schemas import ArticleDomain

    assert _parse_label("something_random") == ArticleDomain.UNRELATED
    assert _parse_label("") == ArticleDomain.UNRELATED
    assert _parse_label("  ") == ArticleDomain.UNRELATED


def test_parse_label_strips_punctuation_and_whitespace():
    """_parse_label normalises labels that include quotes, spaces, punctuation."""
    from services.classifier import _parse_label
    from models.schemas import ArticleDomain

    assert _parse_label('"risk_management"') == ArticleDomain.RISK_MANAGEMENT
    assert _parse_label("  transparency.") == ArticleDomain.TRANSPARENCY
    assert _parse_label("human-oversight") == ArticleDomain.HUMAN_OVERSIGHT
    assert _parse_label("SECURITY") == ArticleDomain.SECURITY


# ── classify_chunks — Ollama backend (mocked) ─────────────────────────────────

@pytest.mark.asyncio
async def test_classify_chunks_ollama_sets_domain():
    """classify_chunks (Ollama path) sets .domain on every chunk."""
    from services.classifier import classify_chunks
    from models.schemas import ArticleDomain, DocumentChunk

    chunks = [
        DocumentChunk(chunk_id="c1", text="We manage risk.", source_file="f.txt", chunk_index=0),
        DocumentChunk(chunk_id="c2", text="Data is governed.", source_file="f.txt", chunk_index=1),
    ]

    mock_ollama = AsyncMock()
    mock_ollama.generate.side_effect = ["risk_management", "data_governance"]

    with patch("services.classifier.settings") as mock_settings:
        mock_settings.use_triton = False
        result = await classify_chunks(chunks, mock_ollama)

    assert result[0].domain == ArticleDomain.RISK_MANAGEMENT
    assert result[1].domain == ArticleDomain.DATA_GOVERNANCE


@pytest.mark.asyncio
async def test_classify_chunks_ollama_unknown_label_maps_to_unrelated():
    """Unrecognised LLM label is mapped to UNRELATED, not an error."""
    from services.classifier import classify_chunks
    from models.schemas import ArticleDomain, DocumentChunk

    chunk = DocumentChunk(chunk_id="c1", text="Some irrelevant text.", source_file="f.txt", chunk_index=0)

    mock_ollama = AsyncMock()
    mock_ollama.generate.return_value = "definitely_not_a_domain"

    with patch("services.classifier.settings") as mock_settings:
        mock_settings.use_triton = False
        result = await classify_chunks([chunk], mock_ollama)

    assert result[0].domain == ArticleDomain.UNRELATED


@pytest.mark.asyncio
async def test_classify_chunks_ollama_error_falls_back_to_unrelated():
    """If Ollama.generate() raises, the chunk is marked UNRELATED (no crash)."""
    from services.classifier import classify_chunks
    from models.schemas import ArticleDomain, DocumentChunk

    chunk = DocumentChunk(chunk_id="c1", text="Text.", source_file="f.txt", chunk_index=0)

    mock_ollama = AsyncMock()
    mock_ollama.generate.side_effect = ConnectionError("Ollama offline")

    with patch("services.classifier.settings") as mock_settings:
        mock_settings.use_triton = False
        result = await classify_chunks([chunk], mock_ollama)

    assert result[0].domain == ArticleDomain.UNRELATED


@pytest.mark.asyncio
async def test_classify_chunks_empty_list_returns_empty():
    """classify_chunks with no chunks returns an empty list without error."""
    from services.classifier import classify_chunks

    mock_ollama = AsyncMock()

    with patch("services.classifier.settings") as mock_settings:
        mock_settings.use_triton = False
        result = await classify_chunks([], mock_ollama)

    assert result == []
    mock_ollama.generate.assert_not_called()


# ── classify_chunks — Triton backend (mocked) ─────────────────────────────────

@pytest.mark.asyncio
async def test_classify_chunks_triton_backend_used_when_flag_set():
    """When USE_TRITON=true, _classify_triton is called instead of Ollama."""
    from services.classifier import classify_chunks
    from models.schemas import ArticleDomain, DocumentChunk

    chunks = [
        DocumentChunk(chunk_id="c1", text="Security logging.", source_file="f.txt", chunk_index=0),
    ]
    mock_ollama = AsyncMock()

    with patch("services.classifier.settings") as mock_settings, \
         patch("services.classifier._classify_triton", new_callable=AsyncMock) as mock_triton:
        mock_settings.use_triton = True

        async def fake_triton(chunks):
            chunks[0].domain = ArticleDomain.SECURITY
            return chunks

        mock_triton.side_effect = fake_triton
        result = await classify_chunks(chunks, mock_ollama)

    mock_triton.assert_called_once()
    mock_ollama.generate.assert_not_called()
    assert result[0].domain == ArticleDomain.SECURITY
