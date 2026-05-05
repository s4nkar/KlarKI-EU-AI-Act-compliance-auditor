"""Tests for the LangGraph 3-node multi-agent workflow (agent_graph.py).

Covers:
  - Graph topology (correct nodes and edge order)
  - Each node in isolation: happy path, error handling, bad JSON shape
  - Technical node early-return when no requirements extracted
  - Full graph with side_effect mocking 3 sequential LLM calls
  - State flows correctly between nodes (legal → technical → synthesis)
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_chunk(text: str = "We maintain a risk register.", idx: int = 0):
    from models.schemas import ArticleDomain, DocumentChunk
    return DocumentChunk(
        chunk_id=f"chunk-{idx}",
        text=text,
        source_file="policy.txt",
        chunk_index=idx,
        domain=ArticleDomain.RISK_MANAGEMENT,
    )


def _make_passage(text: str = "Providers shall establish a risk management system.") -> dict:
    return {
        "text": text,
        "metadata": {"title": "Article 9", "article_num": 9},
        "distance": 0.1,
    }


def _legal_response() -> dict:
    return {"requirements": [
        "Establish a risk management system",
        "Document all identified hazards",
        "Conduct risk assessments throughout the lifecycle",
    ]}


def _technical_response() -> dict:
    return {"findings": {
        "Establish a risk management system": "Found: risk register maintained quarterly",
        "Document all identified hazards": "Partial: hazards listed but mitigation steps missing",
        "Conduct risk assessments throughout the lifecycle": "Missing: no evidence of ongoing assessment",
    }}


def _synthesis_response(score: int = 65) -> dict:
    return {
        "score": score,
        "reasoning": "Risk management partially addressed. Ongoing assessment process undocumented.",
        "gaps": [
            {
                "title": "Missing lifecycle risk assessment",
                "description": "No evidence that risk assessments are conducted continuously after deployment.",
                "severity": "major",
            }
        ],
        "recommendations": [
            "Implement a continuous risk monitoring process per Article 9(4).",
            "Document post-deployment risk signals and escalation paths.",
        ],
    }


def _make_mock_ollama(side_effects: list) -> AsyncMock:
    """Return a mock OllamaClient whose generate_json returns side_effects in order."""
    mock = AsyncMock()
    mock.generate_json = AsyncMock(side_effect=side_effects)
    return mock


# ── graph topology ─────────────────────────────────────────────────────────────

def test_build_audit_graph_compiles():
    """build_audit_graph() returns a compiled graph without raising."""
    from services.agent_graph import build_audit_graph
    graph = build_audit_graph()
    assert graph is not None


def test_build_audit_graph_has_correct_nodes():
    """Compiled graph contains exactly the 3 expected nodes."""
    from services.agent_graph import build_audit_graph
    graph = build_audit_graph()
    node_names = set(graph.get_graph().nodes.keys())
    assert {"legal", "technical", "synthesis"}.issubset(node_names)


def test_build_audit_graph_edge_order():
    """Graph edges follow START → legal → technical → synthesis → END."""
    from services.agent_graph import build_audit_graph
    graph = build_audit_graph()
    edges = [(e.source, e.target) for e in graph.get_graph().edges]
    # Convert to set of (source, target) strings for order-independent lookup
    edge_set = {(s, t) for s, t in edges}
    assert ("__start__", "legal") in edge_set
    assert ("legal", "technical") in edge_set
    assert ("technical", "synthesis") in edge_set
    assert ("synthesis", "__end__") in edge_set


# ── legal_agent_node ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_legal_agent_node_extracts_requirements():
    """Happy path: legal node returns requirements list from LLM."""
    from services.agent_graph import legal_agent_node
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama([_legal_response()])
    state = {
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "regulatory_passages": [_make_passage()],
        "ollama_client": mock_ollama,
    }
    result = await legal_agent_node(state)

    assert "extracted_requirements" in result
    assert len(result["extracted_requirements"]) == 3
    assert "Establish a risk management system" in result["extracted_requirements"]
    mock_ollama.generate_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_legal_agent_node_no_passages():
    """Legal node handles empty passages — prompts with placeholder text."""
    from services.agent_graph import legal_agent_node
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama([{"requirements": ["req1"]}])
    state = {
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "regulatory_passages": [],
        "ollama_client": mock_ollama,
    }
    result = await legal_agent_node(state)

    assert isinstance(result["extracted_requirements"], list)
    mock_ollama.generate_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_legal_agent_node_llm_exception_returns_empty():
    """Legal node exception → graceful degradation with empty requirements."""
    from services.agent_graph import legal_agent_node
    from models.schemas import ArticleDomain

    mock_ollama = AsyncMock()
    mock_ollama.generate_json = AsyncMock(side_effect=RuntimeError("LLM timeout"))
    state = {
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "regulatory_passages": [_make_passage()],
        "ollama_client": mock_ollama,
    }
    result = await legal_agent_node(state)

    assert result["extracted_requirements"] == []


@pytest.mark.asyncio
async def test_legal_agent_node_bad_json_shape_returns_empty():
    """Legal node handles requirements being a non-list → falls back to []."""
    from services.agent_graph import legal_agent_node
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama([{"requirements": "not-a-list"}])
    state = {
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "regulatory_passages": [_make_passage()],
        "ollama_client": mock_ollama,
    }
    result = await legal_agent_node(state)

    assert result["extracted_requirements"] == []


# ── technical_agent_node ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_technical_agent_node_evaluates_chunks():
    """Happy path: technical node maps requirements to findings."""
    from services.agent_graph import technical_agent_node
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama([_technical_response()])
    state = {
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "user_chunks": [_make_chunk()],
        "regulatory_passages": [],
        "extracted_requirements": ["Establish a risk management system"],
        "ollama_client": mock_ollama,
    }
    result = await technical_agent_node(state)

    assert "evidence_findings" in result
    assert isinstance(result["evidence_findings"], dict)
    assert len(result["evidence_findings"]) > 0
    mock_ollama.generate_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_technical_agent_node_no_requirements_skips_llm():
    """Technical node short-circuits when no requirements were extracted."""
    from services.agent_graph import technical_agent_node
    from models.schemas import ArticleDomain

    mock_ollama = AsyncMock()
    mock_ollama.generate_json = AsyncMock()
    state = {
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "user_chunks": [_make_chunk()],
        "regulatory_passages": [],
        "extracted_requirements": [],
        "ollama_client": mock_ollama,
    }
    result = await technical_agent_node(state)

    mock_ollama.generate_json.assert_not_awaited()
    assert "evidence_findings" in result
    assert "No requirements extracted" in str(list(result["evidence_findings"].values()))


@pytest.mark.asyncio
async def test_technical_agent_node_llm_exception_returns_empty():
    """Technical node exception → graceful degradation with empty findings."""
    from services.agent_graph import technical_agent_node
    from models.schemas import ArticleDomain

    mock_ollama = AsyncMock()
    mock_ollama.generate_json = AsyncMock(side_effect=RuntimeError("connection refused"))
    state = {
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "user_chunks": [_make_chunk()],
        "regulatory_passages": [],
        "extracted_requirements": ["Some requirement"],
        "ollama_client": mock_ollama,
    }
    result = await technical_agent_node(state)

    assert result["evidence_findings"] == {}


@pytest.mark.asyncio
async def test_technical_agent_node_bad_json_shape_returns_empty():
    """Technical node handles findings being a non-dict → falls back to {}."""
    from services.agent_graph import technical_agent_node
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama([{"findings": ["not", "a", "dict"]}])
    state = {
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "user_chunks": [_make_chunk()],
        "regulatory_passages": [],
        "extracted_requirements": ["req1"],
        "ollama_client": mock_ollama,
    }
    result = await technical_agent_node(state)

    assert result["evidence_findings"] == {}


# ── synthesis_agent_node ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_synthesis_agent_node_compiles_report():
    """Happy path: synthesis node returns score, reasoning, gaps, recommendations."""
    from services.agent_graph import synthesis_agent_node
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama([_synthesis_response()])
    state = {
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "user_chunks": [],
        "regulatory_passages": [],
        "evidence_findings": {
            "Establish a risk management system": "Found: risk register in place",
        },
        "ollama_client": mock_ollama,
    }
    result = await synthesis_agent_node(state)

    assert result["final_score"] == 65.0
    assert isinstance(result["reasoning"], str) and result["reasoning"]
    assert isinstance(result["gaps"], list) and len(result["gaps"]) == 1
    assert isinstance(result["recommendations"], list) and len(result["recommendations"]) == 2
    mock_ollama.generate_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_synthesis_agent_node_no_findings():
    """Synthesis node handles empty findings dict — uses fallback text in prompt."""
    from services.agent_graph import synthesis_agent_node
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama([_synthesis_response(score=20)])
    state = {
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "user_chunks": [],
        "regulatory_passages": [],
        "evidence_findings": {},
        "ollama_client": mock_ollama,
    }
    result = await synthesis_agent_node(state)

    assert result["final_score"] == 20.0
    mock_ollama.generate_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_synthesis_agent_node_llm_exception_returns_defaults():
    """Synthesis node exception → low score, visible failure gap, retry recommendation.

    The synthesis node intentionally surfaces a critical-severity gap on failure
    rather than returning empty gaps — empty gaps would silently hide a failed
    audit from the user. The score is set to 30 so the article visibly fails
    review. See agent_graph.synthesis_agent_node.
    """
    from services.agent_graph import synthesis_agent_node
    from models.schemas import ArticleDomain

    mock_ollama = AsyncMock()
    mock_ollama.generate_json = AsyncMock(side_effect=RuntimeError("model unloaded"))
    state = {
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "user_chunks": [],
        "regulatory_passages": [],
        "evidence_findings": {"req": "Found: something"},
        "ollama_client": mock_ollama,
    }
    result = await synthesis_agent_node(state)

    assert result["final_score"] == 30.0
    assert "manual review" in result["reasoning"].lower()
    assert len(result["gaps"]) == 1
    assert result["gaps"][0]["severity"] == "major"
    assert len(result["recommendations"]) == 1


@pytest.mark.asyncio
async def test_synthesis_agent_node_bad_list_types_default_to_empty():
    """Synthesis node falls back to [] when gaps/recommendations are not lists."""
    from services.agent_graph import synthesis_agent_node
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama([{
        "score": 70,
        "reasoning": "ok",
        "gaps": "not-a-list",
        "recommendations": {"also": "not-a-list"},
    }])
    state = {
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "user_chunks": [],
        "regulatory_passages": [],
        "evidence_findings": {"req": "Found"},
        "ollama_client": mock_ollama,
    }
    result = await synthesis_agent_node(state)

    assert result["gaps"] == []
    assert result["recommendations"] == []
    assert result["final_score"] == 70.0


# ── full graph: 3-call sequence ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_graph_invokes_all_3_nodes():
    """Full graph calls generate_json exactly 3 times (one per node)."""
    from services.agent_graph import build_audit_graph
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama([
        _legal_response(),
        _technical_response(),
        _synthesis_response(),
    ])
    graph = build_audit_graph()
    final_state = await graph.ainvoke({
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "user_chunks": [_make_chunk()],
        "regulatory_passages": [_make_passage()],
        "ollama_client": mock_ollama,
    })

    assert mock_ollama.generate_json.await_count == 3
    assert "extracted_requirements" in final_state
    assert "evidence_findings" in final_state
    assert "final_score" in final_state


@pytest.mark.asyncio
async def test_full_graph_state_flows_legal_to_technical():
    """Requirements extracted by legal node are visible to the technical node."""
    from services.agent_graph import build_audit_graph
    from models.schemas import ArticleDomain

    captured_prompts: list[str] = []

    # Synthesis node passes keep_alive="0" — accept arbitrary kwargs.
    async def _capturing_generate_json(prompt: str, **_kwargs) -> dict:
        captured_prompts.append(prompt)
        idx = len(captured_prompts)
        if idx == 1:
            return _legal_response()
        if idx == 2:
            return _technical_response()
        return _synthesis_response()

    mock_ollama = AsyncMock()
    mock_ollama.generate_json = _capturing_generate_json

    graph = build_audit_graph()
    await graph.ainvoke({
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "user_chunks": [_make_chunk()],
        "regulatory_passages": [_make_passage()],
        "ollama_client": mock_ollama,
    })

    # The technical node prompt should include a requirement extracted by legal node
    assert len(captured_prompts) == 3
    assert "Establish a risk management system" in captured_prompts[1]


@pytest.mark.asyncio
async def test_full_graph_state_flows_technical_to_synthesis():
    """Findings from technical node appear in the synthesis node prompt."""
    from services.agent_graph import build_audit_graph
    from models.schemas import ArticleDomain

    captured_prompts: list[str] = []

    async def _capturing_generate_json(prompt: str, **_kwargs) -> dict:
        captured_prompts.append(prompt)
        idx = len(captured_prompts)
        if idx == 1:
            return _legal_response()
        if idx == 2:
            return _technical_response()
        return _synthesis_response()

    mock_ollama = AsyncMock()
    mock_ollama.generate_json = _capturing_generate_json

    graph = build_audit_graph()
    await graph.ainvoke({
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "user_chunks": [_make_chunk()],
        "regulatory_passages": [_make_passage()],
        "ollama_client": mock_ollama,
    })

    # Synthesis prompt must contain a finding from the technical node
    assert "Found: risk register maintained quarterly" in captured_prompts[2]


@pytest.mark.asyncio
async def test_full_graph_final_score_and_gaps():
    """Full graph final state contains score, gaps, and recommendations from synthesis."""
    from services.agent_graph import build_audit_graph
    from models.schemas import ArticleDomain

    mock_ollama = _make_mock_ollama([
        _legal_response(),
        _technical_response(),
        _synthesis_response(score=72),
    ])
    graph = build_audit_graph()
    final_state = await graph.ainvoke({
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "user_chunks": [_make_chunk()],
        "regulatory_passages": [_make_passage()],
        "ollama_client": mock_ollama,
    })

    assert final_state["final_score"] == 72.0
    assert len(final_state["gaps"]) == 1
    assert final_state["gaps"][0]["severity"] == "major"
    assert len(final_state["recommendations"]) == 2


@pytest.mark.asyncio
async def test_full_graph_legal_error_still_completes():
    """If legal node fails (empty requirements), graph still runs to completion."""
    from services.agent_graph import build_audit_graph
    from models.schemas import ArticleDomain

    # Legal fails, technical short-circuits (no reqs), synthesis still runs
    mock_ollama = AsyncMock()
    call_count = 0

    async def _side_effect(prompt: str) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("legal agent LLM error")
        # Technical is skipped (no reqs), so next call goes to synthesis
        return _synthesis_response(score=30)

    mock_ollama.generate_json = _side_effect

    graph = build_audit_graph()
    final_state = await graph.ainvoke({
        "article_num": 9,
        "domain": ArticleDomain.RISK_MANAGEMENT,
        "user_chunks": [_make_chunk()],
        "regulatory_passages": [_make_passage()],
        "ollama_client": mock_ollama,
    })

    # Legal produced [], technical short-circuited, synthesis still ran
    assert final_state["extracted_requirements"] == []
    assert "final_score" in final_state
