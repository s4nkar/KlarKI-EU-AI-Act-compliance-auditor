"""Multi-Agent LangGraph Workflow for Compliance Gap Analysis.

Replaces the monolithic prompt in gap_analyser.py with a StateGraph
that routes between specialized agents:
1. Legal Agent: Extracts strict requirements from the regulation.
2. Technical Agent: Matches user documentation against those requirements.
3. Synthesis Agent: Drafts the final severity-graded gap report.
"""

import time
import structlog
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from models.schemas import ArticleDomain, DocumentChunk
from services.ollama_client import OllamaClient
from services.monitoring_stats import stats as _monitor
from services.prompt_registry import load_prompt

logger = structlog.get_logger()


class AuditState(TypedDict):
    """The state dictionary passed between nodes in the LangGraph."""
    article_num: int
    domain: ArticleDomain
    user_chunks: list[DocumentChunk]
    regulatory_passages: list[dict]
    ollama_client: OllamaClient
    
    # Populated by Legal Agent
    extracted_requirements: list[str]
    
    # Populated by Technical Agent
    evidence_findings: dict[str, str]
    
    # Populated by Synthesis Agent
    final_score: float
    gaps: list[dict]
    recommendations: list[str]
    reasoning: str


async def legal_agent_node(state: AuditState) -> dict:
    """Extracts strict, actionable requirements from regulatory text."""
    logger.info("agent_graph_node_start", node="legal_agent", article=state["article_num"])
    _t0 = time.time()
    _error = False

    try:
        reg_text = "\n\n".join(
            f"[{p.get('metadata', {}).get('title', p.get('metadata', {}).get('requirement_id', ''))}]\n{p.get('text', '')[:400]}"
            for p in state["regulatory_passages"][:5]
        ) or "(no regulatory passages retrieved)"

        prompt = (
            load_prompt("legal_agent")
            .replace("{{ARTICLE_NUM}}", str(state["article_num"]))
            .replace("{{REG_TEXT}}", reg_text)
        )
        result = await state["ollama_client"].generate_json(prompt)
        reqs = result.get("requirements", [])
        if not isinstance(reqs, list):
            reqs = []
    except Exception as e:
        logger.warning("legal_agent_error", error=str(e))
        reqs = []
        _error = True

    _monitor.record_graph_node("legal_agent", time.time() - _t0, error=_error)
    return {"extracted_requirements": reqs}


async def technical_agent_node(state: AuditState) -> dict:
    """Evaluates user documentation against the legal checklist."""
    logger.info("agent_graph_node_start", node="technical_agent", article=state["article_num"])
    _t0 = time.time()
    _error = False

    reqs = state.get("extracted_requirements", [])
    if not reqs:
        _monitor.record_graph_node("technical_agent", time.time() - _t0, error=False)
        return {"evidence_findings": {"General": "No requirements extracted from regulation."}}
        
    # Sort by text length descending so the most content-rich chunks go first,
    # then take top 15 instead of 10 to give the agent more coverage.
    ranked_chunks = sorted(state["user_chunks"], key=lambda c: len(c.text), reverse=True)
    user_text = "\n\n".join(c.text for c in ranked_chunks[:15]) or "(no relevant documentation found)"
    req_str = "\n".join(f"- {r}" for r in reqs)
    
    prompt = (
        load_prompt("technical_agent")
        .replace("{{REQ_STR}}", req_str)
        .replace("{{USER_TEXT}}", user_text)
    )
    try:
        result = await state["ollama_client"].generate_json(prompt)
        findings = result.get("findings", {})
        if not isinstance(findings, dict):
            findings = {}
    except Exception as e:
        logger.warning("technical_agent_error", error=str(e))
        findings = {}
        _error = True

    _monitor.record_graph_node("technical_agent", time.time() - _t0, error=_error)
    return {"evidence_findings": findings}


async def synthesis_agent_node(state: AuditState) -> dict:
    """Compiles findings into a structured compliance report."""
    logger.info("agent_graph_node_start", node="synthesis_agent", article=state["article_num"])
    _t0 = time.time()
    _error = False

    findings = state.get("evidence_findings", {})
    if not findings:
        findings_str = "No findings could be extracted."
    else:
        # Truncate each finding to 300 chars to keep the prompt within the
        # 2048-token Ollama context window (phi3:mini fails silently on overflow).
        lines = [f"[{req[:60]}]: {str(finding)[:200]}" for req, finding in findings.items()]
        findings_str = "\n".join(lines[:10])  # cap at 10 requirements

    prompt = (
        load_prompt("synthesis_agent")
        .replace("{{ARTICLE_NUM}}", str(state["article_num"]))
        .replace("{{FINDINGS_STR}}", findings_str)
    )
    try:
        # keep_alive="0" unloads the model immediately after this call so the
        # NLI cross-encoder (loaded by evidence_mapper) has room in system RAM.
        result = await state["ollama_client"].generate_json(prompt, keep_alive="0")

        score = float(result.get("score", 50.0))
        reasoning = str(result.get("reasoning", ""))
        gaps = result.get("gaps", [])
        recs = result.get("recommendations", [])

        if not isinstance(gaps, list): gaps = []
        if not isinstance(recs, list): recs = []

    except Exception as e:
        logger.warning("synthesis_agent_error", error=str(e))
        score = 30.0
        reasoning = "LangGraph analysis failed — manual review required."
        gaps = [{"title": "Analysis failed — review manually",
                 "description": f"Gap analysis could not be completed: {str(e)[:200]}",
                 "severity": "major"}]
        recs = ["Retry the audit or review documentation manually."]
        _error = True

    _monitor.record_graph_node("synthesis_agent", time.time() - _t0, error=_error)
    return {
        "final_score": score,
        "reasoning": reasoning,
        "gaps": gaps,
        "recommendations": recs
    }


def build_audit_graph():
    """Build and compile the multi-agent LangGraph workflow."""
    workflow = StateGraph(AuditState)
    
    workflow.add_node("legal", legal_agent_node)
    workflow.add_node("technical", technical_agent_node)
    workflow.add_node("synthesis", synthesis_agent_node)
    
    workflow.add_edge(START, "legal")
    workflow.add_edge("legal", "technical")
    workflow.add_edge("technical", "synthesis")
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()
