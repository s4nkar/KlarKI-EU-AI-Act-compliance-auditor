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

    reg_text = "\n\n".join(
        f"[{p['metadata'].get('title', p['metadata'].get('requirement_id', ''))}]\n{p['text']}"
        for p in state["regulatory_passages"][:8]
    ) or "(no regulatory passages retrieved)"
    
    prompt = f"""You are a Legal Expert Agent analyzing the EU AI Act.
Extract a precise checklist of strict, actionable requirements from these regulatory passages for Article {state['article_num']}.
Output a JSON array of strings, where each string is a single requirement.

Passages:
{reg_text}

Output ONLY this JSON format:
{{"requirements": ["...", "..."]}}
"""
    try:
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
    
    prompt = f"""You are a Technical Audit Agent.
Evaluate the following user documentation against this checklist of legal requirements.
For each requirement, state whether evidence is found, partially found, or missing, and provide a brief quote or reason.
Output a JSON object mapping each requirement to your finding.

Requirements:
{req_str}

User Documentation:
{user_text}

Output ONLY this JSON format:
{{"findings": {{"Requirement 1": "Found: XYZ", "Requirement 2": "Missing: Reason"}}}}
"""
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
        findings_str = "\n".join(f"[{req}]: {finding}" for req, finding in findings.items())
    
    prompt = f"""You are a Synthesis Agent compiling a compliance report for Article {state['article_num']}.
Based on the Technical Agent's findings, generate a structured Gap Analysis JSON.
Assign an overall score (0-100), identify gaps with severities (critical, major, minor), and provide actionable recommendations.

Findings:
{findings_str}

Output ONLY this JSON format:
{{
  "score": 75,
  "reasoning": "...",
  "gaps": [
    {{"title": "...", "description": "...", "severity": "major"}}
  ],
  "recommendations": ["...", "..."]
}}
"""
    try:
        result = await state["ollama_client"].generate_json(prompt)
        
        score = float(result.get("score", 50.0))
        reasoning = str(result.get("reasoning", ""))
        gaps = result.get("gaps", [])
        recs = result.get("recommendations", [])
        
        if not isinstance(gaps, list): gaps = []
        if not isinstance(recs, list): recs = []
        
    except Exception as e:
        logger.warning("synthesis_agent_error", error=str(e))
        score = 50.0
        reasoning = "Synthesis failed."
        gaps = []
        recs = []
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
