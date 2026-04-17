"""Evidence mapping service — document sections → legal obligation coverage.

This is the core Phase 3 audit capability. It answers: for each legal
obligation that applies to this system, does the uploaded document contain
the required evidence artefacts?

How it works:
  1. Load obligation schemas from data/obligations/ JSONL files.
  2. Filter to obligations matching the detected actor type.
  3. For each obligation's evidence_required list, search document chunks
     for keyword signals of that artefact (deterministic, no LLM).
  4. Return EvidenceMap: per-obligation coverage + aggregate stats.

The LLM is used downstream (gap_analyser) for explanation — not here.
This layer is intentionally deterministic so it's auditable and testable.

Evidence keyword matching uses a curated synonym dictionary so that
"risk register" also matches "risk log", "risk inventory", etc.
"""

import json
import re
from pathlib import Path

import structlog

from models.schemas import (
    ActorType,
    DocumentChunk,
    EvidenceItem,
    EvidenceMap,
)

logger = structlog.get_logger()

_OBLIGATIONS_DIR = Path(__file__).parent.parent.parent / "data" / "obligations"

# ── Evidence synonym map ──────────────────────────────────────────────────────
# Maps canonical evidence terms (from JSONL) to synonym patterns that a
# real compliance document might use instead.
_EVIDENCE_SYNONYMS: dict[str, list[str]] = {
    "risk register": [
        "risk register", "risk log", "risk inventory", "risk catalogue",
        "risikokatalog", "risikoregister", "risk list", "risk ledger",
    ],
    "monitoring procedure": [
        "monitoring procedure", "monitoring process", "monitoring plan",
        "post-market monitoring", "surveillance procedure", "monitoring system",
        "überwachungsverfahren", "monitoring framework", "monitoring protocol",
    ],
    "mitigation controls": [
        "mitigation control", "risk mitigation", "control measure",
        "safeguard", "risk treatment", "corrective action", "risk response",
        "risikominderung", "schutzmaßnahme", "mitigation measure",
    ],
    "technical documentation": [
        "technical documentation", "technical file", "technical dossier",
        "technische dokumentation", "system documentation", "technical spec",
        "technical specification", "technical description",
    ],
    "conformity assessment": [
        "conformity assessment", "conformity evaluation", "third-party assessment",
        "notified body assessment", "konformitätsbewertung", "conformity check",
        "third party conformity", "independent assessment",
    ],
    "conformity assessment record": [
        "conformity assessment record", "assessment certificate", "conformity certificate",
        "third-party certificate", "notified body certificate", "audit certificate",
    ],
    "CE marking declaration": [
        "CE marking", "CE mark", "CE declaration", "declaration of conformity",
        "DoC", "konformitätserklärung", "CE-Kennzeichnung",
    ],
    "EU database registration": [
        "EU database", "EU AI database", "EUID", "database registration",
        "AI database registration", "registration number", "AI Act database",
    ],
    "fundamental rights impact assessment": [
        "fundamental rights impact", "FRIA", "rights impact assessment",
        "human rights assessment", "grundrechte-folgenabschätzung",
        "impact assessment fundamental rights",
    ],
    "bias audit": [
        "bias audit", "bias testing", "fairness testing", "bias evaluation",
        "bias assessment", "discrimination testing", "fairness audit",
        "algorithmic bias", "bias mitigation", "fairness check",
    ],
    "data governance documentation": [
        "data governance", "data management", "data quality", "datenverwaltung",
        "data policy", "data handling", "data documentation", "dataset documentation",
    ],
    "human oversight procedure": [
        "human oversight", "human review", "human-in-the-loop", "HITL",
        "manual review", "human supervision", "menschliche aufsicht",
        "human control", "operator oversight", "human check",
    ],
    "transparency notice": [
        "transparency notice", "transparency information", "user notification",
        "disclosure notice", "AI disclosure", "transparency statement",
        "transparenzhinweis", "information notice", "system disclosure",
    ],
    "logging and audit trail": [
        "audit log", "audit trail", "logging", "event log", "system log",
        "activity log", "protokollierung", "log record", "audit record",
        "transaction log", "traceability log",
    ],
    "appeals mechanism": [
        "appeals mechanism", "appeal process", "right to appeal", "complaint mechanism",
        "redress mechanism", "review mechanism", "widerspruchsverfahren",
        "contestation", "challenge procedure",
    ],
    "explainability mechanism": [
        "explainability", "explanation", "interpretability", "xai",
        "model explanation", "decision explanation", "erklärbarkeit",
        "explainable AI", "reason for decision", "decision rationale",
    ],
    "risk management system": [
        "risk management system", "risk management framework", "risk management process",
        "risikomanagementsystem", "risk framework", "risk management plan",
    ],
    "safety validation records": [
        "safety validation", "safety testing", "safety assessment", "safety record",
        "validation record", "safety check", "sicherheitsvalidierung",
    ],
    "incident monitoring procedure": [
        "incident monitoring", "incident response", "incident management",
        "serious incident", "incident report", "störfallmanagement",
        "post-market monitoring", "incident procedure",
    ],
    "worker notification record": [
        "worker notification", "employee notification", "staff notification",
        "worker information", "employee disclosure", "worker awareness",
        "arbeitnehmerbenachrichtigung",
    ],
    "supervisory authority notification": [
        "supervisory authority", "regulatory notification", "authority notification",
        "competent authority", "national authority notification",
        "behördennotifizierung", "notified authority",
    ],
    "judicial independence safeguard documentation": [
        "judicial independence", "court independence", "impartiality safeguard",
        "judicial safeguard", "richterliche unabhängigkeit",
    ],
}


def _build_pattern(synonyms: list[str]) -> re.Pattern:
    """Compile a case-insensitive OR pattern from a synonym list."""
    escaped = [re.escape(s) for s in synonyms]
    return re.compile("|".join(escaped), re.IGNORECASE)


# Pre-compile all patterns at import time
_EVIDENCE_PATTERNS: dict[str, re.Pattern] = {
    term: _build_pattern(syns)
    for term, syns in _EVIDENCE_SYNONYMS.items()
}


def _load_all_obligations() -> list[dict]:
    """Load all obligation JSONL files from data/obligations/."""
    obligations: list[dict] = []
    if not _OBLIGATIONS_DIR.exists():
        logger.warning("obligations_dir_missing", path=str(_OBLIGATIONS_DIR))
        return obligations

    for jsonl_file in _OBLIGATIONS_DIR.rglob("*.jsonl"):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obligations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return obligations


def _evidence_present(evidence_term: str, chunks: list[DocumentChunk]) -> list[str]:
    """Return chunk_ids where evidence_term (or a synonym) appears.

    Uses pre-compiled regex patterns from _EVIDENCE_PATTERNS. Falls back to
    simple substring match if the term has no registered synonyms.
    """
    pattern = _EVIDENCE_PATTERNS.get(evidence_term)
    matched_ids: list[str] = []

    for chunk in chunks:
        if pattern:
            if pattern.search(chunk.text):
                matched_ids.append(chunk.chunk_id)
        else:
            # Fallback: literal substring match (case-insensitive)
            if evidence_term.lower() in chunk.text.lower():
                matched_ids.append(chunk.chunk_id)

    return matched_ids


def map_evidence(
    chunks: list[DocumentChunk],
    actor_type: ActorType,
    applicable_articles: list[int],
) -> EvidenceMap:
    """Map document chunks to legal obligation evidence requirements.

    For each applicable obligation (filtered by actor and applicable_articles),
    checks which evidence artefacts are present in the document and which are
    missing.

    Args:
        chunks: All document chunks from the legal chunker.
        actor_type: Detected actor role — used to filter actor-specific obligations.
        applicable_articles: From ApplicabilityResult — only map obligations for
                             these article numbers. Empty = no obligations apply.

    Returns:
        EvidenceMap with per-obligation EvidenceItem coverage and aggregate stats.
    """
    if not applicable_articles:
        return EvidenceMap(
            total_obligations=0,
            fully_satisfied=0,
            partially_satisfied=0,
            missing=0,
            overall_coverage=0.0,
            items=[],
        )

    all_obligations = _load_all_obligations()
    actor_str = actor_type.value

    # Filter: actor must be in obligation's actor list AND article must be applicable
    # Article number extraction: "Article 9" → 9, "Annex III" → skip article filter
    def _article_num(obligation: dict) -> int | None:
        article_str = obligation.get("article", "")
        m = re.search(r"\d+", article_str)
        return int(m.group()) if m else None

    def _is_applicable(ob: dict) -> bool:
        # Actor filter
        if actor_str not in ob.get("actor", []):
            return False
        # Article applicability filter
        art_num = _article_num(ob)
        if art_num is None:
            return True  # Annex-level obligations always apply if actor matches
        return art_num in applicable_articles

    applicable_obligations = [ob for ob in all_obligations if _is_applicable(ob)]

    items: list[EvidenceItem] = []

    for ob in applicable_obligations:
        evidence_required: list[str] = ob.get("evidence_required", [])
        satisfied_evidence: list[str] = []
        missing_evidence: list[str] = []
        satisfied_chunk_ids: set[str] = set()

        for ev_term in evidence_required:
            found_ids = _evidence_present(ev_term, chunks)
            if found_ids:
                satisfied_evidence.append(ev_term)
                satisfied_chunk_ids.update(found_ids)
            else:
                missing_evidence.append(ev_term)

        total_ev = len(evidence_required)
        coverage = len(satisfied_evidence) / total_ev if total_ev > 0 else 0.0

        items.append(
            EvidenceItem(
                obligation_id=ob.get("id", ""),
                article=ob.get("article", ""),
                requirement=ob.get("requirement", ""),
                evidence_required=evidence_required,
                satisfied_by_chunks=list(satisfied_chunk_ids),
                satisfied_evidence=satisfied_evidence,
                missing_evidence=missing_evidence,
                coverage=round(coverage, 3),
            )
        )

    # Aggregate stats
    total = len(items)
    fully = sum(1 for i in items if i.coverage >= 1.0)
    partially = sum(1 for i in items if 0 < i.coverage < 1.0)
    missing_count = sum(1 for i in items if i.coverage == 0.0)
    overall = sum(i.coverage for i in items) / total if total > 0 else 0.0

    logger.info(
        "evidence_map_complete",
        total_obligations=total,
        fully_satisfied=fully,
        partially_satisfied=partially,
        missing=missing_count,
        overall_coverage=round(overall, 3),
    )

    return EvidenceMap(
        total_obligations=total,
        fully_satisfied=fully,
        partially_satisfied=partially,
        missing=missing_count,
        overall_coverage=round(overall, 3),
        items=items,
    )
