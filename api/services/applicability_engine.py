"""Article 6 + Annex III applicability gate.

This is the legal decision hierarchy entry point for Phase 3. It answers
the threshold question before any gap analysis runs:

  Does EU AI Act Chapter III Section 2 (Articles 9–15) apply?

Decision tree (matching the legal hierarchy):
  1. Article 5 prohibited practice detected → PROHIBITED, stop.
  2. Annex III category matched → HIGH_RISK, Articles 9–15 apply.
  3. Article 6(1) safety-component trigger → HIGH_RISK, Articles 9–15 apply.
  4. No match → MINIMAL/LIMITED risk, Articles 9–15 do not apply.

All matching is pattern-based (deterministic). No LLM is called here.
The LLM is used downstream only for explanation and gap analysis.

Data source: data/obligations/eu_ai_act/article_6_annex_iii.jsonl
"""

import json
import re
from pathlib import Path

import structlog

from models.schemas import (
    AnnexIIICategory,
    AnnexIIIMatch,
    ApplicabilityResult,
)
from services.ml_classifiers import predict_high_risk as _ml_high_risk
from services.ml_classifiers import predict_prohibited as _ml_prohibited

# ML confidence threshold — if ML model is confident, augment pattern result
_ML_CONFIDENCE_THRESHOLD = 0.85

logger = structlog.get_logger()

_OBLIGATIONS_PATH = (
    Path(__file__).parent.parent.parent
    / "data" / "obligations" / "eu_ai_act" / "article_6_annex_iii.jsonl"
)

# ── Article 5 prohibited practice patterns ────────────────────────────────────
# These are checked first — prohibited outranks high-risk in the decision tree.
_PROHIBITED_PATTERNS: list[tuple[str, str]] = [
    (r"\bsubliminal\s+technique[s]?\b", "subliminal techniques"),
    (r"\bsocial\s+scor(?:ing|e)\b", "social scoring"),
    (r"\breal[-\s]time\s+(?:remote\s+)?biometric\s+identification\b",
     "real-time biometric identification"),
    (r"\bemotion\s+recogni(?:tion|se|ze)\b.*\b(?:workplace|school|education|employee|employees|worker|workers|staff)\b",
     "emotion recognition in workplace/education/employment"),
    (r"\b(?:workplace|school|education|employee|employees|worker|workers|staff)\b.*\bemotion\s+recogni(?:tion|se|ze)\b",
     "emotion recognition in workplace/education/employment"),
    (r"\bEchtzeitbiometrie\b", "Echtzeitbiometrie (DE)"),
    (r"\bSocial[-\s]Scoring\b", "Social-Scoring (DE)"),
    (r"\bunterbewusste\s+Techniken\b", "unterbewusste Techniken (DE)"),
    (r"\bEmotionserkennung\b.*\b(?:Arbeitsplatz|Bildungseinrichtung|Schule)\b",
     "Emotionserkennung am Arbeitsplatz/Bildungseinrichtung (DE)"),
]

# ── Annex III category patterns ───────────────────────────────────────────────
# Keyed by AnnexIIICategory. Each tuple: (pattern, human-readable label).
_ANNEX_III_PATTERNS: dict[AnnexIIICategory, list[tuple[str, str]]] = {
    AnnexIIICategory.BIOMETRIC: [
        (r"\bbiometric\s+(?:identification|recognition|categoris(?:ation|ing))\b",
         "biometric identification/recognition/categorisation"),
        (r"\bfacial\s+recogni(?:tion|se|ze)\b", "facial recognition"),
        (r"\bfingerprint\s+(?:recognition|scan|identification)\b", "fingerprint recognition/scan"),
        (r"\biris\s+(?:scan|recognition)\b", "iris scan/recognition"),
        (r"\bgait\s+(?:recognition|analysis)\b", "gait recognition/analysis"),
        (r"\bvoice\s+(?:identification|recognition)\b", "voice identification/recognition"),
        (r"\bbiometrische\s+(?:Identifizierung|Kategorisierung|Erkennung)\b",
         "biometrische Identifizierung/Kategorisierung (DE)"),
        (r"\bGesichtserkennung\b", "Gesichtserkennung (DE)"),
    ],
    AnnexIIICategory.CRITICAL_INFRASTRUCTURE: [
        (r"\bcritical\s+infrastructure\b", "critical infrastructure"),
        (r"\broad\s+traffic\s+(?:management|control)\b", "road traffic management/control"),
        (r"\bwater\s+supply\s+(?:management|control|network)\b", "water supply management"),
        (r"\belectricity\s+(?:grid|network|supply)\b", "electricity grid/network/supply"),
        (r"\bgas\s+(?:network|supply|distribution)\b", "gas network/supply"),
        (r"\bheating\s+(?:network|supply|distribution)\b", "heating network/supply"),
        (r"\bSCADA\b", "SCADA system"),
        (r"\bindustrial\s+control\s+system\b", "industrial control system"),
        (r"\bkritische\s+Infrastruktur\b", "kritische Infrastruktur (DE)"),
        (r"\bStromversorgung\b", "Stromversorgung (DE)"),
    ],
    AnnexIIICategory.EDUCATION: [
        (r"\bstudent\s+(?:admission|selection|assessment|evaluation|grading)\b",
         "student admission/selection/assessment/evaluation/grading"),
        (r"\beducational\s+(?:assessment|institution|access)\b",
         "educational assessment/institution/access"),
        (r"\bexam\s+(?:monitoring|proctoring|surveillance)\b",
         "exam monitoring/proctoring/surveillance"),
        (r"\blearning\s+outcome[s]?\s+(?:evaluation|assessment)\b",
         "learning outcomes evaluation/assessment"),
        (r"\bvocational\s+training\s+(?:assignment|access)\b",
         "vocational training assignment/access"),
        (r"\bacademic\s+performance\s+(?:prediction|scoring)\b",
         "academic performance prediction/scoring"),
        (r"\bStudienzulassung\b", "Studienzulassung (DE)"),
        (r"\bPrüfungsüberwachung\b", "Prüfungsüberwachung (DE)"),
        (r"\bBildungszugang\b", "Bildungszugang (DE)"),
    ],
    AnnexIIICategory.EMPLOYMENT: [
        (r"\brecruitment\s+(?:AI|automation|system|software|decision)\b",
         "recruitment AI/automation/system/decision"),
        (r"\bhiring\s+(?:decision|algorithm|AI|system)\b",
         "hiring decision/algorithm/AI/system"),
        (r"\bjob\s+application\s+(?:screening|filtering|ranking)\b",
         "job application screening/filtering/ranking"),
        (r"\bcandidate\s+(?:evaluation|ranking|scoring|selection)\b",
         "candidate evaluation/ranking/scoring/selection"),
        (r"\bCV\s+(?:screening|parsing|ranking)\b", "CV screening/parsing/ranking"),
        (r"\bresume\s+(?:screening|parsing|ranking)\b", "resume screening/parsing/ranking"),
        (r"\bperformance\s+(?:monitoring|scoring|evaluation)\s+(?:AI|system|algorithm)\b",
         "performance monitoring/scoring AI/system/algorithm"),
        (r"\bworkforce\s+(?:management|optimisation|AI)\b",
         "workforce management/optimisation AI"),
        (r"\bemployee\s+(?:promotion|termination)\s+(?:decision|AI|algorithm)\b",
         "employee promotion/termination decision AI"),
        (r"\bPersonalentscheidung\b", "Personalentscheidung (DE)"),
        (r"\bBewerbungsscreening\b", "Bewerbungsscreening (DE)"),
        (r"\bLeistungsüberwachung\b", "Leistungsüberwachung (DE)"),
    ],
    AnnexIIICategory.ESSENTIAL_SERVICES: [
        (r"\bcredit\s+scor(?:ing|e)\b", "credit scoring/score"),
        (r"\bcreditworthiness\s+(?:assessment|evaluation|determination)\b",
         "creditworthiness assessment/evaluation"),
        (r"\bloan\s+(?:decision|application|approval)\s+(?:AI|algorithm|automation)\b",
         "loan decision/application AI/algorithm"),
        (r"\bsocial\s+(?:benefit[s]?|assistance|welfare)\s+(?:eligibility|assessment)\b",
         "social benefit/assistance/welfare eligibility"),
        (r"\bhealthcare\s+(?:access|eligibility|triage|AI)\b",
         "healthcare access/eligibility/triage AI"),
        (r"\binsurance\s+(?:pricing|risk\s+assessment|AI)\b",
         "insurance pricing/risk assessment AI"),
        (r"\bpublic\s+service\s+eligibility\b", "public service eligibility"),
        (r"\bKreditwürdigkeitsprüfung\b", "Kreditwürdigkeitsprüfung (DE)"),
        (r"\bSozialleistungen\b.*\bKI\b", "Sozialleistungen KI (DE)"),
    ],
    AnnexIIICategory.LAW_ENFORCEMENT: [
        (r"\blaw\s+enforcement\s+(?:AI|system|algorithm|tool)\b",
         "law enforcement AI/system/algorithm/tool"),
        (r"\bpredictive\s+polic(?:ing|e)\b", "predictive policing/police"),
        (r"\bcrime\s+(?:prediction|prevention\s+AI|analytics\s+AI)\b",
         "crime prediction/prevention/analytics AI"),
        (r"\bcriminal\s+(?:profiling|risk\s+assessment|recidivism)\b",
         "criminal profiling/risk assessment/recidivism"),
        (r"\bforensic\s+(?:AI|analysis\s+AI)\b", "forensic AI/analysis AI"),
        (r"\bevidence\s+(?:evaluation|assessment)\s+(?:AI|algorithm)\b",
         "evidence evaluation/assessment AI/algorithm"),
        (r"\bStrafverfolgung\b.*\bKI\b", "Strafverfolgung KI (DE)"),
        (r"\bkriminelle\s+Profilierung\b", "kriminelle Profilierung (DE)"),
    ],
    AnnexIIICategory.MIGRATION: [
        (r"\basylum\s+(?:application|assessment|decision|AI)\b",
         "asylum application/assessment/decision AI"),
        (r"\bvisa\s+(?:processing|decision|application)\s+(?:AI|algorithm|system)\b",
         "visa processing/decision AI/algorithm/system"),
        (r"\bborder\s+(?:control|management)\s+(?:AI|system)\b",
         "border control/management AI/system"),
        (r"\bmigration\s+(?:management|risk\s+assessment|AI)\b",
         "migration management/risk assessment AI"),
        (r"\brefugee\s+(?:assessment|screening|decision)\b",
         "refugee assessment/screening/decision"),
        (r"\bresidence\s+permit\s+(?:decision|assessment|AI)\b",
         "residence permit decision/assessment AI"),
        (r"\bAsylantrag\b.*\bKI\b", "Asylantrag KI (DE)"),
        (r"\bGrenzkontrolle\b.*\bKI\b", "Grenzkontrolle KI (DE)"),
    ],
    AnnexIIICategory.JUSTICE: [
        (r"\bjudicial\s+(?:decision|AI|support|assistance)\b",
         "judicial decision/AI/support/assistance"),
        (r"\bcourt\s+(?:AI|decision\s+support|algorithm)\b",
         "court AI/decision support/algorithm"),
        (r"\blegal\s+(?:research\s+AI|decision\s+AI|judgment\s+AI)\b",
         "legal research/decision/judgment AI"),
        (r"\belection\s+(?:influence|manipulation|AI)\b",
         "election influence/manipulation AI"),
        (r"\bvoting\s+(?:behaviour|influence|AI)\b",
         "voting behaviour/influence AI"),
        (r"\bdemocratic\s+process(?:es)?\s+(?:AI|influence)\b",
         "democratic process AI/influence"),
        (r"\bJustiz[-\s]KI\b", "Justiz-KI (DE)"),
        (r"\bWahlbeeinflussung\b", "Wahlbeeinflussung (DE)"),
    ],
}

# Annex I safety-component signals (Article 6(1))
_ANNEX_I_PATTERNS: list[tuple[str, str]] = [
    (r"\bsafety\s+component\b", "safety component"),
    (r"\bCE\s+marking\b", "CE marking"),
    (r"\bCE[-\s]Kennzeichnung\b", "CE-Kennzeichnung (DE)"),
    (r"\bMedical\s+Device\s+Regulation\b", "Medical Device Regulation"),
    (r"\bMDR\b", "MDR"),
    (r"\bIVDR\b", "IVDR"),
    (r"\bClass\s+II[ab]?\b", "Class IIa/IIb device class"),
    (r"\bClass\s+III\b", "Class III device class"),
    (r"\bnotified\s+body\b", "notified body"),
    (r"\bbenannte\s+Stelle\b", "benannte Stelle (DE)"),
    (r"\bconformity\s+assessment\s+(?:by\s+a\s+)?third\s+party\b",
     "third-party conformity assessment"),
    (r"\baviation\s+safety\s+(?:AI|component|system)\b", "aviation safety AI/component"),
    (r"\brailway\s+safety\s+(?:AI|component|system)\b", "railway safety AI/component"),
    (r"\bmachinery\s+directive\b", "machinery directive"),
]

_CATEGORY_NAMES: dict[AnnexIIICategory, str] = {
    AnnexIIICategory.BIOMETRIC: "Biometric Identification and Categorisation",
    AnnexIIICategory.CRITICAL_INFRASTRUCTURE: "Critical Infrastructure Management",
    AnnexIIICategory.EDUCATION: "Education and Vocational Training",
    AnnexIIICategory.EMPLOYMENT: "Employment and Workers Management",
    AnnexIIICategory.ESSENTIAL_SERVICES: "Essential Private/Public Services",
    AnnexIIICategory.LAW_ENFORCEMENT: "Law Enforcement",
    AnnexIIICategory.MIGRATION: "Migration, Asylum and Border Control",
    AnnexIIICategory.JUSTICE: "Administration of Justice and Democratic Processes",
}

_ARTICLES_9_TO_15 = [9, 10, 11, 12, 13, 14, 15]


def _load_obligations() -> dict[int, dict]:
    """Load Article 6 + Annex III obligation schemas keyed by annex_category."""
    obligations: dict[int, dict] = {}
    if not _OBLIGATIONS_PATH.exists():
        logger.warning("obligations_file_missing", path=str(_OBLIGATIONS_PATH))
        return obligations
    with open(_OBLIGATIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obligations[obj["annex_category"]] = obj
            except (json.JSONDecodeError, KeyError):
                continue
    return obligations


def check_applicability(chunks: list) -> ApplicabilityResult:
    """Determine whether EU AI Act Articles 9–15 apply to the uploaded document.

    Implements the legal decision hierarchy:
      1. Article 5 (prohibited) — checked first, highest priority.
      2. Annex III categories — pattern match across all chunks.
      3. Article 6(1) Annex I safety component signals.
      4. If nothing matches — MINIMAL risk, Articles 9–15 do not apply.

    Args:
        chunks: List of DocumentChunk objects from the chunker service.

    Returns:
        ApplicabilityResult with is_high_risk, is_prohibited, matched categories,
        applicable_articles, and reasoning.
    """
    obligations = _load_obligations()
    full_text = " ".join(c.text for c in chunks)

    # ── Step 1: Article 5 prohibited practice check ───────────────────────────
    # ML model augments pattern matching when available.
    # Either ML (high confidence) OR pattern match triggers prohibited.
    prohibited_hits: list[str] = []
    for pattern, label in _PROHIBITED_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            prohibited_hits.append(label)

    ml_prohibited = _ml_prohibited(full_text[:2000])  # truncate for inference speed
    if ml_prohibited and ml_prohibited.label == "prohibited" and ml_prohibited.confidence >= _ML_CONFIDENCE_THRESHOLD:
        prohibited_hits.append(f"ML model ({ml_prohibited.confidence:.0%} confidence)")

    if prohibited_hits:
        return ApplicabilityResult(
            is_high_risk=False,
            is_prohibited=True,
            annex_iii_matches=[],
            annex_i_triggered=False,
            applicable_articles=[5],
            reasoning=(
                f"PROHIBITED PRACTICE DETECTED under Article 5. "
                f"Signals found: {', '.join(prohibited_hits)}. "
                "Deployment of this AI system is unlawful under the EU AI Act. "
                "Articles 9–15 do not apply — the system must not be placed on the market."
            ),
        )

    # ── Step 2: Annex III category matching ───────────────────────────────────
    annex_iii_matches: list[AnnexIIIMatch] = []
    for category, patterns in _ANNEX_III_PATTERNS.items():
        matched_keywords: list[str] = []
        for pattern, label in patterns:
            if re.search(pattern, full_text, re.IGNORECASE):
                matched_keywords.append(label)

        if matched_keywords:
            obligation = obligations.get(category.value, {})
            annex_iii_matches.append(
                AnnexIIIMatch(
                    category=category,
                    category_name=_CATEGORY_NAMES[category],
                    matched_keywords=matched_keywords,
                    obligation_id=obligation.get("id", f"AIACT_ART6_ANNEXIII_CAT{category.value}"),
                    evidence_required=obligation.get("evidence_required", []),
                )
            )

    # ── Step 3: Article 6(1) Annex I safety-component check ──────────────────
    annex_i_hits: list[str] = []
    for pattern, label in _ANNEX_I_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            annex_i_hits.append(label)
    annex_i_triggered = len(annex_i_hits) >= 2  # require at least 2 signals to avoid false positives

    # ── Step 4: ML high-risk augmentation ────────────────────────────────────
    # ML model can catch Annex III cases where keyword patterns missed.
    ml_risk = _ml_high_risk(full_text[:2000])
    ml_high_risk_triggered = (
        ml_risk is not None
        and ml_risk.label == "high_risk"
        and ml_risk.confidence >= _ML_CONFIDENCE_THRESHOLD
    )

    is_high_risk = bool(annex_iii_matches) or annex_i_triggered or ml_high_risk_triggered

    # ── Build reasoning ───────────────────────────────────────────────────────
    reasoning_parts: list[str] = []
    if annex_iii_matches:
        cats = [m.category_name for m in annex_iii_matches]
        reasoning_parts.append(
            f"HIGH-RISK: Annex III categories matched: {', '.join(cats)}. "
            "Full Articles 9–15 compliance obligations apply."
        )
    if annex_i_triggered:
        reasoning_parts.append(
            f"HIGH-RISK: Article 6(1) safety-component signals detected: "
            f"{', '.join(annex_i_hits[:3])}. Third-party conformity assessment required."
        )
    if ml_high_risk_triggered and not annex_iii_matches and not annex_i_triggered:
        reasoning_parts.append(
            f"HIGH-RISK: ML risk classifier detected high-risk use case "
            f"({ml_risk.confidence:.0%} confidence) — no keyword match found. "  # type: ignore[union-attr]
            "Manual review of Annex III category is recommended."
        )
    if not is_high_risk:
        reasoning_parts.append(
            "No Annex III categories or Article 6(1) safety-component signals found. "
            "System does not appear to be high-risk under the EU AI Act. "
            "Article 52 (limited risk transparency) or no obligations may apply. "
            "If your use case is not reflected in the document, review manually."
        )

    applicable_articles = _ARTICLES_9_TO_15 if is_high_risk else []

    logger.info(
        "applicability_check_complete",
        is_high_risk=is_high_risk,
        is_prohibited=False,
        annex_iii_categories=[m.category.value for m in annex_iii_matches],
        annex_i_triggered=annex_i_triggered,
    )

    return ApplicabilityResult(
        is_high_risk=is_high_risk,
        is_prohibited=False,
        annex_iii_matches=annex_iii_matches,
        annex_i_triggered=annex_i_triggered,
        applicable_articles=applicable_articles,
        reasoning=" ".join(reasoning_parts),
    )
