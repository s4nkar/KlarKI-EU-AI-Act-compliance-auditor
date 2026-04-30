"""Article 3 actor role classifier.

Pattern-based detection of provider / deployer / importer / distributor from
document text. No LLM required — deterministic keyword + phrase matching.

Legal basis: EU AI Act Article 3(3)–(7).

Decision logic:
  1. Provider signals (strongest): developed, built, trained, placed on market, our model
  2. Deployer signals: use/deploy third-party AI, implemented vendor system
  3. Importer signals: imported AI, third-party system from outside EU
  4. Distributor signals: resell, make available, distribute AI product
  5. Default: DEPLOYER (most SMEs are deployers, not providers)

Confidence is the ratio of matched signals to total signals checked, capped by
the strength of the winning class relative to all classes.
"""

import re
from models.schemas import ActorClassification, ActorType
from services.ml_classifiers import predict_actor as _ml_predict_actor

# ML confidence threshold — if ML model is confident enough, skip pattern matching
_ML_CONFIDENCE_THRESHOLD = 0.80

# ── Signal dictionaries ────────────────────────────────────────────────────────
# Each list contains regex patterns (case-insensitive). More specific patterns
# are weighted higher by appearing multiple times or being listed first.

_PROVIDER_PATTERNS: list[tuple[str, str]] = [
    (r"\bwe\s+(?:have\s+)?developed\b", "we (have) developed"),
    (r"\bwe\s+(?:have\s+)?built\b", "we (have) built"),
    (r"\bwe\s+(?:have\s+)?trained\b", "we (have) trained"),
    (r"\bour\s+(?:own\s+)?(?:AI\s+)?model\b", "our (own) (AI) model"),
    (r"\bour\s+(?:own\s+)?(?:AI\s+)?system\b", "our (own) (AI) system"),
    (r"\bplace[sd]?\b.{0,50}\bon\s+the\s+market\b", "placed on the market"),
    (r"\bin\s+Verkehr\s+gebracht\b", "in Verkehr gebracht"),  # DE: placed on market
    (r"\bwir\s+(?:haben\s+)?entwickelt\b", "wir (haben) entwickelt"),  # DE: we developed
    (r"\bproprietary\s+(?:AI|model|algorithm)\b", "proprietary AI/model/algorithm"),
    (r"\bwe\s+(?:are\s+the\s+)?provider\b", "we (are the) provider"),
    (r"\bAnbieter\b", "Anbieter"),  # DE: provider
    (r"\bour\s+(?:machine\s+learning|deep\s+learning|neural\s+network)\b", "our ML/DL/NN"),
    (r"\bwe\s+(?:designed|created|produced)\s+(?:an?\s+|the\s+)?AI\b", "we designed/created/produced AI"),
    (r"\bintellectual\s+property\s+(?:in|of)\s+(?:the\s+)?(?:AI|model|system)\b", "IP in AI/model/system"),
]

_DEPLOYER_PATTERNS: list[tuple[str, str]] = [
    (r"\bwe\s+use\s+(?:an?\s+|the\s+)?AI\b", "we use (an/the) AI"),
    (r"\bwe\s+(?:have\s+)?implemented\b", "we (have) implemented"),
    (r"\bwe\s+(?:have\s+)?deployed\b", "we (have) deployed"),
    (r"\bwe\s+(?:are\s+(?:using|deploying))\b", "we are using/deploying"),
    (r"\bwir\s+(?:nutzen|verwenden|einsetzen|betreiben)\b", "wir nutzen/verwenden/einsetzen"),  # DE
    (r"\bBetreiber\b", "Betreiber"),  # DE: deployer/operator
    (r"\bthird[-\s]party\s+(?:AI|model|system|solution)\b", "third-party AI/model/system/solution"),
    (r"\bpurchased\s+(?:from|an?\s+AI)\b", "purchased from/an AI"),
    (r"\blicensed\s+(?:AI|model|system|software)\b", "licensed AI/model/system/software"),
    (r"\bvendor(?:'s)?\s+(?:AI|model|system|solution)\b", "vendor's AI/model/system/solution"),
    (r"\bwe\s+(?:are\s+the\s+)?(?:user|deployer|operator)\b", "we (are the) user/deployer/operator"),
    (r"\bour\s+(?:use|deployment|operation)\s+of\b", "our use/deployment/operation of"),
    (r"\bexternal\s+(?:AI|model|algorithm|provider)\b", "external AI/model/algorithm/provider"),
]

_IMPORTER_PATTERNS: list[tuple[str, str]] = [
    (r"\bwe\s+import\b", "we import"),
    (r"\bimporter\b", "importer"),
    (r"\bEinführer\b", "Einführer"),  # DE: importer
    (r"\bimported\s+(?:AI|model|system|from)\b", "imported AI/model/system/from"),
    (r"\b(?:AI|model|system)\s+(?:developed|provided)\s+(?:outside|by\s+a\s+non[-\s]EU)\b",
     "AI developed outside/by non-EU"),
    (r"\bestablished\s+outside\s+the\s+(?:EU|Union|European\s+Union)\b",
     "established outside EU/Union"),
]

_DISTRIBUTOR_PATTERNS: list[tuple[str, str]] = [
    (r"\bwe\s+distribute\b", "we distribute"),
    (r"\bdistributor\b", "distributor"),
    (r"\bHändler\b", "Händler"),  # DE: distributor
    (r"\bwe\s+resell\b", "we resell"),
    (r"\bwe\s+make\s+(?:the\s+)?(?:AI|system|model)\s+available\b", "we make AI/system/model available"),
    (r"\bdistribution\s+(?:of\s+)?(?:AI|model|system)\b", "distribution of AI/model/system"),
]

_ALL_PATTERNS = {
    ActorType.PROVIDER: _PROVIDER_PATTERNS,
    ActorType.DEPLOYER: _DEPLOYER_PATTERNS,
    ActorType.IMPORTER: _IMPORTER_PATTERNS,
    ActorType.DISTRIBUTOR: _DISTRIBUTOR_PATTERNS,
}


# First-person ownership prefixes that make an AI_SYSTEM NER entity a provider signal.
# "our AI system" / "unser KI-System" means the author owns/built the system.
_OWNERSHIP_PREFIXES: tuple[str, ...] = ("our ", "unser ", "unsere ")


def classify_actor(full_text: str, chunks: list | None = None) -> ActorClassification:
    """Detect the EU AI Act Article 3 actor role from document text.

    Phase 3 ensemble: tries ML model first. If the model is trained and
    returns confidence ≥ _ML_CONFIDENCE_THRESHOLD, uses that result.
    Falls back to pattern matching otherwise.

    Args:
        full_text: Full raw text of the uploaded document.
        chunks: Optional enriched chunks — NER AI_SYSTEM entities whose text
                starts with a first-person ownership prefix ("our", "unser")
                are added as PROVIDER signals in the pattern fallback path.

    Returns:
        ActorClassification with detected actor type and matched signals.
    """
    # ── ML path (if model is trained) ────────────────────────────────────────
    ml = _ml_predict_actor(full_text)
    if ml is not None and ml.confidence >= _ML_CONFIDENCE_THRESHOLD:
        try:
            actor_type = ActorType(ml.label)
        except ValueError:
            actor_type = ActorType.UNKNOWN
        return ActorClassification(
            actor_type=actor_type,
            confidence=ml.confidence,
            matched_signals=[],
            reasoning=f"ML model (actor_classifier): {ml.label} with {ml.confidence:.0%} confidence.",
        )

    # ── Pattern fallback ──────────────────────────────────────────────────────
    text_lower = full_text.lower()

    hits: dict[ActorType, list[str]] = {actor: [] for actor in _ALL_PATTERNS}

    for actor_type, patterns in _ALL_PATTERNS.items():
        for pattern, label in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                hits[actor_type].append(label)

    # ── NER AI_SYSTEM ownership signals ──────────────────────────────────────
    # AI_SYSTEM entities with a first-person prefix ("our AI system",
    # "unser KI-System") unambiguously indicate the author owns the system —
    # a provider signal the regex may have missed in unusual phrasing.
    if chunks:
        for chunk in chunks:
            for ent_text in chunk.metadata.get("ner_entities", {}).get("AI_SYSTEM", []):
                if ent_text.lower().startswith(_OWNERSHIP_PREFIXES):
                    signal = f"NER AI_SYSTEM: {ent_text}"
                    if signal not in hits[ActorType.PROVIDER]:
                        hits[ActorType.PROVIDER].append(signal)

    scores = {actor: len(signals) for actor, signals in hits.items()}
    total_hits = sum(scores.values())

    if total_hits == 0:
        return ActorClassification(
            actor_type=ActorType.DEPLOYER,
            confidence=0.3,
            matched_signals=[],
            reasoning=(
                "No actor signals detected. Defaulting to DEPLOYER — the most common "
                "role for SMEs using third-party AI systems. Review manually."
            ),
        )

    winner = max(scores, key=lambda a: scores[a])
    winner_hits = scores[winner]

    # Confidence: winner share of total hits, penalised if runner-up is close
    runner_up_hits = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
    raw_confidence = winner_hits / total_hits
    gap_penalty = max(0.0, (winner_hits - runner_up_hits) / max(winner_hits, 1))
    confidence = round(min(1.0, raw_confidence * 0.7 + gap_penalty * 0.3), 2)

    reasoning_parts = [
        f"Detected {winner.value.upper()} role ({winner_hits} signal(s) matched)."
    ]
    for actor, signals in hits.items():
        if signals:
            reasoning_parts.append(f"{actor.value}: {', '.join(signals[:3])}")

    # Provider obligation tree is heavier — warn if close call
    if winner != ActorType.PROVIDER and scores[ActorType.PROVIDER] > 0:
        reasoning_parts.append(
            "Note: some provider signals present — verify whether your organisation "
            "also places this AI system on the market under its own name."
        )

    return ActorClassification(
        actor_type=winner,
        confidence=confidence,
        matched_signals=hits[winner],
        reasoning=" ".join(reasoning_parts),
    )
