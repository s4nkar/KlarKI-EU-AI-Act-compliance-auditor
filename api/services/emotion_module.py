"""Emotion recognition detection module — Article 5 prohibition check.

Scans document chunks for emotion recognition keywords combined with
workplace/education/commercial context to determine Art. 5 prohibition.
"""

import structlog
from models.schemas import DocumentChunk, EmotionFlag

logger = structlog.get_logger()

EMOTION_KEYWORDS = [
    "emotion recognition", "facial emotion", "sentiment analysis",
    "affect detection", "mood detection", "emotional state",
    "gefühlserkennung", "emotionserkennung", "stimmungsanalyse",
]
WORKPLACE_KEYWORDS = ["workplace", "employee", "arbeitsplatz", "mitarbeiter", "arbeitnehmer"]
EDUCATION_KEYWORDS = ["school", "student", "education", "classroom", "schule", "bildung", "lernende"]
COMMERCIAL_KEYWORDS = ["customer", "consumer", "retail", "marketing", "kunde", "verbraucher"]


async def check_emotion_recognition(chunks: list[DocumentChunk]) -> EmotionFlag:
    """Scan document chunks for Art. 5 emotion recognition prohibition triggers.

    Keyword scan logic:
    - EMOTION + WORKPLACE/EDUCATION → is_prohibited = True (Art. 5(1)(f))
    - EMOTION + COMMERCIAL           → detected = True, is_prohibited = False (high-risk)
    - EMOTION only                   → detected = True, context unknown

    Args:
        chunks: All document chunks from the uploaded documents.

    Returns:
        EmotionFlag with detected, is_prohibited, context, and explanation.
    """
    full_text = " ".join(c.text.lower() for c in chunks)

    emotion_match = next((kw for kw in EMOTION_KEYWORDS if kw in full_text), None)
    if not emotion_match:
        logger.debug("emotion_check_clean")
        return EmotionFlag(detected=False)

    is_workplace  = any(kw in full_text for kw in WORKPLACE_KEYWORDS)
    is_education  = any(kw in full_text for kw in EDUCATION_KEYWORDS)
    is_commercial = any(kw in full_text for kw in COMMERCIAL_KEYWORDS)

    if is_workplace or is_education:
        context = "workplace" if is_workplace else "education"
        logger.warning("emotion_prohibited_context", context=context, keyword=emotion_match)
        return EmotionFlag(
            detected=True,
            context=context,
            is_prohibited=True,
            explanation=(
                f"Emotion recognition in a {context} context is prohibited under "
                "Article 5(1)(f) of the EU AI Act. This system may not be legally deployed."
            ),
        )

    if is_commercial:
        logger.info("emotion_high_risk_context", keyword=emotion_match)
        return EmotionFlag(
            detected=True,
            context="commercial",
            is_prohibited=False,
            explanation=(
                "Emotion recognition in a commercial context is high-risk but not prohibited. "
                "Ensure full compliance with Article 13 (transparency) and Article 9 (risk management)."
            ),
        )

    logger.info("emotion_detected_unknown_context", keyword=emotion_match)
    return EmotionFlag(
        detected=True,
        context="unknown",
        is_prohibited=False,
        explanation=(
            f"Emotion-related capability detected ('{emotion_match}'). "
            "Context is unclear — manually verify compliance with Article 5(1)(f)."
        ),
    )
