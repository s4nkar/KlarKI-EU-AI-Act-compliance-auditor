"""Language detection service.

Uses the langdetect library on the first 500 characters of text.
Returns ISO-639-1 codes 'de' or 'en', defaulting to 'en' on failure.
"""

import structlog

logger = structlog.get_logger()


async def detect_language(text: str) -> str:
    """Detect the primary language of the given text.

    Samples the first 500 characters for speed. Defaults to 'en' on any
    error (e.g. too-short text, ambiguous result).

    Args:
        text: Input text to analyse.

    Returns:
        ISO-639-1 language code: 'de' for German, 'en' for all others.
    """
    try:
        from langdetect import detect

        sample = text[:500].strip()
        if len(sample) < 20:
            return "en"

        lang = detect(sample)
        result = "de" if lang == "de" else "en"
        logger.debug("language_detected", lang=lang, result=result)
        return result

    except Exception as exc:
        logger.warning("language_detect_failed", error=str(exc))
        return "en"
