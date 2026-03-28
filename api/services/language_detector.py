"""Language detection service. (Phase 2)

Uses the langdetect library on the first 500 characters of text.
Returns ISO-639-1 codes 'de' or 'en', defaulting to 'en' on failure.
"""

import structlog

logger = structlog.get_logger()


async def detect_language(text: str) -> str:
    """Detect the primary language of the given text.

    Samples the first 500 characters for speed. Defaults to 'en' on any error
    (e.g. too-short text, ambiguous result).

    Args:
        text: Input text to analyse.

    Returns:
        ISO-639-1 language code: 'de' for German, 'en' for all others.
    """
    raise NotImplementedError("language_detector.detect_language — implemented in Phase 2")
