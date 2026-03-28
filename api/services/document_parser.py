"""Document parsing service — PDF, DOCX, TXT, MD to raw text. (Phase 2)

Handles German characters (äöüß) correctly across all formats.
"""

import structlog

logger = structlog.get_logger()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


async def parse_document(file_path: str, filename: str) -> str:
    """Extract raw text from a document file.

    Supports .pdf (PyMuPDF), .docx (python-docx), .txt and .md (UTF-8).
    German characters (äöüß) are preserved in all formats.

    Args:
        file_path: Absolute path to the uploaded file on disk.
        filename: Original filename (used to detect extension).

    Returns:
        Extracted raw text as a single string.

    Raises:
        ValueError: If the file type is not supported.
        IOError: If the file cannot be read.
    """
    raise NotImplementedError("document_parser.parse_document — implemented in Phase 2")
