"""Document parsing service — PDF, DOCX, TXT, MD to raw text.

Handles German characters (äöüß) correctly across all formats.
All functions are async; blocking I/O runs in asyncio.to_thread.
"""

import asyncio
from pathlib import Path

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
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    logger.info("document_parse_start", filename=filename, ext=ext)

    if ext == ".pdf":
        text = await asyncio.to_thread(_parse_pdf, file_path)
    elif ext == ".docx":
        text = await asyncio.to_thread(_parse_docx, file_path)
    else:  # .txt or .md
        text = await asyncio.to_thread(_parse_text, file_path)

    logger.info("document_parse_done", filename=filename, chars=len(text))
    return text


def _parse_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF (fitz).

    Preserves paragraph structure; joins pages with double newlines.
    German ligatures and special characters handled by fitz natively.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(file_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(pages)


def _parse_docx(file_path: str) -> str:
    """Extract text from DOCX using python-docx.

    Iterates all paragraphs; preserves paragraph breaks.
    Tables are extracted row by row with tab separation.
    """
    from docx import Document

    doc = Document(file_path)
    parts = []

    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)

    # Also extract table content
    for table in doc.tables:
        for row in table.rows:
            row_text = "\t".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                parts.append(row_text)

    return "\n\n".join(parts)


def _parse_text(file_path: str) -> str:
    """Read plain text or Markdown file as UTF-8."""
    with open(file_path, encoding="utf-8", errors="replace") as f:
        return f.read()
