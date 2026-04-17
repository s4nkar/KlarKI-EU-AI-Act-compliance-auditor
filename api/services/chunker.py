"""Text chunking service.

Two strategies are provided:

chunk_text()        — original 512-char RecursiveCharacterTextSplitter (used by
                      tests and any caller that needs deterministic behaviour).

legal_chunk_text()  — [Phase 3] heading-aware splitter for user-uploaded policy
                      documents. Detects section headings, keeps paragraphs
                      together within a section, and stores section_heading /
                      section_index in chunk metadata. Falls back to paragraph
                      splitting when no headings are found.

Use legal_chunk_text() in the audit pipeline. chunk_text() is kept for
backwards-compatibility with tests.
"""

import re
import uuid

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.schemas import DocumentChunk

logger = structlog.get_logger()

# ── Heading detection patterns (ordered: most specific first) ─────────────────
# Each pattern: (regex, group-index for the heading text)
_HEADING_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE), 2),             # ## Heading
    (re.compile(r"^(\d+\.\d+\.?\s+[A-Z].{3,80})$", re.MULTILINE), 1), # 1.1 Sub-section
    (re.compile(r"^(\d+\.\s+[A-Z].{3,80})$", re.MULTILINE), 1),       # 1. Section
    (re.compile(r"^(Article\s+\d+[^\n]{0,60})$", re.MULTILINE), 1),   # Article 9 —
    (re.compile(r"^(Artikel\s+\d+[^\n]{0,60})$", re.MULTILINE), 1),   # Artikel 9 — (DE)
    (re.compile(r"^([A-Z][A-Z\s\-]{4,50}):?\s*$", re.MULTILINE), 1), # RISK MANAGEMENT
]

# Max characters per legal chunk before sub-splitting
_MAX_LEGAL_CHUNK = 800
# Min characters — chunks smaller than this are merged with the next
_MIN_LEGAL_CHUNK = 80


def _detect_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (heading, body) pairs using heading pattern detection.

    Returns a list of (heading, body) tuples. If no headings are found,
    returns a single ("", full_text) tuple for paragraph-level fallback.
    """
    # Find all heading positions using the first matching pattern family
    all_matches: list[tuple[int, int, str]] = []  # (start, end, heading_text)

    for pattern, group in _HEADING_PATTERNS:
        for m in pattern.finditer(text):
            heading_text = m.group(group).strip()
            all_matches.append((m.start(), m.end(), heading_text))

    if not all_matches:
        return [("", text)]

    # Sort by position and deduplicate overlapping matches
    all_matches.sort(key=lambda x: x[0])
    deduped: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, heading in all_matches:
        if start >= last_end:
            deduped.append((start, end, heading))
            last_end = end

    sections: list[tuple[str, str]] = []
    for i, (start, end, heading) in enumerate(deduped):
        next_start = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        body = text[end:next_start].strip()
        sections.append((heading, body))

    # Capture any text before the first heading as an unlabelled preamble
    first_heading_start = deduped[0][0]
    preamble = text[:first_heading_start].strip()
    if preamble:
        sections.insert(0, ("", preamble))

    return sections


def _split_long_body(body: str, max_size: int, overlap: int) -> list[str]:
    """Sub-split a section body that exceeds max_size using paragraph boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return [t for t in splitter.split_text(body) if t.strip()]


async def legal_chunk_text(
    raw_text: str,
    source_file: str,
    max_chunk_size: int = _MAX_LEGAL_CHUNK,
    overlap: int = 80,
) -> list[DocumentChunk]:
    """[Phase 3] Heading-aware chunker for user-uploaded compliance documents.

    Strategy:
      1. Detect section headings → split into (heading, body) sections.
      2. For each section, if body ≤ max_chunk_size → one chunk.
         If body > max_chunk_size → sub-split by paragraph with overlap.
      3. Merge sub-chunks smaller than _MIN_LEGAL_CHUNK into the preceding chunk.
      4. Store section_heading, section_index in chunk.metadata.

    Falls back to paragraph splitting when no headings are found.

    Args:
        raw_text: Full raw text from the document parser.
        source_file: Original filename.
        max_chunk_size: Maximum characters per output chunk.
        overlap: Character overlap when sub-splitting long sections.

    Returns:
        List of DocumentChunk objects with heading metadata populated.
    """
    if not raw_text.strip():
        logger.warning("legal_chunker_empty_text", source_file=source_file)
        return []

    sections = _detect_sections(raw_text)

    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for section_idx, (heading, body) in enumerate(sections):
        if not body.strip():
            continue

        if len(body) <= max_chunk_size:
            sub_texts = [body]
        else:
            sub_texts = _split_long_body(body, max_chunk_size, overlap)

        for sub_text in sub_texts:
            if not sub_text.strip():
                continue

            # Merge tiny trailing fragments into the previous chunk
            if len(sub_text) < _MIN_LEGAL_CHUNK and chunks:
                prev = chunks[-1]
                merged_text = prev.text + " " + sub_text
                chunks[-1] = DocumentChunk(
                    chunk_id=prev.chunk_id,
                    text=merged_text,
                    source_file=prev.source_file,
                    chunk_index=prev.chunk_index,
                    language=prev.language,
                    domain=prev.domain,
                    metadata=prev.metadata,
                )
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=sub_text,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    metadata={
                        "section_heading": heading,
                        "section_index": section_idx,
                    },
                )
            )
            chunk_index += 1

    logger.info(
        "legal_chunker_done",
        source_file=source_file,
        sections=len(sections),
        chunks=len(chunks),
        headings_detected=sum(1 for h, _ in sections if h),
    )
    return chunks


async def chunk_text(
    raw_text: str,
    source_file: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """Split raw text into overlapping chunks (original strategy).

    Uses LangChain RecursiveCharacterTextSplitter with separators:
    ['\\n\\n', '\\n', '. ', ' ', ''].

    Kept for backward-compatibility with tests. Use legal_chunk_text() in
    the audit pipeline.

    Args:
        raw_text: Full document text to split.
        source_file: Original filename stored in each chunk's metadata.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between adjacent chunks.

    Returns:
        List of DocumentChunk objects with chunk_id, text, source_file, chunk_index.
    """
    if not raw_text.strip():
        logger.warning("chunker_empty_text", source_file=source_file)
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    texts = splitter.split_text(raw_text)

    chunks = [
        DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            text=text,
            source_file=source_file,
            chunk_index=i,
        )
        for i, text in enumerate(texts)
        if text.strip()
    ]

    logger.info("chunker_done", source_file=source_file, chunks=len(chunks))
    return chunks
