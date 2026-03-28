"""Text chunking service using LangChain RecursiveCharacterTextSplitter.

Each chunk receives a UUID4 chunk_id and retains source_file metadata.
"""

import uuid

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.schemas import DocumentChunk

logger = structlog.get_logger()


async def chunk_text(
    raw_text: str,
    source_file: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """Split raw text into overlapping chunks suitable for embedding.

    Uses LangChain RecursiveCharacterTextSplitter with separators:
    ['\\n\\n', '\\n', '. ', ' ', ''].

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
