"""Text chunking service using LangChain RecursiveCharacterTextSplitter. (Phase 2)

Each chunk receives a UUID4 chunk_id and retains source_file metadata.
"""

import structlog
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
    raise NotImplementedError("chunker.chunk_text — implemented in Phase 2")
