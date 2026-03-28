"""Gap analysis service — per-article LLM structured analysis. (Phase 2)

Concatenates user document chunks with retrieved regulatory passages,
sends to Ollama with the gap_analysis.txt prompt, parses the JSON response
into an ArticleScore with GapItems and recommendations.
"""

import structlog
from models.schemas import ArticleDomain, ArticleScore, DocumentChunk
from services.ollama_client import OllamaClient

logger = structlog.get_logger()


async def analyse_article(
    article_num: int,
    domain: ArticleDomain,
    user_chunks: list[DocumentChunk],
    regulatory_passages: list[dict],
    ollama: OllamaClient,
) -> ArticleScore:
    """Perform gap analysis for a single EU AI Act article.

    Builds a prompt from user documentation and retrieved regulatory text,
    calls Ollama with JSON output mode, and parses the result into ArticleScore.

    Args:
        article_num: EU AI Act article number (9–15).
        domain: ArticleDomain enum value for this article.
        user_chunks: Document chunks classified to this domain.
        regulatory_passages: Top-k passages from ChromaDB for this domain.
        ollama: OllamaClient instance.

    Returns:
        ArticleScore with score, gaps, and recommendations populated.
    """
    raise NotImplementedError("gap_analyser.analyse_article — implemented in Phase 2")
