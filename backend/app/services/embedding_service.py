"""
Embedding service — generates vector embeddings for text chunks.
Uses OpenAI text-embedding-3-small via LangChain wrapper.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.rag.embedder import get_embeddings
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.services.chunking_service import ChunkingResult

logger = get_logger(__name__)


async def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of strings using the configured OpenAI model.

    Args:
        texts: A list of string texts to embed.

    Returns:
        A list of embedding vectors (list of floats).
    """
    if not texts:
        return []
    
    logger.info("Generating embeddings for %d chunks...", len(texts))
    embedder = get_embeddings()
    
    # LangChain's aembed_documents automatically handles batching and rate limiting
    embeddings = await embedder.aembed_documents(texts)
    
    logger.info("Successfully generated %d embeddings.", len(embeddings))
    return embeddings


async def embed_chunking_result(result: "ChunkingResult") -> list[list[float]]:
    """
    Convenience function to generate embeddings directly from a ChunkingResult.

    Args:
        result: ChunkingResult returned by chunking_service.

    Returns:
        A list of embedding vectors corresponding to the chunks.
    """
    texts = [chunk.text for chunk in result.chunks]
    return await generate_embeddings(texts)
