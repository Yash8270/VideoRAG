"""
OpenAI Embeddings wrapper.
Returns a LangChain-compatible embeddings object used by both
the ingest pipeline and the RAG retrieval chain.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.

    The @lru_cache decorator ensures only one instance exists for the
    entire application lifetime, avoiding redundant network calls.

    Returns:
        LangChain HuggingFaceEmbeddings configured from settings.
    """
    settings = get_settings()
    logger.info("Loading HuggingFace embeddings model: %s", settings.EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
    )
