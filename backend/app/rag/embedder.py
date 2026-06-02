"""
OpenAI Embeddings wrapper.
Returns a LangChain-compatible embeddings object used by both
the ingest pipeline and the RAG retrieval chain.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def get_embeddings() -> FastEmbedEmbeddings | GoogleGenerativeAIEmbeddings:
    """
    Return a cached FastEmbedEmbeddings instance.

    The @lru_cache decorator ensures only one instance exists for the
    entire application lifetime, avoiding redundant network calls.

    Returns:
        LangChain FastEmbedEmbeddings configured from settings.
    """
    settings = get_settings()
    model = settings.EMBEDDING_MODEL

    if model.startswith("models/"):
        logger.info("Loading Google Gemini embeddings model: %s", model)
        return GoogleGenerativeAIEmbeddings(
            model=model, 
            google_api_key=settings.GOOGLE_API_KEY
        )
    else:
        logger.info("Loading FastEmbed ONNX embeddings model: %s", model)
        return FastEmbedEmbeddings(
            model_name=model,
        )
