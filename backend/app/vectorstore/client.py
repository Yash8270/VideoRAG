"""
ChromaDB client — singleton wrapper around the on-disk persistent vector store.

Usage:
    from app.vectorstore.client import get_collection, is_healthy
"""

from __future__ import annotations

from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()

# ── Module-level singletons ───────────────────────────────────────────────────
_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Lazily create and return the global Chroma persistent client."""
    global _client
    if _client is None:
        logger.info(
            "Initialising ChromaDB persistent client at '%s'",
            _settings.CHROMA_PERSIST_DIR,
        )
        _client = chromadb.PersistentClient(
            path=_settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection() -> chromadb.Collection:
    """Lazily create and return the active Chroma collection."""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=_settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for embeddings
        )
        logger.info(
            "Using ChromaDB collection '%s' — %d documents indexed",
            _settings.CHROMA_COLLECTION_NAME,
            _collection.count(),
        )
    return _collection


def reset_collection() -> None:
    """
    Delete and recreate the collection — drops ALL stored vectors.
    Use only when the user explicitly requests a full reset.
    """
    global _collection
    client = get_chroma_client()
    client.delete_collection(_settings.CHROMA_COLLECTION_NAME)
    _collection = None
    logger.warning(
        "ChromaDB collection '%s' has been deleted and reset.",
        _settings.CHROMA_COLLECTION_NAME,
    )


def is_healthy() -> bool:
    """Return True if the ChromaDB client responds to a heartbeat ping."""
    try:
        get_chroma_client().heartbeat()
        return True
    except Exception:
        return False
