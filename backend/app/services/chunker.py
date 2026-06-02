"""
Text chunking service.

Splits a full transcript into overlapping chunks using LangChain's
RecursiveCharacterTextSplitter, producing LangChain Document objects
ready for embedding and storage in ChromaDB.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()


def chunk_transcript(
    transcript: str,
    metadata: dict,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split a cleaned transcript into overlapping LangChain Documents.

    Args:
        transcript:    Full cleaned transcript string.
        metadata:      Dict attached to every produced chunk
                       (must include at least 'source' and 'video_id').
        chunk_size:    Optional override for settings.CHUNK_SIZE.
        chunk_overlap: Optional override for settings.CHUNK_OVERLAP.

    Returns:
        List of LangChain Document objects, each carrying
        page_content (the chunk text) and metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or _settings.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or _settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    docs = splitter.create_documents(
        texts=[transcript],
        metadatas=[metadata],
    )

    logger.info(
        "Transcript chunked  [source=%s | video_id=%s] → %d chunks",
        metadata.get("source", "?"),
        metadata.get("video_id", "?"),
        len(docs),
    )
    return docs
