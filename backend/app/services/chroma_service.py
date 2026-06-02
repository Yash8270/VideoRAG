"""
ChromaDB service — high-level operations for storing and querying video chunks.
Handles persistence, metadata storage, and retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from app.vectorstore.client import get_collection
from app.services.chunking_service import to_chroma_payload
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.services.chunking_service import ChunkingResult

logger = get_logger(__name__)


def create_collection() -> None:
    """
    Initialize the ChromaDB collection.
    If the collection does not exist, it will be created.
    """
    logger.info("Ensuring ChromaDB collection exists...")
    col = get_collection()
    logger.info("ChromaDB collection is ready. Current document count: %d", col.count())


def insert_chunks(result: "ChunkingResult", embeddings: Optional[list[list[float]]] = None) -> None:
    """
    Insert transcript chunks into ChromaDB.
    
    Args:
        result: ChunkingResult containing the chunks and metadata.
        embeddings: Optional pre-computed embeddings for the chunks.
                    If None, Chroma will use its default embedding function, 
                    but passing pre-computed OpenAI embeddings is recommended.
    """
    if not result.chunks:
        logger.warning("No chunks to insert for video %s", result.video_id)
        return
        
    collection = get_collection()
    payload = to_chroma_payload(result)
    
    logger.info("Inserting %d chunks for video %s into ChromaDB...", len(payload.ids), result.video_id)
    
    collection.upsert(
        ids=payload.ids,
        documents=payload.documents,
        metadatas=payload.metadatas,
        embeddings=embeddings
    )
    
    logger.info("Successfully inserted chunks for video %s.", result.video_id)


def search_chunks(
    query_embedding: Optional[list[float]] = None, 
    query_text: Optional[str] = None, 
    top_k: int = 5, 
    video_id: Optional[str] = None
) -> dict[str, Any]:
    """
    Search ChromaDB for chunks matching the query.
    
    Args:
        query_embedding: The vector embedding of the search query.
        query_text: The plain text of the search query (used if embedding not provided).
        top_k: Number of results to return.
        video_id: Optional filter to restrict search to a specific video.
        
    Returns:
        ChromaDB query results dictionary.
    """
    collection = get_collection()
    logger.info("Searching ChromaDB for top %d matches...", top_k)
    
    # Build filter if video_id is provided
    where_filter = {"video_id": video_id} if video_id else None
    
    if query_embedding is not None:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )
    elif query_text is not None:
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter
        )
    else:
        raise ValueError("Either query_embedding or query_text must be provided.")
        
    return results
