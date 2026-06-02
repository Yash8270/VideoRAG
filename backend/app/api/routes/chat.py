"""
Chat / RAG query endpoint.
"""

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import ChatRequest
from app.services.rag_service import ask_question
from app.utils.logger import get_logger

router = APIRouter(tags=["Chat"])
logger = get_logger(__name__)


@router.post(
    "/chat",
    status_code=status.HTTP_200_OK,
    summary="Ask questions using the Conversational RAG pipeline",
)
async def chat_query(request: ChatRequest) -> dict[str, Any]:
    """
    RAG Endpoint: Retrieves the most relevant transcript chunks from ChromaDB,
    then generates a GPT-4o-mini answer with strict metadata citations.
    """
    # Auto-generate a session ID if the client does not supply one
    session_id = request.session_id or str(uuid.uuid4())
    logger.info("Processing chat query for session %s: %s", session_id, request.message)

    try:
        # Run Conversational RAG
        result = await ask_question(
            session_id=session_id,
            question=request.message,
            video_ids=request.video_ids
        )

        # Map LangChain Document format to our custom source response format
        sources = []
        for doc in result.get("context", []):
            sources.append({
                "video_id": doc.metadata.get("video_id"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "source": doc.metadata.get("source"),
                "text": doc.page_content[:400]  # truncate to keep response payload compact
            })

        return {
            "answer": result.get("answer", ""),
            "sources": sources,
            "session_id": session_id
        }

    except Exception as exc:
        logger.exception("Chat RAG failed [session=%s]:", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat pipeline failed: {exc}",
        )
