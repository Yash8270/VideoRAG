"""
RAG Service — handles conversational QA over stored transcripts.
Rewritten manually to avoid LangChain complex Chain paradigms.
Compatible with Python 3.14.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.rag.embedder import get_embeddings
from app.utils.logger import get_logger
from app.vectorstore.client import get_chroma_client

logger = get_logger(__name__)
_settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Simple Dictionary Memory Store
# ─────────────────────────────────────────────────────────────────────────────

# Keys are session_id strings, values are lists of dicts: {"role": "user"|"assistant", "content": "..."}
_chat_history_store: Dict[str, List[Dict[str, str]]] = {}


def clear_session_history(session_id: str) -> None:
    """Clear the chat history for a specific session."""
    if session_id in _chat_history_store:
        del _chat_history_store[session_id]
        logger.info("Cleared chat history for session: %s", session_id)


# ─────────────────────────────────────────────────────────────────────────────
# Vectorstore & Retriever
# ─────────────────────────────────────────────────────────────────────────────


def get_vectorstore() -> Chroma:
    """Initialize LangChain Chroma wrapper using our persistent client."""
    return Chroma(
        client=get_chroma_client(),
        collection_name=_settings.CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public Async Interface
# ─────────────────────────────────────────────────────────────────────────────


async def ask_question(session_id: str, question: str, video_ids: list[str]) -> dict[str, Any]:
    """
    Main entry point for asking questions to the Conversational RAG system.
    Runs a manual sequence: Retrieve -> Build Context -> Call LLM -> Update Memory.
    """
    logger.info("RAG query received [session=%s]: %s (filtering to %s)", session_id, question, video_ids)

    # 1. Initialize or fetch conversation history
    if session_id not in _chat_history_store:
        _chat_history_store[session_id] = []
    
    history = _chat_history_store[session_id]

    # 2. Retrieve top 6 documents
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 6,
            "filter": {"video_id": {"$in": video_ids}}
        }
    )
    
    docs = await retriever.ainvoke(question)

    # 3. Build context string manually
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        vid = doc.metadata.get("video_id", "unknown")
        cid = doc.metadata.get("chunk_id", "unknown")
        text = doc.page_content
        
        context_parts.append(
            f"--- CHUNK START ---\n"
            f"Platform: {source}\n"
            f"Metadata: [Video ID: {vid}, Chunk: {cid}]\n"
            f"Content:\n{text}\n"
            f"--- CHUNK END ---"
        )
    
    context_str = "\n\n".join(context_parts)

    # 4. Construct prompt manually
    # We will pass the messages list directly to ChatOpenAI using primitive dicts.
    system_prompt_content = (
        "You are an expert video content strategist analyzing YouTube videos and Instagram Reels.\n"
        "Use the provided context chunks to answer the user's question, compare hooks, "
        "explain performance differences, or suggest improvements.\n\n"
        "CRITICAL INSTRUCTION: Every time you make a claim, provide information, or reference specific content "
        "from the context, you MUST explicitly cite the source using EXACTLY this format: [Video ID: <video_id>, Chunk: <chunk_id>].\n"
        "Example 1: 'The Instagram reel hook was much faster-paced [Video ID: abc123xyz, Chunk: 1].'\n\n"
        "If the answer cannot be determined from the provided context, clearly state that you do not know.\n\n"
        f"Context:\n{context_str}"
    )

    # Build the payload using primitive dictionaries
    messages = [
        {"role": "system", "content": system_prompt_content}
    ]
    
    # Add history
    messages.extend(history)
    
    # Add user question
    messages.append({"role": "user", "content": question})

    # 5. Call Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model=_settings.LLM_MODEL,  # Defaults to "gemini-2.5-flash"
        temperature=0.2,
        google_api_key=_settings.GOOGLE_API_KEY,
    )
    
    # ChatOpenAI ainvoke natively supports standard dict lists
    response_message = await llm.ainvoke(messages)
    answer_text = str(response_message.content)

    # Update memory dict
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer_text})

    logger.info(
        "RAG manual response generated for session %s (Retrieved %d chunks).",
        session_id,
        len(docs),
    )

    # 6. Return response
    return {
        "answer": answer_text,
        "context": docs
    }
