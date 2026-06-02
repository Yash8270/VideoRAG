"""
LangChain RAG chain — the conversational question-answering pipeline.

Flow:
  1. User question → semantic retrieval from ChromaDB (top-k chunks)
  2. Retrieved chunks + chat history → GPT-4o-mini → answer
  3. Answer + source documents returned to the API layer

Session memory is kept in-process (dict-based). For production, swap
_memories with a Redis-backed store.
"""

from __future__ import annotations

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.rag.embedder import get_embeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()

# ── Session memory store (in-process) ────────────────────────────────────────
_memories: dict[str, ConversationBufferWindowMemory] = {}

# ── System prompt injected into every retrieval call ─────────────────────────
_SYSTEM_PROMPT = """You are a knowledgeable AI assistant specialising in video content analysis.
You have access to transcripts from two videos:
  • A YouTube video
  • An Instagram Reel

Guidelines:
- Always cite which source (YouTube or Instagram) your answer draws from.
- If asked to compare the videos, highlight similarities and differences.
- If the answer is not in the provided context, say so honestly.
- Be concise, structured, and accurate.
"""


def _get_memory(session_id: str) -> ConversationBufferWindowMemory:
    """Return (or lazily create) a window-buffered memory for a session."""
    if session_id not in _memories:
        _memories[session_id] = ConversationBufferWindowMemory(
            k=6,                    # retain last 6 conversation turns
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
    return _memories[session_id]


def _build_retriever():
    """
    Build a LangChain Chroma retriever that reads from the persisted vector store.
    A new retriever is built per call to pick up fresh data after each ingest.
    """
    vector_store = Chroma(
        collection_name=_settings.CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=_settings.CHROMA_PERSIST_DIR,
    )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},     # return top-5 most relevant chunks
    )


async def run_rag_query(
    question: str,
    session_id: str = "default",
    history: list[dict] | None = None,
) -> dict:
    """
    Execute a RAG query against the stored video transcripts.

    Args:
        question:   The user's natural-language question.
        session_id: Conversation session identifier for memory isolation.
        history:    Optional list of prior messages (unused directly;
                    ConversationBufferWindowMemory manages history internally).

    Returns:
        dict with keys:
            'answer'           — LLM-generated response string
            'source_documents' — list of LangChain Document objects used
    """
    logger.info("RAG query  [session=%s]: %.80s", session_id, question)

    llm = ChatOpenAI(
        model=_settings.LLM_MODEL,
        temperature=0.3,            # low temperature → factual, deterministic answers
        openai_api_key=_settings.OPENAI_API_KEY,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_build_retriever(),
        memory=_get_memory(session_id),
        return_source_documents=True,
        output_key="answer",
        verbose=_settings.DEBUG,
    )

    result = await chain.ainvoke({"question": question})
    logger.info("RAG answer generated [session=%s]", session_id)
    return result
