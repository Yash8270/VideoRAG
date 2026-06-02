"""
Pydantic v2 schemas — all API request / response bodies are defined here.
Keeps the data contract in a single, discoverable location.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class VideoSource(str, Enum):
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"


# ─────────────────────────────────────────────────────────────────────────────
# Video — Ingest
# ─────────────────────────────────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    """Payload sent by the frontend to kick off extraction and analysis."""

    youtube_url: str = Field(..., description="Public YouTube video URL")
    instagram_url: str = Field(..., description="Public Instagram Reel URL")


class VideoMetadata(BaseModel):
    """
    Normalised metadata returned for any ingested video, regardless of source.
    Stored in-memory as the 'compare' cache and returned by the ingest endpoint.
    """

    source: VideoSource
    video_id: str = Field(..., description="Platform-specific unique identifier")
    url: str
    title: str
    description: Optional[str] = None
    transcript: str = Field(..., description="Cleaned transcript / caption text")
    duration: Optional[float] = None       # seconds
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    channel_name: Optional[str] = None
    thumbnail_url: Optional[str] = None
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Platform-specific extra fields (tags, upload date, etc.)",
    )


class VideoIngestResponse(BaseModel):
    success: bool
    message: str
    youtube: Optional[VideoMetadata] = None
    instagram: Optional[VideoMetadata] = None
    chunks_stored: int = 0


class VideoCompareResponse(BaseModel):
    """Side-by-side metadata for the last ingested video pair."""

    youtube: Optional[VideoMetadata] = None
    instagram: Optional[VideoMetadata] = None


# ─────────────────────────────────────────────────────────────────────────────
# Chat / RAG
# ─────────────────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    """Message sent by the user, optionally with conversation history."""

    message: str = Field(..., min_length=1, description="User's message/question about the videos")
    session_id: Optional[str] = Field(
        None, description="Conversation session ID — auto-generated if omitted"
    )
    video_ids: list[str] = Field(..., description="List of video IDs to restrict the RAG search to")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous turns for multi-turn conversation support",
    )


class SourceChunk(BaseModel):
    """A single retrieved document chunk surfaced as a citation."""

    source: VideoSource
    video_id: str
    chunk_text: str
    relevance_score: Optional[float] = None


class ChatResponse(BaseModel):
    """LLM answer plus the source chunks used to generate it."""

    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    session_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    version: str
    chroma_ready: bool
