"""
Video ingestion and comparison endpoints.

Routes:
  POST   /api/v1/video/ingest   — extract + embed YouTube & Instagram videos
  GET    /api/v1/video/compare  — return cached metadata for the last ingested pair
  DELETE /api/v1/video/reset    — wipe vector store and in-memory cache
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import (
    VideoCompareResponse,
    VideoIngestRequest,
    VideoIngestResponse,
    VideoMetadata,
)
from app.rag.embedder import get_embeddings
from app.services.chunker import chunk_transcript
from app.services.instagram_service import extract_instagram
from app.services.youtube_service import extract_youtube
from app.utils.logger import get_logger
from app.vectorstore.client import get_collection, reset_collection

router = APIRouter(prefix="/video", tags=["Video"])
logger = get_logger(__name__)

# ── In-memory cache for the compare endpoint ──────────────────────────────────
# In production, persist this in Redis or a lightweight DB.
_last_youtube: VideoMetadata | None = None
_last_instagram: VideoMetadata | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────


def _embed_and_store(video: VideoMetadata) -> int:
    """
    Chunk a video transcript, embed the chunks, and upsert into ChromaDB.

    Args:
        video: Normalised VideoMetadata object.

    Returns:
        Number of chunks stored.
    """
    chunk_metadata = {
        "source": video.source.value,
        "video_id": video.video_id,
        "title": video.title,
        "url": video.url,
    }
    docs = chunk_transcript(video.transcript, chunk_metadata)

    if not docs:
        logger.warning("No chunks produced for [%s] %s", video.source.value, video.video_id)
        return 0

    embeddings = get_embeddings()
    collection = get_collection()

    texts = [doc.page_content for doc in docs]
    metas = [doc.metadata for doc in docs]
    ids = [f"{video.source.value}_{video.video_id}_{i}" for i in range(len(docs))]

    vectors = embeddings.embed_documents(texts)

    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metas,
    )
    logger.info(
        "Stored %d chunks → ChromaDB  [%s | %s]",
        len(docs),
        video.source.value,
        video.video_id,
    )
    return len(docs)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=VideoIngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a YouTube video and an Instagram Reel",
    description=(
        "Extracts metadata and transcripts from both URLs, chunks the transcripts, "
        "generates OpenAI embeddings, and stores them in ChromaDB. "
        "Partial success is returned if one source fails."
    ),
)
async def ingest_videos(request: VideoIngestRequest) -> VideoIngestResponse:
    global _last_youtube, _last_instagram

    errors: list[str] = []
    youtube_meta: VideoMetadata | None = None
    instagram_meta: VideoMetadata | None = None
    total_chunks = 0

    # ── 1. YouTube ────────────────────────────────────────────────────────────
    try:
        youtube_meta = await extract_youtube(request.youtube_url)
        total_chunks += _embed_and_store(youtube_meta)
        _last_youtube = youtube_meta
    except Exception as exc:
        logger.error("YouTube extraction error: %s", exc)
        errors.append(f"YouTube — {exc}")

    # ── 2. Instagram ──────────────────────────────────────────────────────────
    try:
        instagram_meta = await extract_instagram(request.instagram_url)
        total_chunks += _embed_and_store(instagram_meta)
        _last_instagram = instagram_meta
    except Exception as exc:
        logger.error("Instagram extraction error: %s", exc)
        errors.append(f"Instagram — {exc}")

    # ── 3. If both fail, raise HTTP error ────────────────────────────────────
    if errors and not youtube_meta and not instagram_meta:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="; ".join(errors),
        )

    return VideoIngestResponse(
        success=len(errors) == 0,
        message=(
            "Both videos ingested successfully."
            if not errors
            else f"Partial success. Errors: {'; '.join(errors)}"
        ),
        youtube=youtube_meta,
        instagram=instagram_meta,
        chunks_stored=total_chunks,
    )


@router.get(
    "/compare",
    response_model=VideoCompareResponse,
    summary="Get metadata for the last ingested video pair",
    description="Returns cached metadata for side-by-side comparison in the UI.",
)
async def compare_videos() -> VideoCompareResponse:
    if not _last_youtube and not _last_instagram:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No videos ingested yet. Call POST /api/v1/video/ingest first.",
        )
    return VideoCompareResponse(youtube=_last_youtube, instagram=_last_instagram)


@router.delete(
    "/reset",
    status_code=status.HTTP_200_OK,
    summary="Reset the vector store and clear cached videos",
)
async def reset_videos() -> dict:
    """Drop all stored vectors and clear the in-memory video cache."""
    global _last_youtube, _last_instagram
    reset_collection()
    _last_youtube = None
    _last_instagram = None
    return {"message": "Vector store and video cache cleared successfully."}
