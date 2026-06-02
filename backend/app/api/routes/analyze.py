"""
Analyze endpoint - full pipeline for extraction, analytics, embedding, and storage.
"""

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import AnalyzeRequest
from app.services.youtube_service import extract_youtube
from app.services.instagram_service import extract_instagram
from app.services.transcription_service import transcribe, segments_to_json
from app.services.analytics_service import (
    from_youtube,
    from_instagram,
    compare,
    ComparisonSummary,
)
from app.services.chunking_service import chunk_from_youtube, chunk_from_instagram
from app.services.embedding_service import embed_chunking_result
from app.services.chroma_service import insert_chunks
from app.utils.logger import get_logger

router = APIRouter(tags=["Analyze"])
logger = get_logger(__name__)


@router.post(
    "/analyze",
    response_model=ComparisonSummary,
    status_code=status.HTTP_200_OK,
    summary="End-to-end ingestion and analysis pipeline",
)
async def analyze_videos(request: AnalyzeRequest) -> ComparisonSummary:
    """
    1. Extracts metadata + transcript from YouTube
    2. Extracts metadata + audio from Instagram -> transcribes audio
    3. Calculates engagement metrics
    4. Chunks transcripts
    5. Generates OpenAI embeddings
    6. Stores in ChromaDB
    """
    logger.info("Starting /analyze pipeline for %s and %s", request.youtube_url, request.instagram_url)
    try:
        # 1. YouTube Extraction
        logger.info("Extracting YouTube data...")
        yt_data = await extract_youtube(request.youtube_url)

        # 2. Instagram Extraction & Transcription
        logger.info("Extracting Instagram data...")
        ig_data = await extract_instagram(request.instagram_url)

        if ig_data.audio_path:
            logger.info("Transcribing Instagram audio via faster-whisper...")
            try:
                segments = await transcribe(ig_data.audio_path)
                ig_data.transcript = segments_to_json(segments)
            except Exception as exc:
                logger.warning("Failed to transcribe Instagram audio (might be silent/corrupted): %s", exc)
                ig_data.transcript = None

        # 3. Analytics & Engagement Calculation
        logger.info("Computing engagement analytics...")
        yt_summary = from_youtube(yt_data)
        ig_summary = from_instagram(ig_data)
        comparison = compare(yt_summary, ig_summary)

        # 4. Transcript Chunking
        logger.info("Chunking transcripts...")
        yt_chunks = chunk_from_youtube(yt_data)
        ig_chunks = chunk_from_instagram(ig_data)

        # 5 & 6. Embed and Store - YouTube
        if yt_chunks.chunks:
            logger.info(
                "YouTube Ingestion Summary | Transcript length: %d chars | Chunk count: %d", 
                len(yt_data.transcript) if yt_data.transcript else 0, 
                yt_chunks.total_chunks
            )
            yt_embeddings = await embed_chunking_result(yt_chunks)
            logger.info("YouTube Ingestion Summary | Embedding count: %d", len(yt_embeddings))
            insert_chunks(yt_chunks, embeddings=yt_embeddings)
            logger.info("YouTube Ingestion Summary | Vector insertion count: %d", len(yt_embeddings))
        else:
            logger.warning("No YouTube chunks to embed/store.")

        # 5 & 6. Embed and Store - Instagram
        if ig_chunks.chunks:
            transcript_len = len(ig_data.transcript) if isinstance(ig_data.transcript, str) else len(str(ig_data.transcript))
            logger.info(
                "Instagram Ingestion Summary | Transcript length: %d chars | Chunk count: %d", 
                transcript_len, 
                ig_chunks.total_chunks
            )
            ig_embeddings = await embed_chunking_result(ig_chunks)
            logger.info("Instagram Ingestion Summary | Embedding count: %d", len(ig_embeddings))
            insert_chunks(ig_chunks, embeddings=ig_embeddings)
            logger.info("Instagram Ingestion Summary | Vector insertion count: %d", len(ig_embeddings))
        else:
            logger.warning("No Instagram chunks to embed/store.")

        logger.info("Pipeline completed successfully!")
        return comparison

    except Exception as exc:
        logger.exception("Pipeline failed:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline failed: {exc}",
        )
