"""
Production-ready YouTube extraction service.

Full pipeline:
  1. Validate URL → extract 11-character video ID
  2. Fetch rich metadata via yt-dlp  (title, creator, views, likes,
     comments, upload date, duration, thumbnail, tags, categories)
  3. Fetch transcript — 3-level fallback chain:
       a. Manual captions       (youtube-transcript-api 1.x, English preferred)
       b. Auto-generated captions  (any language, translated if needed)
       c. faster-whisper          (download audio → local transcription)
  4. Clean transcript text
  5. Return structured YouTubeVideoData (Pydantic model)

All blocking I/O runs inside asyncio.to_thread() so the async event loop
is never blocked.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import json
import urllib.parse
import urllib.request
from enum import Enum
from typing import Any, Optional

import yt_dlp
from faster_whisper import WhisperModel
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)
from pydantic import BaseModel, Field

from app.utils.logger import get_logger
from app.utils.text_cleaner import clean_transcript

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Pydantic model
# ─────────────────────────────────────────────────────────────────────────────


class TranscriptSource(str, Enum):
    """Describes how the transcript text was obtained."""
    MANUAL         = "manual_captions"      # human-created captions
    AUTO_GENERATED = "auto_generated"       # YouTube auto-captions
    WHISPER        = "whisper_fallback"     # faster-whisper local transcription
    UNAVAILABLE    = "unavailable"          # no transcript could be obtained


class YouTubeVideoData(BaseModel):
    """
    Fully structured, validated representation of an extracted YouTube video.
    This is the single return type of extract_youtube().
    """

    # ── Identifiers ──────────────────────────────────────────────────────────
    video_id:           str             = Field(..., description="11-character YouTube video ID")
    url:                str             = Field(..., description="Canonical watch URL")

    # ── Core metadata ────────────────────────────────────────────────────────
    title:              str             = Field(..., description="Video title")
    creator:            str             = Field(..., description="Channel / uploader name")
    description:        Optional[str]   = Field(None, description="Full video description")
    upload_date:        Optional[str]   = Field(None, description="ISO 8601 date — YYYY-MM-DD")
    duration_seconds:   Optional[int]   = Field(None, description="Duration in seconds")
    duration_formatted: Optional[str]   = Field(None, description="Human-readable HH:MM:SS")
    thumbnail_url:      Optional[str]   = Field(None, description="Highest-resolution thumbnail URL")
    language:           Optional[str]   = Field(None, description="Primary language code (e.g. 'en')")

    # ── Engagement ───────────────────────────────────────────────────────────
    views:              Optional[int]   = Field(None, description="View count")
    likes:              Optional[int]   = Field(None, description="Like count")
    comments:           Optional[int]   = Field(None, description="Comment count")

    # ── Taxonomy ─────────────────────────────────────────────────────────────
    tags:               list[str]       = Field(default_factory=list)
    categories:         list[str]       = Field(default_factory=list)

    # ── Transcript ───────────────────────────────────────────────────────────
    transcript:         str             = Field(..., description="Cleaned transcript text")
    transcript_source:  TranscriptSource= Field(..., description="How the transcript was obtained")

    # ── Raw payload ──────────────────────────────────────────────────────────
    raw_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Selected raw fields from yt-dlp for downstream use",
    )


# ─────────────────────────────────────────────────────────────────────────────
# yt-dlp configuration
# ─────────────────────────────────────────────────────────────────────────────

_YDL_META_OPTS: dict[str, Any] = {
    "quiet":         True,
    "no_warnings":   True,
    "skip_download": True,   # metadata only — never download the video
    "extract_flat":  False,
}

_YDL_AUDIO_OPTS_TEMPLATE: dict[str, Any] = {
    "quiet":       True,
    "no_warnings": True,
    # Download best audio stream — no video
    "format":      "bestaudio/best",
    # Write the file to a caller-specified path; %(ext)s will be filled in
    # 'outtmpl' is set dynamically per request
}


# ─────────────────────────────────────────────────────────────────────────────
# Whisper model — module-level lazy singleton
# ─────────────────────────────────────────────────────────────────────────────

# Model size trade-offs:
#   "tiny"   →  fastest,  ~1 GB VRAM,  lower accuracy
#   "base"   →  fast,     ~1 GB VRAM,  good for clear speech
#   "small"  →  balanced, ~2 GB VRAM,  recommended default ✓
#   "medium" →  accurate, ~5 GB VRAM
#   "turbo"  →  best,     ~6 GB VRAM,  large-v3 quality at faster speed
_WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")

_whisper_model: Optional[WhisperModel] = None


def _get_whisper_model() -> WhisperModel:
    """
    Lazily load the faster-whisper model on first use.
    Runs on CPU with int8 quantisation by default.
    Set WHISPER_DEVICE=cuda in .env for GPU acceleration.
    """
    global _whisper_model
    if _whisper_model is None:
        device       = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        logger.info(
            "Loading faster-whisper model '%s' on %s (%s) …",
            _WHISPER_MODEL_SIZE, device, compute_type,
        )
        _whisper_model = WhisperModel(
            _WHISPER_MODEL_SIZE,
            device=device,
            compute_type=compute_type,
        )
        logger.info("faster-whisper model loaded successfully.")
    return _whisper_model


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers — synchronous (run inside asyncio.to_thread)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_video_id(url: str) -> str:
    """
    Extract the 11-character YouTube video ID from any URL format:
      • https://www.youtube.com/watch?v=VIDEO_ID
      • https://youtu.be/VIDEO_ID
      • https://www.youtube.com/embed/VIDEO_ID
      • https://www.youtube.com/shorts/VIDEO_ID
      • https://www.youtube.com/live/VIDEO_ID

    Raises:
        ValueError: If no valid video ID can be parsed.
    """
    patterns = [
        r"[?&]v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"/(?:embed|v|shorts|live)/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    raise ValueError(
        f"Cannot extract a YouTube video ID from: '{url}'\n"
        "Supported formats: watch?v=, youtu.be/, /embed/, /shorts/, /live/"
    )


def _format_duration(seconds: int) -> str:
    """Convert raw seconds → 'HH:MM:SS' or 'MM:SS' string."""
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _parse_upload_date(raw: str) -> Optional[str]:
    """
    Convert yt-dlp's compact 'YYYYMMDD' string to ISO-8601 'YYYY-MM-DD'.
    Returns None for empty / malformed input.
    """
    if not raw or len(raw) != 8:
        return None
    try:
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    except Exception:
        return raw


def _best_thumbnail(thumbnails: list[dict]) -> Optional[str]:
    """
    Pick the thumbnail with the highest resolution from yt-dlp's list.
    Falls back to the first available thumbnail.
    """
    if not thumbnails:
        return None
    # yt-dlp thumbnails have optional 'width' and 'height' keys
    with_size = [t for t in thumbnails if t.get("width")]
    if with_size:
        return max(with_size, key=lambda t: t.get("width", 0)).get("url")
    return thumbnails[-1].get("url")


def _fetch_oembed_metadata(url: str) -> dict[str, Any]:
    """Fetch basic metadata from YouTube's official oEmbed API to bypass bot blockers."""
    oembed_url = f"https://www.youtube.com/oembed?url={urllib.parse.quote(url)}&format=json"
    try:
        req = urllib.request.Request(oembed_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())
            return {
                "title": data.get("title", "Unknown Title"),
                "uploader": data.get("author_name", "Unknown Creator"),
                "thumbnail": data.get("thumbnail_url"),
            }
    except Exception as exc:
        logger.warning("oEmbed fallback failed: %s", exc)
        return {}

def _fetch_metadata(url: str) -> dict[str, Any]:
    """
    Pull video metadata. Try yt-dlp first. If blocked by YouTube's anti-bot,
    fallback to oEmbed to gracefully degrade instead of crashing.
    """
    logger.debug("yt-dlp fetching metadata for: %s", url)
    meta = {}
    
    # 1. Try yt-dlp for rich metadata (views, likes, comments)
    try:
        with yt_dlp.YoutubeDL(_YDL_META_OPTS) as ydl:
            meta = ydl.extract_info(url, download=False) or {}
    except Exception as exc:
        logger.warning("yt-dlp blocked or failed to fetch metadata: %s", exc)
        
    # 2. If yt-dlp completely failed (no title), fallback to oEmbed
    if not meta.get("title"):
        logger.info("Falling back to YouTube oEmbed API for core metadata...")
        fallback_meta = _fetch_oembed_metadata(url)
        meta.update(fallback_meta)
        
    return meta


def _fetch_transcript_api(video_id: str) -> tuple[str, TranscriptSource]:
    """
    Attempt to fetch transcript via youtube-transcript-api 1.x.

    Strategy:
      1. Fetch English manual captions  → TranscriptSource.MANUAL
      2. Fetch any manual captions      → TranscriptSource.MANUAL
      3. Fetch auto-generated captions  → TranscriptSource.AUTO_GENERATED
      4. Return ("", TranscriptSource.UNAVAILABLE) if all fail

    Returns:
        (raw_transcript_text, TranscriptSource)
    """
    api = YouTubeTranscriptApi()

    # ── Attempt 1: English manual captions ───────────────────────────────────
    try:
        fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB", "en-CA"])
        text = " ".join(snip.text for snip in fetched)
        if text.strip():
            logger.info("Transcript obtained via manual captions [%s]", video_id)
            return text, TranscriptSource.MANUAL
    except NoTranscriptFound:
        logger.debug("No English manual captions for %s.", video_id)
    except TranscriptsDisabled:
        logger.warning("Transcripts are disabled for %s.", video_id)
        return "", TranscriptSource.UNAVAILABLE
    except Exception as exc:
        logger.debug("Attempt 1 failed for %s: %s", video_id, exc)

    # ── Attempt 2: Any manual caption (any language) ─────────────────────────
    try:
        transcript_list = api.list(video_id)
        # Prefer manually created over auto-generated
        for t in transcript_list:
            if not t.is_generated:
                fetched = t.fetch()
                text = " ".join(snip.text for snip in fetched)
                if text.strip():
                    logger.info(
                        "Transcript obtained via manual captions [lang=%s] for %s",
                        t.language_code, video_id,
                    )
                    return text, TranscriptSource.MANUAL
    except Exception as exc:
        logger.debug("Attempt 2 (manual any-lang) failed for %s: %s", video_id, exc)

    # ── Attempt 3: Auto-generated captions ───────────────────────────────────
    try:
        transcript_list = api.list(video_id)
        for t in transcript_list:
            if t.is_generated:
                fetched = t.fetch()
                text = " ".join(snip.text for snip in fetched)
                if text.strip():
                    logger.info(
                        "Transcript obtained via auto-generated captions [lang=%s] for %s",
                        t.language_code, video_id,
                    )
                    return text, TranscriptSource.AUTO_GENERATED
    except Exception as exc:
        logger.debug("Attempt 3 (auto-generated) failed for %s: %s", video_id, exc)

    logger.warning("No transcript available via youtube-transcript-api for %s.", video_id)
    return "", TranscriptSource.UNAVAILABLE


def _transcribe_with_whisper(video_id: str, url: str) -> tuple[str, TranscriptSource]:
    """
    Download the best audio stream and transcribe it locally with faster-whisper.

    Steps:
      1. Create a temp directory
      2. Use yt-dlp to download the best audio stream
      3. Pass the audio file to faster-whisper
      4. Concatenate all segments into a single transcript string
      5. Clean up temp directory

    Returns:
        (transcript_text, TranscriptSource.WHISPER) on success
        ("", TranscriptSource.UNAVAILABLE) on failure
    """
    tmpdir = tempfile.mkdtemp(prefix="yt_whisper_")
    audio_path: Optional[str] = None

    try:
        logger.info("Whisper fallback: downloading audio for %s …", video_id)

        ydl_opts = {
            **_YDL_AUDIO_OPTS_TEMPLATE,
            "outtmpl": os.path.join(tmpdir, f"{video_id}.%(ext)s"),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Determine the actual downloaded file path
            ext = info.get("ext", "webm")
            audio_path = os.path.join(tmpdir, f"{video_id}.{ext}")

        if not audio_path or not os.path.isfile(audio_path):
            logger.error("Audio file not found after yt-dlp download: %s", audio_path)
            return "", TranscriptSource.UNAVAILABLE

        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(
            "Audio downloaded (%.1f MB). Running faster-whisper [model=%s] …",
            file_size_mb, _WHISPER_MODEL_SIZE,
        )

        model = _get_whisper_model()
        segments, info_result = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,        # skip silent segments — faster & cleaner
            vad_parameters={
                "min_silence_duration_ms": 500,
            },
        )

        logger.info(
            "Whisper detected language: '%s' (prob=%.2f)",
            info_result.language,
            info_result.language_probability,
        )

        # Consume the generator and concatenate all segments
        text = " ".join(seg.text.strip() for seg in segments)

        if not text.strip():
            logger.warning("Whisper produced an empty transcript for %s.", video_id)
            return "", TranscriptSource.UNAVAILABLE

        logger.info(
            "Whisper transcription complete for %s (%d chars).",
            video_id, len(text),
        )
        return text, TranscriptSource.WHISPER

    except Exception as exc:
        logger.error("Whisper transcription failed for %s: %s", video_id, exc)
        return "", TranscriptSource.UNAVAILABLE

    finally:
        # Always clean up the temp directory regardless of success/failure
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
            logger.debug("Cleaned up temp directory: %s", tmpdir)
        except Exception:
            pass


def _get_transcript(video_id: str, url: str) -> tuple[str, TranscriptSource]:
    """
    Master transcript resolver — tries the transcript API first,
    falls back to Whisper if unavailable.

    Returns:
        (cleaned_transcript_text, TranscriptSource)
    """
    raw_text, source = _fetch_transcript_api(video_id)

    if not raw_text.strip():
        logger.info(
            "Transcript API returned nothing for %s. Falling back to Whisper …", video_id
        )
        raw_text, source = _transcribe_with_whisper(video_id, url)

    cleaned = clean_transcript(raw_text) if raw_text.strip() else ""
    return cleaned, source


def _build_video_data(
    video_id: str,
    url: str,
    meta: dict[str, Any],
    transcript: str,
    transcript_source: TranscriptSource,
) -> YouTubeVideoData:
    """
    Construct a fully validated YouTubeVideoData from raw yt-dlp metadata.
    """
    raw_duration: Optional[int] = meta.get("duration")

    # Select the best quality thumbnail
    thumbnails = meta.get("thumbnails") or []
    thumbnail  = _best_thumbnail(thumbnails) or meta.get("thumbnail")

    # yt-dlp comment count may be in 'comment_count' or nested
    comment_count: Optional[int] = (
        meta.get("comment_count")
        or meta.get("comments")
    )

    return YouTubeVideoData(
        # ── Identifiers ──────────────────────────────────────────────────────
        video_id=video_id,
        url=meta.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",

        # ── Core metadata ────────────────────────────────────────────────────
        title=meta.get("title") or "Unknown Title",
        creator=(
            meta.get("uploader")
            or meta.get("channel")
            or meta.get("uploader_id")
            or "Unknown Creator"
        ),
        description=meta.get("description"),
        upload_date=_parse_upload_date(meta.get("upload_date", "")),
        duration_seconds=raw_duration,
        duration_formatted=_format_duration(raw_duration) if raw_duration else None,
        thumbnail_url=thumbnail,
        language=meta.get("language"),

        # ── Engagement ───────────────────────────────────────────────────────
        views=meta.get("view_count"),
        likes=meta.get("like_count"),
        comments=comment_count,

        # ── Taxonomy ─────────────────────────────────────────────────────────
        tags=meta.get("tags") or [],
        categories=meta.get("categories") or [],

        # ── Transcript ───────────────────────────────────────────────────────
        transcript=transcript or "No transcript could be obtained.",
        transcript_source=transcript_source,

        # ── Raw payload (key fields only — avoid bloating the response) ──────
        raw_metadata={
            "video_id":      video_id,
            "channel_id":    meta.get("channel_id"),
            "channel_url":   meta.get("channel_url"),
            "webpage_url":   meta.get("webpage_url"),
            "upload_date":   meta.get("upload_date"),
            "duration":      raw_duration,
            "view_count":    meta.get("view_count"),
            "like_count":    meta.get("like_count"),
            "comment_count": comment_count,
            "age_limit":     meta.get("age_limit"),
            "is_live":       meta.get("is_live"),
            "availability":  meta.get("availability"),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public async interface
# ─────────────────────────────────────────────────────────────────────────────


async def extract_youtube(url: str) -> YouTubeVideoData:
    """
    Full async extraction pipeline for a YouTube video.

    All blocking operations (yt-dlp, transcript API, Whisper) run inside
    asyncio.to_thread() so the FastAPI event loop is never blocked.

    Args:
        url: Any valid public YouTube URL.

    Returns:
        YouTubeVideoData — fully validated Pydantic model.

    Raises:
        ValueError:   If the URL cannot be parsed (no video ID found).
        RuntimeError: If yt-dlp cannot retrieve metadata.
    """
    logger.info("▶  extract_youtube called | url=%s", url)

    # ── Step 1: Validate URL ─────────────────────────────────────────────────
    video_id = _extract_video_id(url)
    logger.info("   video_id resolved: %s", video_id)

    # ── Step 2 & 3: Metadata + Transcript in parallel ────────────────────────
    # Both are blocking — run concurrently in separate threads
    meta_task       = asyncio.to_thread(_fetch_metadata, url)
    transcript_task = asyncio.to_thread(_get_transcript, video_id, url)

    meta, (transcript, transcript_source) = await asyncio.gather(
        meta_task, transcript_task
    )

    logger.info(
        "   transcript_source=%s | transcript_length=%d chars",
        transcript_source.value, len(transcript),
    )

    # ── Step 4: Assemble structured response ─────────────────────────────────
    data = _build_video_data(
        video_id=video_id,
        url=url,
        meta=meta,
        transcript=transcript,
        transcript_source=transcript_source,
    )

    logger.info(
        "✓  extract_youtube complete | video_id=%s | title='%s' | "
        "views=%s | transcript_source=%s",
        data.video_id,
        data.title[:60],
        f"{data.views:,}" if data.views else "N/A",
        data.transcript_source.value,
    )

    return data
