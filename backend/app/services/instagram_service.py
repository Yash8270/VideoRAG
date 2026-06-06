"""
Production-ready Instagram Reel extraction service.

Full pipeline (yt-dlp only — no Instaloader dependency):
  1. Validate URL → extract shortcode
  2. Single yt-dlp call:
       • Fetches rich metadata (title, creator, views, likes, comments,
         upload date, duration, thumbnail, hashtags)
       • Downloads best-quality audio to  data/raw/instagram/<shortcode>.<ext>
  3. Resolve the exact audio file path from yt-dlp's info dict
  4. Return structured InstagramReelData (Pydantic model) + audio_path

The audio file is intentionally kept on disk so the caller (route or
service) can pass it to faster-whisper or any other STT engine for
transcription.

Authentication note:
  Public Reels work without credentials.
  For private/restricted content, supply cookies via INSTAGRAM_COOKIES_FILE
  in your .env.  yt-dlp will use the Netscape-format cookie file.
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yt_dlp
from pydantic import BaseModel, Field

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output directories
# ─────────────────────────────────────────────────────────────────────────────

# Persisted audio files go here — relative to the project root (RAG/)
# Override via INSTAGRAM_AUDIO_DIR env var if needed
_AUDIO_DIR: str = os.getenv("INSTAGRAM_AUDIO_DIR", "data/raw/instagram")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic response model
# ─────────────────────────────────────────────────────────────────────────────


class InstagramReelData(BaseModel):
    """
    Fully validated, structured output for an extracted Instagram Reel.
    Returned by extract_instagram().
    """

    # ── Identifiers ──────────────────────────────────────────────────────────
    shortcode:          str             = Field(..., description="Instagram shortcode (unique post ID)")
    url:                str             = Field(..., description="Canonical Reel URL")

    # ── Core metadata ────────────────────────────────────────────────────────
    title:              str             = Field(..., description="First line of caption or auto-title")
    creator:            str             = Field(..., description="Instagram username (@handle)")
    creator_display:    Optional[str]   = Field(None, description="Full display name if available")
    description:        Optional[str]   = Field(None, description="Full caption text")
    transcript:         Optional[str]   = Field(None, description="Extracted audio transcript")
    upload_date:        Optional[str]   = Field(None, description="ISO 8601 date — YYYY-MM-DD")
    upload_timestamp:   Optional[int]   = Field(None, description="Unix timestamp of upload")
    duration_seconds:   Optional[float]   = Field(None, description="Reel duration in seconds")
    duration_formatted: Optional[str]   = Field(None, description="Human-readable MM:SS")
    thumbnail_url:      Optional[str]   = Field(None, description="Highest-resolution thumbnail URL")

    # ── Engagement ───────────────────────────────────────────────────────────
    views:              Optional[int]   = Field(None, description="Play / view count")
    likes:              Optional[int]   = Field(None, description="Like count (None if hidden by IG)")
    comments:           Optional[int]   = Field(None, description="Comment count (None if hidden)")
    follower_count:     Optional[int]   = Field(None, description="Creator's Instagram follower count")

    # ── Taxonomy ─────────────────────────────────────────────────────────────
    hashtags:           list[str]       = Field(default_factory=list, description="Hashtags from caption")
    tags:               list[str]       = Field(default_factory=list, description="yt-dlp tags field")

    # ── Audio ─────────────────────────────────────────────────────────────────
    audio_path:         Optional[str]   = Field(
        None,
        description="Absolute path to the downloaded audio file. "
                    "Pass to faster-whisper for transcription.",
    )
    audio_format:       Optional[str]   = Field(None, description="Audio file extension (e.g. m4a, webm)")
    audio_size_mb:      Optional[float] = Field(None, description="Downloaded audio file size in MB")

    # ── Raw payload ──────────────────────────────────────────────────────────
    raw_metadata:       dict[str, Any]  = Field(
        default_factory=dict,
        description="Selected raw yt-dlp fields for downstream use",
    )


# ─────────────────────────────────────────────────────────────────────────────
# yt-dlp configuration
# ─────────────────────────────────────────────────────────────────────────────

def _build_ydl_opts(output_dir: str, shortcode: str) -> dict[str, Any]:
    """
    Build yt-dlp options for downloading best-quality audio + full metadata.

    The filename template uses the shortcode so files are deterministic and
    idempotent (re-ingesting the same Reel overwrites rather than duplicates).
    """
    opts: dict[str, Any] = {
        "quiet":       True,
        "no_warnings": True,
        # Best audio stream available (m4a / webm / mp4 — no re-encoding)
        "format":      "bestaudio/best",
        # Deterministic filename:  data/raw/instagram/<shortcode>.<ext>
        "outtmpl":     os.path.join(output_dir, f"{shortcode}.%(ext)s"),
        # Write thumbnail alongside audio (useful for UI display)
        "writethumbnail": False,
        # Pull comment count, like count (requires slightly more work from yt-dlp)
        "getcomments": False,
    }

    # Optional: supply Instagram cookies for private / age-restricted content
    cookies_file = os.getenv("INSTAGRAM_COOKIES_FILE")
    if not cookies_file or not Path(cookies_file).is_file():
        # Fallback check for the unified cookies file
        possible_paths = [
            "backend/cookies.txt",
            "cookies.txt",
            "/etc/secrets/cookies.txt",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../cookies.txt"))
        ]
        for path in possible_paths:
            if os.path.isfile(path):
                cookies_file = path
                break

    if cookies_file and Path(cookies_file).is_file():
        opts["cookiefile"] = cookies_file
        logger.debug("Using Instagram cookies from: %s", cookies_file)

    return opts


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers — synchronous (called via asyncio.to_thread)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_shortcode(url: str) -> str:
    """
    Parse the unique shortcode from any Instagram Reel / post URL:
      • https://www.instagram.com/reel/ABC123xyz/
      • https://www.instagram.com/reels/ABC123xyz/
      • https://www.instagram.com/p/ABC123xyz/
      • https://instagr.am/reel/ABC123xyz/

    Raises:
        ValueError: If no valid shortcode pattern is found.
    """
    patterns = [
        r"/(?:reel|reels|p)/([A-Za-z0-9_-]+)/?",
        r"instagram\.com/([A-Za-z0-9_-]{10,})",   # fallback for unusual formats
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)

    raise ValueError(
        f"Cannot extract an Instagram shortcode from: '{url}'\n"
        "Supported formats:\n"
        "  • https://www.instagram.com/reel/<shortcode>/\n"
        "  • https://www.instagram.com/p/<shortcode>/"
    )


def _format_duration(seconds: int) -> str:
    """Convert raw seconds → 'MM:SS' or 'HH:MM:SS' string."""
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _parse_upload_date(raw: str) -> Optional[str]:
    """Convert yt-dlp 'YYYYMMDD' → ISO 8601 'YYYY-MM-DD'. Returns None on failure."""
    if not raw or len(raw) != 8:
        return None
    try:
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    except Exception:
        return raw


def _extract_hashtags(text: Optional[str]) -> list[str]:
    """Pull all #hashtag tokens from caption text."""
    if not text:
        return []
    return re.findall(r"#([A-Za-z0-9_]+)", text)


def _best_thumbnail(thumbnails: list[dict]) -> Optional[str]:
    """Return URL of the highest-resolution thumbnail from yt-dlp's list."""
    if not thumbnails:
        return None
    sized = [t for t in thumbnails if t.get("width") and t.get("url")]
    if sized:
        return max(sized, key=lambda t: t.get("width", 0))["url"]
    for t in reversed(thumbnails):
        if t.get("url"):
            return t["url"]
    return None


def _resolve_audio_path(info: dict[str, Any], output_dir: str, shortcode: str) -> Optional[str]:
    """
    Resolve the exact path of the downloaded audio file.

    Strategy (most reliable first):
      1. info['requested_downloads'][0]['filepath']  — yt-dlp's authoritative path
      2. Constructed path: output_dir/<shortcode>.<ext>
      3. Glob for any file starting with shortcode in output_dir
    """
    # Strategy 1 — authoritative yt-dlp path
    try:
        path = (
            info.get("requested_downloads", [{}])[0]
            .get("filepath")
        )
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    except (IndexError, TypeError, KeyError):
        pass

    # Strategy 2 — constructed from extension
    ext = info.get("ext") or info.get("audio_ext") or "webm"
    constructed = os.path.join(output_dir, f"{shortcode}.{ext}")
    if os.path.isfile(constructed):
        return os.path.abspath(constructed)

    # Strategy 3 — glob fallback
    try:
        matches = list(Path(output_dir).glob(f"{shortcode}.*"))
        if matches:
            return str(matches[0].resolve())
    except Exception:
        pass

    logger.warning("Could not resolve audio file path for shortcode %s", shortcode)
    return None


def _ensure_output_dir(path: str) -> str:
    """Create the output directory if it doesn't exist. Returns the path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _download_reel(url: str, shortcode: str) -> tuple[dict[str, Any], Optional[str]]:
    """
    Core blocking function: run yt-dlp to download audio + fetch metadata.

    Args:
        url:       Instagram Reel URL.
        shortcode: Pre-parsed shortcode (used for filename and logging).

    Returns:
        (info_dict, audio_file_path)
        audio_file_path is None if the download failed.

    Raises:
        RuntimeError: If yt-dlp raises a DownloadError.
    """
    output_dir = _ensure_output_dir(_AUDIO_DIR)
    ydl_opts   = _build_ydl_opts(output_dir, shortcode)

    logger.info("yt-dlp downloading Reel audio [shortcode=%s] → %s", shortcode, output_dir)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(
            f"yt-dlp could not download the Instagram Reel.\n"
            f"Shortcode: {shortcode}\n"
            f"Reason: {exc}\n\n"
            "Common causes:\n"
            "  • Post is private → supply INSTAGRAM_COOKIES_FILE in .env\n"
            "  • Post was deleted\n"
            "  • Instagram rate-limited this IP\n"
            "  • Outdated yt-dlp → run: pip install -U yt-dlp"
        ) from exc

    if not info:
        raise RuntimeError(f"yt-dlp returned no data for shortcode: {shortcode}")

    audio_path = _resolve_audio_path(info, output_dir, shortcode)

    if audio_path:
        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(
            "Audio downloaded: %s  (%.2f MB, format=%s)",
            audio_path, size_mb, info.get("ext", "?"),
        )
    else:
        logger.error("Audio file could not be located after download for %s", shortcode)

    return info, audio_path


def _build_reel_data(
    shortcode: str,
    url: str,
    info: dict[str, Any],
    audio_path: Optional[str],
) -> InstagramReelData:
    """
    Construct a fully validated InstagramReelData from raw yt-dlp info.
    Handles missing / None fields gracefully for all optional attributes.
    """
    # ── Duration ─────────────────────────────────────────────────────────────
    raw_duration: Optional[int] = info.get("duration")
    duration_fmt = _format_duration(raw_duration) if raw_duration else None

    # ── Thumbnail ─────────────────────────────────────────────────────────────
    thumbnails   = info.get("thumbnails") or []
    thumbnail    = _best_thumbnail(thumbnails) or info.get("thumbnail")

    # ── Caption / description ─────────────────────────────────────────────────
    description  = info.get("description") or info.get("caption")

    # ── Title: use first non-empty line of caption, fallback to generic title ─
    title = info.get("title") or ""
    if not title.strip() and description:
        first_line = description.strip().splitlines()[0]
        title = first_line[:120].strip() or f"Instagram Reel by @{info.get('uploader_id', 'unknown')}"
    if not title.strip():
        title = f"Instagram Reel by @{info.get('uploader_id', 'unknown')}"

    # ── Creator ───────────────────────────────────────────────────────────────
    creator = (
        info.get("uploader_id")
        or info.get("uploader")
        or info.get("channel")
        or "unknown"
    )
    # Strip leading @ if yt-dlp includes it
    creator = creator.lstrip("@")

    # ── Engagement ────────────────────────────────────────────────────────────
    views          = info.get("view_count")   or info.get("play_count")
    likes          = info.get("like_count")
    comments       = info.get("comment_count")
    follower_count = info.get("channel_follower_count")

    # ── Hashtags ──────────────────────────────────────────────────────────────
    hashtags = _extract_hashtags(description)

    # ── Audio file size ───────────────────────────────────────────────────────
    audio_size_mb: Optional[float] = None
    if audio_path and os.path.isfile(audio_path):
        audio_size_mb = round(os.path.getsize(audio_path) / (1024 * 1024), 2)

    # ── Upload timestamp ──────────────────────────────────────────────────────
    upload_ts: Optional[int] = info.get("timestamp")

    return InstagramReelData(
        # Identifiers
        shortcode=shortcode,
        url=info.get("webpage_url") or url,

        # Core metadata
        title=title,
        creator=creator,
        creator_display=info.get("uploader") if info.get("uploader") != creator else None,
        description=description,
        upload_date=_parse_upload_date(info.get("upload_date", "")),
        upload_timestamp=upload_ts,
        duration_seconds=raw_duration,
        duration_formatted=duration_fmt,
        thumbnail_url=thumbnail,

        # Engagement
        views=views,
        likes=likes,
        comments=comments,
        follower_count=follower_count,

        # Taxonomy
        hashtags=hashtags,
        tags=info.get("tags") or [],

        # Audio
        audio_path=audio_path,
        audio_format=info.get("ext") or (Path(audio_path).suffix.lstrip(".") if audio_path else None),
        audio_size_mb=audio_size_mb,

        # Raw payload — key fields only
        raw_metadata={
            "shortcode":     shortcode,
            "uploader_id":   info.get("uploader_id"),
            "uploader":      info.get("uploader"),
            "channel_id":    info.get("channel_id"),
            "webpage_url":   info.get("webpage_url"),
            "upload_date":   info.get("upload_date"),
            "timestamp":     upload_ts,
            "duration":      raw_duration,
            "view_count":    views,
            "like_count":    likes,
            "comment_count": comments,
            "follower_count": follower_count,
            "ext":           info.get("ext"),
            "format":        info.get("format"),
            "format_id":     info.get("format_id"),
            "filesize":      info.get("filesize") or info.get("filesize_approx"),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public async interface
# ─────────────────────────────────────────────────────────────────────────────


async def extract_instagram(url: str) -> InstagramReelData:
    """
    Full async extraction pipeline for a public Instagram Reel.

    Downloads the best-quality audio and captures all available metadata
    in a single yt-dlp call. The blocking download runs in a thread pool
    so the FastAPI event loop is never blocked.

    Args:
        url: Public Instagram Reel URL (reel/ or p/ format).

    Returns:
        InstagramReelData — fully validated Pydantic model including
        audio_path pointing to the downloaded audio file on disk.

    Raises:
        ValueError:   If the URL cannot be parsed.
        RuntimeError: If yt-dlp cannot fetch / download the Reel.

    Example:
        data = await extract_instagram("https://www.instagram.com/reel/ABC123/")
        print(data.title, data.views, data.audio_path)
        # → Pass data.audio_path to faster-whisper for transcription
    """
    logger.info("▶  extract_instagram called | url=%s", url)

    # ── Step 1: Validate URL ──────────────────────────────────────────────────
    shortcode = _extract_shortcode(url)
    logger.info("   shortcode resolved: %s", shortcode)

    # ── Step 2: Download audio + fetch metadata (blocking → thread) ───────────
    info, audio_path = await asyncio.to_thread(_download_reel, url, shortcode)

    # ── Step 3: Build structured response ────────────────────────────────────
    data = _build_reel_data(shortcode, url, info, audio_path)

    logger.info(
        "✓  extract_instagram complete | shortcode=%s | creator=@%s | "
        "views=%s | duration=%s | audio=%s",
        data.shortcode,
        data.creator,
        f"{data.views:,}" if data.views else "N/A",
        data.duration_formatted or "N/A",
        data.audio_path or "MISSING",
    )

    return data


async def delete_audio(data: InstagramReelData) -> bool:
    """
    Remove the downloaded audio file from disk once transcription is done.

    Args:
        data: InstagramReelData returned by extract_instagram().

    Returns:
        True if deleted, False if the file didn't exist or deletion failed.
    """
    if not data.audio_path:
        return False
    try:
        path = Path(data.audio_path)
        if path.is_file():
            await asyncio.to_thread(path.unlink)
            logger.info("Deleted audio file: %s", data.audio_path)
            return True
        return False
    except Exception as exc:
        logger.warning("Could not delete audio file %s: %s", data.audio_path, exc)
        return False
