"""
Chunking service — splits video transcripts into overlapping text chunks
using LangChain's RecursiveCharacterTextSplitter.

Default settings (as specified):
    chunk_size    = 500  characters
    chunk_overlap = 50   characters

Each chunk carries structured metadata:
    {
        "video_id":        "abc123",
        "chunk_id":        1,          ← 1-indexed
        "source":          "youtube",
        "title":           "Video title",
        "url":             "https://...",
        "total_chunks":    24,
        "char_start":      0,
        "char_end":        498,
        "chunk_length":    498
    }

Public interface:
    chunk_transcript(text, video_id, ...)   → ChunkingResult
    chunk_from_youtube(data, ...)           → ChunkingResult
    chunk_from_instagram(data, ...)         → ChunkingResult
    to_langchain_docs(result)               → list[Document]
    to_chroma_payload(result)               → ChromaPayload
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.services.instagram_service import InstagramReelData
    from app.services.youtube_service import YouTubeVideoData

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE:    int = 500
DEFAULT_CHUNK_OVERLAP: int = 50

# Separator priority — tried left-to-right; falls back to the next on failure.
# Designed specifically for video transcripts that lack punctuation structure.
_SEPARATORS: list[str] = [
    "\n\n",   # paragraph breaks (rare in transcripts but highest priority)
    "\n",     # line breaks
    ". ",     # sentence boundary
    "! ",     # exclamation end
    "? ",     # question end
    ", ",     # clause boundary
    " ",      # word boundary (most common fallback)
    "",       # character-level last resort
]


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────


class ChunkMetadata(BaseModel):
    """
    Metadata attached to every produced chunk.
    Minimum required fields: video_id + chunk_id.
    Additional fields enrich downstream retrieval and debugging.
    """

    # ── Required (as specified) ───────────────────────────────────────────────
    video_id:     str = Field(..., description="Unique identifier of the source video")
    chunk_id:     int = Field(..., description="1-indexed position of this chunk in the sequence")

    # ── Source context ────────────────────────────────────────────────────────
    source:       str           = Field("unknown",  description="'youtube' or 'instagram'")
    title:        str           = Field("",         description="Video / Reel title")
    url:          str           = Field("",         description="Canonical content URL")
    creator:      str           = Field("",         description="Channel name or @username")
    upload_date:  Optional[str] = Field(None,       description="ISO 8601 upload date")

    # ── Chunk position ────────────────────────────────────────────────────────
    total_chunks: int = Field(..., description="Total number of chunks for this video")
    char_start:   int = Field(..., description="Start character offset in the original transcript")
    char_end:     int = Field(..., description="End character offset in the original transcript")
    chunk_length: int = Field(..., description="Character length of this chunk's text")

    # ── Deterministic ID ─────────────────────────────────────────────────────
    chunk_hash:   str = Field(
        ...,
        description="SHA-256 (first 16 chars) of video_id + chunk_id — stable unique ID for upsert",
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for ChromaDB metadata (no nested objects)."""
        return {
            "video_id":    self.video_id,
            "chunk_id":    self.chunk_id,
            "source":      self.source,
            "title":       self.title,
            "url":         self.url,
            "creator":     self.creator,
            "upload_date": self.upload_date or "",
            "total_chunks":self.total_chunks,
            "char_start":  self.char_start,
            "char_end":    self.char_end,
            "chunk_length":self.chunk_length,
            "chunk_hash":  self.chunk_hash,
        }


class TextChunk(BaseModel):
    """A single chunk of transcript text paired with its metadata."""

    text:     str           = Field(..., description="The chunk text content")
    metadata: ChunkMetadata = Field(..., description="Structured metadata for this chunk")

    def to_langchain_doc(self) -> Document:
        """Convert to a LangChain Document for embedding pipelines."""
        return Document(
            page_content=self.text,
            metadata=self.metadata.to_dict(),
        )


class ChunkingResult(BaseModel):
    """Full output of a chunking operation — chunks + statistics."""

    # ── Core output ───────────────────────────────────────────────────────────
    chunks:      list[TextChunk] = Field(..., description="All produced text chunks")
    video_id:    str             = Field(..., description="Source video identifier")
    source:      str             = Field(..., description="'youtube' or 'instagram'")

    # ── Statistics ────────────────────────────────────────────────────────────
    total_chunks:        int   = Field(..., description="Number of chunks produced")
    transcript_length:   int   = Field(..., description="Total character length of input transcript")
    avg_chunk_length:    float = Field(..., description="Mean character length per chunk")
    min_chunk_length:    int   = Field(..., description="Shortest chunk character length")
    max_chunk_length:    int   = Field(..., description="Longest chunk character length")
    chunk_size_setting:  int   = Field(..., description="chunk_size used for splitting")
    chunk_overlap_setting:int  = Field(..., description="chunk_overlap used for splitting")

    def to_langchain_docs(self) -> list[Document]:
        """Convert all chunks to LangChain Documents (for use with embedders)."""
        return [chunk.to_langchain_doc() for chunk in self.chunks]

    def to_texts_and_metadatas(self) -> tuple[list[str], list[dict]]:
        """
        Return parallel (texts, metadatas) lists — the format expected by
        ChromaDB's collection.add() / collection.upsert().
        """
        texts     = [c.text for c in self.chunks]
        metadatas = [c.metadata.to_dict() for c in self.chunks]
        return texts, metadatas

    def to_ids(self) -> list[str]:
        """
        Return stable chunk IDs for ChromaDB upsert.
        Format: <video_id>_chunk_<chunk_id>
        """
        return [f"{c.metadata.video_id}_chunk_{c.metadata.chunk_id}" for c in self.chunks]


class ChromaPayload(BaseModel):
    """Ready-to-use payload for chromadb collection.upsert()."""
    ids:        list[str]        = Field(..., description="Stable chunk IDs")
    documents:  list[str]        = Field(..., description="Chunk text strings")
    metadatas:  list[dict]       = Field(..., description="Flat metadata dicts")


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_splitter(
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """
    Build a RecursiveCharacterTextSplitter tuned for video transcripts.

    Args:
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Shared characters between adjacent chunks.

    Returns:
        Configured splitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
        keep_separator=False,      # don't include separator char in chunk text
    )


def _chunk_hash(video_id: str, chunk_id: int) -> str:
    """
    Generate a short deterministic hash for a chunk — stable across runs.
    Used as a ChromaDB document ID for idempotent upserts.

    Args:
        video_id:  Source video identifier.
        chunk_id:  1-indexed chunk sequence number.

    Returns:
        First 16 characters of the SHA-256 hex digest.
    """
    raw = f"{video_id}::{chunk_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _find_char_offset(original: str, chunk_text: str, search_from: int = 0) -> int:
    """
    Locate where chunk_text begins in the original transcript.
    Uses a forward search starting from search_from to handle repeated phrases.

    Args:
        original:    The full original transcript.
        chunk_text:  The chunk text to find.
        search_from: Offset to start searching from (avoids matching earlier chunk).

    Returns:
        Character index or -1 if not found.
    """
    # Try exact match first (fast path)
    idx = original.find(chunk_text.strip(), search_from)
    if idx != -1:
        return idx

    # Fallback: match on first 60 chars (handles minor whitespace differences)
    prefix = chunk_text.strip()[:60]
    idx = original.find(prefix, search_from)
    return idx if idx != -1 else search_from


def _build_chunks(
    raw_chunks: list[str],
    original_transcript: str,
    video_id: str,
    source: str,
    title: str,
    url: str,
    creator: str,
    upload_date: Optional[str],
) -> list[TextChunk]:
    """
    Pair each raw text chunk with its full ChunkMetadata.

    Args:
        raw_chunks:          List of plain text strings from the splitter.
        original_transcript: The original input transcript for offset calculation.
        video_id:            Source video identifier.
        source, title, url, creator, upload_date: Video context fields.

    Returns:
        List of TextChunk objects.
    """
    total = len(raw_chunks)
    result: list[TextChunk] = []
    search_cursor = 0   # tracks where to start offset search for next chunk

    for idx, text in enumerate(raw_chunks):
        chunk_id   = idx + 1    # 1-indexed as required
        text       = text.strip()
        length     = len(text)

        # Compute character offsets in the original transcript
        char_start = _find_char_offset(original_transcript, text, search_cursor)
        char_end   = char_start + length if char_start >= 0 else -1

        # Advance search cursor past the current chunk start (minus overlap)
        if char_start >= 0:
            search_cursor = max(search_cursor, char_start + max(0, length - DEFAULT_CHUNK_OVERLAP))

        meta = ChunkMetadata(
            video_id=video_id,
            chunk_id=chunk_id,
            source=source,
            title=title,
            url=url,
            creator=creator,
            upload_date=upload_date,
            total_chunks=total,
            char_start=max(char_start, 0),
            char_end=max(char_end, 0),
            chunk_length=length,
            chunk_hash=_chunk_hash(video_id, chunk_id),
        )

        result.append(TextChunk(text=text, metadata=meta))

    return result


def _compute_stats(chunks: list[TextChunk]) -> dict[str, Any]:
    """Compute descriptive statistics over the produced chunks."""
    lengths = [c.metadata.chunk_length for c in chunks]
    return {
        "total_chunks":  len(chunks),
        "avg_chunk_length": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        "min_chunk_length": min(lengths) if lengths else 0,
        "max_chunk_length": max(lengths) if lengths else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────


def chunk_transcript(
    transcript: str,
    video_id: str,
    *,
    source:       str           = "unknown",
    title:        str           = "",
    url:          str           = "",
    creator:      str           = "",
    upload_date:  Optional[str] = None,
    chunk_size:   int           = DEFAULT_CHUNK_SIZE,
    chunk_overlap:int           = DEFAULT_CHUNK_OVERLAP,
) -> ChunkingResult:
    """
    Split a transcript into overlapping chunks using
    LangChain's RecursiveCharacterTextSplitter.

    Args:
        transcript:    Full cleaned transcript text to split.
        video_id:      Unique identifier for the source video (attached to every chunk).

        source:        Platform label — 'youtube' or 'instagram'.
        title:         Video title (attached to every chunk for context).
        url:           Canonical video URL.
        creator:       Channel name or @username.
        upload_date:   ISO 8601 date string.

        chunk_size:    Maximum characters per chunk  (default: 500).
        chunk_overlap: Shared characters between adjacent chunks (default: 50).

    Returns:
        ChunkingResult containing all TextChunks, each with:
            metadata.video_id  = video_id
            metadata.chunk_id  = 1-indexed chunk number
            + source, title, url, creator, upload_date, offsets, hash

    Raises:
        ValueError: If transcript is empty or video_id is blank.

    Example:
        result = chunk_transcript(
            transcript="Hello world. This is a test transcript...",
            video_id="dQw4w9WgXcQ",
            source="youtube",
            title="Never Gonna Give You Up",
            url="https://youtu.be/dQw4w9WgXcQ",
            creator="Rick Astley",
        )
        print(result.total_chunks)
        print(result.chunks[0].metadata.video_id)  # "dQw4w9WgXcQ"
        print(result.chunks[0].metadata.chunk_id)  # 1
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if not video_id or not video_id.strip():
        raise ValueError("video_id must be a non-empty string.")

    # Safely convert lists (e.g. from whisper segments or YT API) to a single string
    if isinstance(transcript, list):
        if transcript and isinstance(transcript[0], dict):
            transcript = " ".join(t.get("text", "") for t in transcript if isinstance(t, dict))
        else:
            transcript = " ".join(str(getattr(t, "text", t)) for t in transcript)
            
    if transcript is None:
        transcript = ""
    elif not isinstance(transcript, str):
        transcript = str(transcript)

    if not transcript or not transcript.strip():
        logger.warning(
            "chunk_transcript called with empty transcript for video_id='%s'. "
            "Returning empty ChunkingResult.",
            video_id,
        )
        return ChunkingResult(
            chunks=[],
            video_id=video_id,
            source=source,
            total_chunks=0,
            transcript_length=0,
            avg_chunk_length=0.0,
            min_chunk_length=0,
            max_chunk_length=0,
            chunk_size_setting=chunk_size,
            chunk_overlap_setting=chunk_overlap,
        )

    transcript = transcript.strip()
    transcript_length = len(transcript)

    logger.info(
        "Chunking transcript  [video_id=%s | source=%s | len=%d chars | "
        "chunk_size=%d | overlap=%d]",
        video_id, source, transcript_length, chunk_size, chunk_overlap,
    )

    # ── Split ─────────────────────────────────────────────────────────────────
    splitter   = _make_splitter(chunk_size, chunk_overlap)
    raw_chunks = splitter.split_text(transcript)

    if not raw_chunks:
        logger.warning("Splitter produced 0 chunks for video_id='%s'.", video_id)
        return ChunkingResult(
            chunks=[],
            video_id=video_id,
            source=source,
            total_chunks=0,
            transcript_length=transcript_length,
            avg_chunk_length=0.0,
            min_chunk_length=0,
            max_chunk_length=0,
            chunk_size_setting=chunk_size,
            chunk_overlap_setting=chunk_overlap,
        )

    # ── Attach metadata ───────────────────────────────────────────────────────
    chunks = _build_chunks(
        raw_chunks=raw_chunks,
        original_transcript=transcript,
        video_id=video_id,
        source=source,
        title=title,
        url=url,
        creator=creator,
        upload_date=upload_date,
    )

    stats = _compute_stats(chunks)

    logger.info(
        "✓ Chunking complete  [video_id=%s | chunks=%d | avg_len=%.0f | "
        "min=%d | max=%d]",
        video_id,
        stats["total_chunks"],
        stats["avg_chunk_length"],
        stats["min_chunk_length"],
        stats["max_chunk_length"],
    )

    return ChunkingResult(
        chunks=chunks,
        video_id=video_id,
        source=source,
        total_chunks=stats["total_chunks"],
        transcript_length=transcript_length,
        avg_chunk_length=stats["avg_chunk_length"],
        min_chunk_length=stats["min_chunk_length"],
        max_chunk_length=stats["max_chunk_length"],
        chunk_size_setting=chunk_size,
        chunk_overlap_setting=chunk_overlap,
    )


def chunk_from_youtube(
    data: "YouTubeVideoData",
    *,
    chunk_size:    int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> ChunkingResult:
    """
    Convenience wrapper — chunk directly from a YouTubeVideoData object.

    Args:
        data:          YouTubeVideoData returned by youtube_service.extract_youtube().
        chunk_size:    Characters per chunk (default: 500).
        chunk_overlap: Overlap between adjacent chunks (default: 50).

    Returns:
        ChunkingResult with all chunks tagged as source='youtube'.
    """
    return chunk_transcript(
        transcript=data.transcript,
        video_id=data.video_id,
        source="youtube",
        title=data.title,
        url=data.url,
        creator=data.creator,
        upload_date=data.upload_date,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def chunk_from_instagram(
    data: "InstagramReelData",
    *,
    chunk_size:    int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> ChunkingResult:
    """
    Convenience wrapper — chunk directly from an InstagramReelData object.

    Args:
        data:          InstagramReelData returned by instagram_service.extract_instagram().
        chunk_size:    Characters per chunk (default: 500).
        chunk_overlap: Overlap between adjacent chunks (default: 50).

    Returns:
        ChunkingResult with all chunks tagged as source='instagram'.

    Note:
        data.description (caption) is used as the transcript when no
        dedicated transcript field is present on InstagramReelData.
    """
    transcript = getattr(data, "transcript", None) or data.description or ""

    return chunk_transcript(
        transcript=transcript,
        video_id=data.shortcode,
        source="instagram",
        title=data.title,
        url=data.url,
        creator=data.creator,
        upload_date=data.upload_date,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def to_langchain_docs(result: ChunkingResult) -> list[Document]:
    """
    Convert a ChunkingResult to a list of LangChain Document objects.
    Ready to pass directly to an embedder or vector store.

    Args:
        result: ChunkingResult from any chunk_* function.

    Returns:
        list[Document], each with page_content=chunk.text and metadata dict.
    """
    return result.to_langchain_docs()


def to_chroma_payload(result: ChunkingResult) -> ChromaPayload:
    """
    Convert a ChunkingResult into the exact format expected by
    chromadb collection.add() / collection.upsert().

    Args:
        result: ChunkingResult from any chunk_* function.

    Returns:
        ChromaPayload(ids, documents, metadatas)

    Example:
        payload = to_chroma_payload(result)
        collection.upsert(
            ids=payload.ids,
            documents=payload.documents,
            metadatas=payload.metadatas,
            embeddings=...,   # from your embedder
        )
    """
    texts, metadatas = result.to_texts_and_metadatas()
    return ChromaPayload(
        ids=result.to_ids(),
        documents=texts,
        metadatas=metadatas,
    )
