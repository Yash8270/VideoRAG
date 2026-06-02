"""
Transcription service — converts any audio file to timestamped text
using faster-whisper (an optimised CTranslate2 implementation of Whisper).

Primary output format (as requested):
    [
      {"start": 0.0, "end": 4.2, "text": "..."},
      ...
    ]

The richer TranscriptionResult model is also returned and gives access to:
  • Language detection (code + probability)
  • Per-segment confidence scores   (derived from avg_logprob)
  • Per-segment noise filter score  (no_speech_prob)
  • Optional word-level timestamps  (word_timestamps=True)
  • Full joined transcript string
  • Transcription wall-clock time

Model lifecycle:
  • The WhisperModel is a lazy singleton — loaded on first use and reused
    for all subsequent calls (loading takes 2–5 s the first time).
  • Call preload_model() at app startup to pay this cost once.
  • Config via environment variables (see .env.example):
      WHISPER_MODEL_SIZE  tiny | base | small | medium | turbo  (default: small)
      WHISPER_DEVICE      cpu  | cuda                           (default: cpu)
      WHISPER_COMPUTE_TYPE int8 | float16 | float32             (default: int8)
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
from pydantic import BaseModel, Field

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model configuration — read from environment (mirrors .env.example)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_SIZE:    str = os.getenv("WHISPER_MODEL_SIZE",   "small")
_DEVICE:        str = os.getenv("WHISPER_DEVICE",       "cpu")
_COMPUTE_TYPE:  str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Audio extensions faster-whisper / PyAV can handle natively
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".mp3", ".mp4", ".m4a", ".wav", ".flac",
    ".ogg", ".opus", ".webm", ".aac", ".wma",
    ".mkv", ".avi", ".mov",
})

# Segments whose no_speech_prob exceeds this threshold are flagged
_NOISE_THRESHOLD: float = 0.6


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────


class WordTimestamp(BaseModel):
    """Word-level timing (only populated when word_timestamps=True)."""
    start:       float = Field(..., description="Word start time in seconds")
    end:         float = Field(..., description="Word end time in seconds")
    word:        str   = Field(..., description="The word token")
    probability: float = Field(..., description="Token probability (0–1)")


class TranscriptSegment(BaseModel):
    """
    A single timed segment of transcribed speech.

    The minimal required output is {start, end, text}.
    All other fields are informational and can be ignored by callers
    that only need the simple timestamp format.
    """
    start: float = Field(..., description="Segment start time in seconds")
    end:   float = Field(..., description="Segment end time in seconds")
    text:  str   = Field(..., description="Transcribed text for this segment")

    # ── Quality indicators ───────────────────────────────────────────────────
    confidence:          Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="Segment confidence score (0–1). Derived from avg_logprob.",
    )
    no_speech_probability: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="Probability that segment is silence/noise (0–1). "
                    "Segments above 0.6 are likely background noise.",
    )
    is_noise: Optional[bool] = Field(
        None,
        description=f"True if no_speech_probability > {_NOISE_THRESHOLD}",
    )

    # ── Word-level detail (opt-in via word_timestamps=True) ──────────────────
    words: Optional[list[WordTimestamp]] = Field(
        None,
        description="Per-word timing. Populated only when word_timestamps=True.",
    )


class TranscriptionResult(BaseModel):
    """
    Full structured output of a transcription run.
    segments contains the primary timestamped data.
    Use segments_to_json(result) for the minimal [{start, end, text}] format.
    """

    # ── Primary data ─────────────────────────────────────────────────────────
    segments:     list[TranscriptSegment] = Field(..., description="All timed transcript segments")
    full_text:    str                     = Field(..., description="Complete transcript as a single string")

    # ── Language detection ───────────────────────────────────────────────────
    language:             str   = Field(..., description="Detected language code (e.g. 'en', 'hi', 'es')")
    language_probability: float = Field(..., description="Confidence in language detection (0–1)")

    # ── Audio info ───────────────────────────────────────────────────────────
    audio_path:      str           = Field(..., description="Path to the source audio file")
    duration_seconds: float        = Field(..., description="Total audio duration in seconds")

    # ── Statistics ───────────────────────────────────────────────────────────
    word_count:               int   = Field(..., description="Approximate word count")
    segment_count:            int   = Field(..., description="Total number of segments")
    noise_segment_count:      int   = Field(..., description="Segments flagged as likely noise")
    avg_confidence:           float = Field(..., description="Mean confidence across all segments")
    transcription_time_seconds: float = Field(..., description="Wall-clock time spent transcribing")

    # ── Model info ───────────────────────────────────────────────────────────
    model_size:    str = Field(..., description="Whisper model size used")
    device:        str = Field(..., description="Hardware device (cpu / cuda)")
    compute_type:  str = Field(..., description="Quantisation type (int8 / float16 / float32)")


# ─────────────────────────────────────────────────────────────────────────────
# Whisper model — lazy module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_model: Optional[WhisperModel] = None


def get_model() -> WhisperModel:
    """
    Lazily load and cache the WhisperModel.

    Thread-safe enough for asyncio (single-threaded event loop).
    Called by transcribe() on first use, or by preload_model() at startup.

    Returns:
        Loaded WhisperModel instance.
    """
    global _model
    if _model is None:
        logger.info(
            "Loading faster-whisper model  [size=%s | device=%s | compute=%s] …",
            _MODEL_SIZE, _DEVICE, _COMPUTE_TYPE,
        )
        t0 = time.perf_counter()
        _model = WhisperModel(
            _MODEL_SIZE,
            device=_DEVICE,
            compute_type=_COMPUTE_TYPE,
        )
        elapsed = time.perf_counter() - t0
        logger.info("faster-whisper model loaded in %.2f s.", elapsed)
    return _model


def model_info() -> dict:
    """Return current Whisper model configuration (useful for /health endpoint)."""
    return {
        "model_size":    _MODEL_SIZE,
        "device":        _DEVICE,
        "compute_type":  _COMPUTE_TYPE,
        "loaded":        _model is not None,
    }


async def preload_model() -> None:
    """
    Eagerly load the model in a background thread.
    Call this from the FastAPI lifespan startup handler to avoid a
    cold-start penalty on the first transcription request.
    """
    logger.info("Preloading faster-whisper model …")
    await asyncio.to_thread(get_model)
    logger.info("faster-whisper model preloaded and ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers — synchronous
# ─────────────────────────────────────────────────────────────────────────────


def _validate_audio_file(path: str) -> Path:
    """
    Verify the audio file exists and has a recognised extension.

    Args:
        path: Absolute or relative path to the audio file.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the extension is not supported.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Audio file not found: {path}\n"
            "Ensure the file was downloaded before calling transcribe()."
        )
    if not p.is_file():
        raise ValueError(f"Path exists but is not a file: {path}")

    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{ext}' for file: {path}\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return p


def _logprob_to_confidence(avg_logprob: float) -> float:
    """
    Convert Whisper's average log-probability to a human-readable 0–1 score.

    avg_logprob is typically in the range [-1, 0].
    exp(0) = 1.0  → perfect confidence.
    exp(-1) ≈ 0.37 → low confidence.
    Values below -1 are clamped to 0.0.
    """
    return round(max(0.0, min(1.0, math.exp(avg_logprob))), 4)


def _build_segment(raw) -> TranscriptSegment:
    """
    Convert a faster-whisper Segment namedtuple to a TranscriptSegment.

    Args:
        raw: faster-whisper Segment object.

    Returns:
        TranscriptSegment Pydantic model.
    """
    confidence      = _logprob_to_confidence(raw.avg_logprob)
    no_speech_prob  = round(float(raw.no_speech_prob), 4)
    is_noise        = no_speech_prob > _NOISE_THRESHOLD

    # ── Word-level timestamps (only if present) ──────────────────────────────
    words: Optional[list[WordTimestamp]] = None
    if raw.words:
        words = [
            WordTimestamp(
                start=round(w.start, 3),
                end=round(w.end,   3),
                word=w.word,
                probability=round(float(w.probability), 4),
            )
            for w in raw.words
        ]

    return TranscriptSegment(
        start=round(float(raw.start), 3),
        end=round(float(raw.end),     3),
        text=raw.text.strip(),
        confidence=confidence,
        no_speech_probability=no_speech_prob,
        is_noise=is_noise,
        words=words,
    )


def _run_transcription(
    audio_path:      str,
    language:        Optional[str],
    task:            str,
    word_timestamps: bool,
    beam_size:       int,
    vad_filter:      bool,
    vad_min_silence_ms: int,
    initial_prompt:  Optional[str],
    temperature:     float | tuple[float, ...],
    condition_on_previous_text: bool,
) -> TranscriptionResult:
    """
    Blocking transcription call — runs inside asyncio.to_thread().

    Args:
        audio_path:               Validated path to audio file.
        language:                 Force a language code or None for auto-detect.
        task:                     'transcribe' or 'translate' (→ English).
        word_timestamps:          Enable per-word timing in segments.
        beam_size:                Beam width — higher = more accurate, slower.
        vad_filter:               Skip silent segments via VAD.
        vad_min_silence_ms:       Minimum silence duration (ms) for VAD.
        initial_prompt:           Optional text to prime the decoder context.
        temperature:              Sampling temperature or fallback schedule.
        condition_on_previous_text: Use previous segment output as context.

    Returns:
        TranscriptionResult with all segments and metadata.
    """
    audio_path_obj = _validate_audio_file(audio_path)
    model          = get_model()

    logger.info(
        "▶  Transcribing: %s  [lang=%s | task=%s | beam=%d | vad=%s | words=%s]",
        audio_path_obj.name,
        language or "auto",
        task,
        beam_size,
        vad_filter,
        word_timestamps,
    )

    t_start = time.perf_counter()

    # ── Run faster-whisper ───────────────────────────────────────────────────
    raw_segments, info = model.transcribe(
        str(audio_path_obj),
        language=language,
        task=task,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
        vad_parameters={
            "min_silence_duration_ms": vad_min_silence_ms,
            "speech_pad_ms": 200,        # pad speech regions for better coverage
        },
        initial_prompt=initial_prompt,
        temperature=temperature,
        condition_on_previous_text=condition_on_previous_text,
        log_progress=False,
    )

    # ── Consume the generator (must happen before measuring time) ────────────
    segments: list[TranscriptSegment] = []
    noise_count  = 0
    confidences  = []

    for raw in raw_segments:
        seg = _build_segment(raw)
        segments.append(seg)

        if seg.is_noise:
            noise_count += 1
        if seg.confidence is not None:
            confidences.append(seg.confidence)

    t_elapsed = round(time.perf_counter() - t_start, 3)

    # ── Aggregate metrics ────────────────────────────────────────────────────
    full_text   = " ".join(seg.text for seg in segments if not seg.is_noise).strip()
    word_count  = len(full_text.split()) if full_text else 0
    avg_conf    = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    logger.info(
        "✓  Transcription complete: %d segments | %d words | lang=%s (%.0f%%) | "
        "noise=%d | avg_conf=%.2f | time=%.2fs",
        len(segments),
        word_count,
        info.language,
        info.language_probability * 100,
        noise_count,
        avg_conf,
        t_elapsed,
    )

    return TranscriptionResult(
        # Primary
        segments=segments,
        full_text=full_text,
        # Language
        language=info.language,
        language_probability=round(float(info.language_probability), 4),
        # Audio
        audio_path=str(audio_path_obj),
        duration_seconds=round(float(info.duration), 3),
        # Statistics
        word_count=word_count,
        segment_count=len(segments),
        noise_segment_count=noise_count,
        avg_confidence=avg_conf,
        transcription_time_seconds=t_elapsed,
        # Model
        model_size=_MODEL_SIZE,
        device=_DEVICE,
        compute_type=_COMPUTE_TYPE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output formatters
# ─────────────────────────────────────────────────────────────────────────────


def segments_to_json(result: TranscriptionResult) -> list[dict]:
    """
    Convert a TranscriptionResult to the minimal requested output format:

        [
          {"start": 0.0, "end": 4.2, "text": "..."},
          ...
        ]

    Noise-flagged segments are excluded by default.

    Args:
        result: TranscriptionResult from transcribe().

    Returns:
        List of dicts with 'start', 'end', 'text' keys.
    """
    return [
        {
            "start": seg.start,
            "end":   seg.end,
            "text":  seg.text,
        }
        for seg in result.segments
        if not seg.is_noise and seg.text
    ]


def segments_to_srt(result: TranscriptionResult) -> str:
    """
    Export transcript as SRT subtitle format.

    Args:
        result: TranscriptionResult from transcribe().

    Returns:
        SRT-formatted string, ready to save as a .srt file.
    """
    def _ts(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: list[str] = []
    idx = 1
    for seg in result.segments:
        if seg.is_noise or not seg.text:
            continue
        lines.append(str(idx))
        lines.append(f"{_ts(seg.start)} --> {_ts(seg.end)}")
        lines.append(seg.text)
        lines.append("")
        idx += 1
    return "\n".join(lines)


def segments_to_plain_text(result: TranscriptionResult) -> str:
    """Return the full transcript as plain text (no timestamps)."""
    return result.full_text


# ─────────────────────────────────────────────────────────────────────────────
# Public async interface
# ─────────────────────────────────────────────────────────────────────────────


async def transcribe(
    audio_path: str,
    *,
    language:                   Optional[str]              = None,
    task:                       str                        = "transcribe",
    word_timestamps:            bool                       = False,
    beam_size:                  int                        = 5,
    vad_filter:                 bool                       = True,
    vad_min_silence_ms:         int                        = 500,
    initial_prompt:             Optional[str]              = None,
    temperature:                float | tuple[float, ...]  = 0.0,
    condition_on_previous_text: bool                       = True,
) -> TranscriptionResult:
    """
    Async entry point — transcribe an audio file with faster-whisper.

    The blocking Whisper inference runs inside asyncio.to_thread() so the
    FastAPI event loop is never blocked.

    Args:
        audio_path:
            Path to the audio file (absolute or relative).
            Supported formats: mp3, m4a, wav, flac, ogg, opus, webm, mp4, …

        language:
            BCP-47 language code (e.g. 'en', 'hi', 'es').
            Pass None (default) to auto-detect from the first 30 seconds.

        task:
            'transcribe' — return text in the source language.
            'translate'  — translate to English regardless of source language.

        word_timestamps:
            If True, populate TranscriptSegment.words with per-word timing.
            Adds a small overhead per segment.

        beam_size:
            Beam search width. 5 is a good balance of accuracy vs speed.
            Set to 1 for greedy decoding (fastest, lower quality).

        vad_filter:
            Apply Voice Activity Detection to skip silent / noise-only regions.
            Strongly recommended — reduces hallucinations and speeds up processing.

        vad_min_silence_ms:
            Minimum consecutive silence in ms before VAD marks a region as silent.

        initial_prompt:
            Optional text to prime the decoder (e.g. domain-specific vocabulary,
            speaker names). Improves accuracy for niche content.

        temperature:
            Sampling temperature. 0.0 = greedy (deterministic).
            Pass a tuple e.g. (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) for fallback
            schedule (Whisper retries with higher temperature on low confidence).

        condition_on_previous_text:
            Use the previous segment's text as context for the next segment.
            Improves coherence for long recordings.

    Returns:
        TranscriptionResult — use segments_to_json(result) for the minimal
        [{start, end, text}] format as requested.

    Raises:
        FileNotFoundError: Audio file does not exist.
        ValueError:        Unsupported audio format.
        RuntimeError:      Whisper model failed to transcribe.

    Example:
        result = await transcribe("data/raw/instagram/ABC123.m4a")
        timestamps = segments_to_json(result)
        # → [{"start": 0.0, "end": 4.2, "text": "Hello world"}, ...]

        # With word-level detail:
        result = await transcribe(path, word_timestamps=True, language="en")

        # Translate to English from any language:
        result = await transcribe(path, task="translate")
    """
    logger.info("transcribe() called | path=%s | lang=%s | task=%s", audio_path, language, task)

    result = await asyncio.to_thread(
        _run_transcription,
        audio_path,
        language,
        task,
        word_timestamps,
        beam_size,
        vad_filter,
        vad_min_silence_ms,
        initial_prompt,
        temperature,
        condition_on_previous_text,
    )

    return result
