"""
Text cleaning and normalisation utilities used across the pipeline.
"""

from __future__ import annotations

import re


def clean_transcript(text: str) -> str:
    """
    Remove common artefacts found in auto-generated video transcripts:
    - HTML tags
    - Bracketed sound/music annotations  e.g. [Music], [Applause]
    - Inline timestamps                  e.g. 00:01:23 or 1:05
    - Zero-width / invisible characters
    - Redundant whitespace

    Args:
        text: Raw transcript string.

    Returns:
        Cleaned, normalised transcript string.
    """
    # 1. Strip HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # 2. Remove bracketed annotations
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?(?:music|applause|laughter|inaudible).*?\)", "", text, flags=re.IGNORECASE)

    # 3. Remove inline timestamps  00:01:23 | 1:05
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", text)

    # 4. Remove zero-width / invisible Unicode characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # 5. Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text


def truncate_text(text: str, max_chars: int = 500) -> str:
    """
    Truncate text to a maximum character count, breaking at word boundary.

    Args:
        text:      Input string.
        max_chars: Maximum allowed characters (default 500).

    Returns:
        Potentially truncated string with ellipsis appended.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def sanitise_collection_name(name: str) -> str:
    """
    Ensure a ChromaDB collection name contains only alphanumerics and underscores.

    Args:
        name: Raw collection name string.

    Returns:
        Sanitised string safe for use as a Chroma collection name.
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)
