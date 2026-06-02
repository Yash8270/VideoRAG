"""
Application configuration — reads from .env file and environment variables.
All settings are validated at startup via Pydantic Settings.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Google Gemini ───────────────────────────────────────────────────────
    GOOGLE_API_KEY: str

    # ── LLM / Embedding Models ───────────────────────────────────────────────
    LLM_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── ChromaDB ────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "data/vectorstore"
    CHROMA_COLLECTION_NAME: str = "rag_videos"

    # ── Text Chunking ────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "VideoRAG API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── CORS ─────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance (loaded once at startup)."""
    return Settings()
