"""
VideoRAG — FastAPI application entry point.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

API docs available at:
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chat, health, analyze
from app.core.config import get_settings
from app.services.transcription_service import preload_model as preload_whisper
from app.utils.logger import get_logger
from app.vectorstore.client import get_collection

logger = get_logger(__name__)
_settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle handler.

    Startup:  Warms up the ChromaDB connection so the first request
              doesn't pay the cold-start penalty.
    Shutdown: Logs graceful shutdown.
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Starting %s  v%s …", _settings.APP_NAME, _settings.APP_VERSION)
    logger.info("Debug mode: %s", _settings.DEBUG)

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    try:
        col = get_collection()
        logger.info(
            "ChromaDB ready — collection '%s' (%d docs indexed)",
            _settings.CHROMA_COLLECTION_NAME,
            col.count(),
        )
    except Exception as exc:
        logger.error("ChromaDB warm-up failed: %s", exc)
        logger.warning("Continuing startup — ChromaDB will be retried on first request.")

    # ── faster-whisper — preload model to avoid first-request cold start ──────
    # Disabled for Render Free Tier to prevent Out of Memory (OOM) crash on startup
    # try:
    #     await preload_whisper()
    # except Exception as exc:
    #     logger.error("Whisper model preload failed: %s", exc)
    logger.warning("Whisper will be loaded on first transcription request.")

    yield  # ← application runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down %s. Goodbye.", _settings.APP_NAME)


# ── App instance ──────────────────────────────────────────────────────────────


app = FastAPI(
    title=_settings.APP_NAME,
    version=_settings.APP_VERSION,
    description=(
        "**VideoRAG API** — A Retrieval-Augmented Generation backend that:\n\n"
        "- Ingests a **YouTube video** and an **Instagram Reel**\n"
        "- Extracts metadata and transcripts from both sources\n"
        "- Chunks and embeds transcripts using **OpenAI text-embedding-3-small**\n"
        "- Stores embeddings in **ChromaDB** (persistent on-disk)\n"
        "- Answers natural-language questions via **GPT-4o-mini** with source citations\n"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── Middleware ────────────────────────────────────────────────────────────────


app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────────────


_API_PREFIX = "/api/v1"

app.include_router(health.router, prefix=_API_PREFIX)
app.include_router(analyze.router, prefix=_API_PREFIX)
app.include_router(chat.router, prefix=_API_PREFIX)


# ── Root ──────────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """Root redirect — useful for quickly confirming the server is alive."""
    return JSONResponse(
        content={
            "name": _settings.APP_NAME,
            "version": _settings.APP_VERSION,
            "docs": "/docs",
            "health": "/api/v1/health",
        }
    )
