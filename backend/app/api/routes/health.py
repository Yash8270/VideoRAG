"""
Health check endpoint — used by load balancers and monitoring tools.
"""

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.schemas import HealthResponse
from app.vectorstore.client import is_healthy

router = APIRouter(tags=["Health"])
_settings = get_settings()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application health check",
    description="Returns API version and ChromaDB connectivity status.",
)
async def health_check() -> HealthResponse:
    """Ping endpoint confirming the API is alive and ChromaDB is reachable."""
    return HealthResponse(
        status="ok",
        version=_settings.APP_VERSION,
        chroma_ready=is_healthy(),
    )
