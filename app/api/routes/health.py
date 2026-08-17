"""
Health check endpoints.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app.api.schemas import HealthResponse, DependencyHealth
from app.core.config import settings
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Basic health check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version=settings.VERSION,
    )


@router.get("/health/live", response_model=HealthResponse)
async def health_live() -> HealthResponse:
    """Liveness probe — returns 200 if the process is running."""
    return HealthResponse(
        status="alive",
        timestamp=datetime.now(timezone.utc),
        version=settings.VERSION,
    )


@router.get("/health/ready", response_model=HealthResponse)
async def health_ready(request: Request) -> HealthResponse:
    """Readiness probe — checks configurable dependencies."""
    checks: dict[str, str] = {}
    all_healthy = True

    # Database check
    try:
        from app.infrastructure.database.session import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["database"] = f"error: {exc}"
        all_healthy = False

    # Redis check
    try:
        from app.infrastructure.storage.redis_client import get_redis_client

        redis = get_redis_client()
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        if settings.REDIS_URL:
            checks["redis"] = f"error: {exc}"
            all_healthy = False
        else:
            checks["redis"] = "disabled"

    # Vector DB check
    try:
        from app.services.service_container import get_service

        container = get_service()
        vs = container.vector_store
        collections = await vs.get_sync_client().get_collections()
        checks["vector_db"] = "ok"
    except Exception as exc:  # noqa: BLE001
        if settings.VECTOR_DB_ENABLED:
            checks["vector_db"] = f"error: {exc}"
            all_healthy = False
        else:
            checks["vector_db"] = "disabled"

    # LLM provider check
    try:
        from app.services.service_container import get_service

        container = get_service()
        if settings.DEFAULT_LLM_PROVIDER == "ollama":
            import httpx

            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                if resp.status_code == 200:
                    checks["llm"] = "ok"
                else:
                    checks["llm"] = f"status: {resp.status_code}"
        else:
            checks["llm"] = "configured"
    except Exception as exc:  # noqa: BLE001
        checks["llm"] = f"error: {exc}"
        if settings.DEFAULT_LLM_PROVIDER != "ollama":
            all_healthy = False

    status = "ready" if all_healthy else "degraded"
    return HealthResponse(
        status=status,
        timestamp=datetime.now(timezone.utc),
        version=settings.VERSION,
        checks=checks,
    )
