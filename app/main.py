"""
Main FastAPI application entry point for the Legal AI Assistant.

This is an independent Python service that runs alongside (or behind)
an ASP.NET Core + Blazor frontend. It exposes REST APIs and manages
LangGraph workflows for legal AI operations.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import api_router
from app.core.config import settings
from app.core.exceptions import LegalAIError
from app.core.logging import get_logger
from app.core.telemetry import init_prometheus, init_tracing, instrument_app
from app.api.middleware import register_middlewares

logger = get_logger()


def create_exception_handlers(app: FastAPI) -> None:
    """Register structured exception handlers."""

    @app.exception_handler(LegalAIError)
    async def legal_ai_error_handler(request: Request, exc: LegalAIError):
        logger.warning(
            "request_error",
            code=exc.code,
            message=exc.message,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
            headers={"X-Request-ID": request.headers.get("X-Request-ID", "")} if False else {},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred",
                    "request_id": request.headers.get("X-Request-ID", "unknown"),
                }
            },
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle: startup and shutdown."""
    startup_start = time.perf_counter()
    logger.info("application_starting", version=settings.VERSION, env=settings.APP_ENV)

    # Initialize tracing (if enabled)
    init_tracing()

    # Initialize Prometheus metrics
    init_prometheus(app)

    # Log startup
    duration_ms = round((time.perf_counter() - startup_start) * 1000, 2)
    logger.info("application_started", duration_ms=duration_ms)

    yield

    # Shutdown
    logger.info("application_shutdown", shutting_down=True)
    try:
        from app.infrastructure.database.session import close_engine

        await close_engine()
    except Exception as exc:  # noqa: BLE001
        logger.warning("database_close_failed", error=str(exc))

    try:
        from app.infrastructure.llm.factory import LLMProviderFactoryImpl

        factory = LLMProviderFactoryImpl()
        await factory.close_all() if hasattr(factory, "close_all") else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("llm_factory_close_failed", error=str(exc))

    logger.info("application_stopped")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="Legal AI Assistant API — powered by LangGraph",
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "Legal AI Team",
            "url": "https://github.com/legal-ai/legal-ai",
            "email": "ai-team@legal-ai.internal",
        },
        license_info={
            "name": "Apache-2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0",
        },
        lifespan=lifespan,
    )

    # Middleware
    register_middlewares(app)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.APP_DEBUG else ["https://" + settings.JWT_AUDIENCE],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["*"],
        max_age=600,
    )

    # Exception handlers
    create_exception_handlers(app)

    # Instrument for tracing
    instrument_app(app)

    # Routes
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # Root redirect
    @app.get("/", include_in_schema=False)
    async def _root():
        return PlainTextResponse(
            f"Legal AI Assistant v{settings.VERSION}\n"
            f"Docs: /docs\n"
            f"Health: /health\n"
        )

    return app


# Create the application instance
app = create_app()
