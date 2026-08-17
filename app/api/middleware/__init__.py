"""
API middleware for logging, request context, security headers, and rate limiting.
"""

from __future__ import annotations

import time
import uuid
from typing import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import settings
from app.core.exceptions import LegalAIError
from app.core.logging import get_logger, set_request_id, set_run_id, set_tenant_id, set_user_id

logger = get_logger()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Sets request-scoped context variables and structured logging fields."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        set_request_id(rid)
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
            response = JSONResponse(
                status_code=500,
                content={"error": {"code": "INTERNAL_ERROR", "message": "Internal server error", "request_id": rid}},
                headers={"X-Request-ID": rid},
            )
            return response
        finally:
            set_request_id(None)
            set_tenant_id(None)
            set_user_id(None)
            set_run_id(None)
        response.headers["X-Request-ID"] = rid
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logs every incoming request and its outcome."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        start = time.perf_counter()
        method = request.method
        path = request.url.path
        logger.info("request_started", method=method, path=path, request_id=request.headers.get("X-Request-ID"))

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        from app.core.telemetry import request_counter

        request_counter.labels(method=method, endpoint=path, status=response.status_code).inc()
        logger.info(
            "request_completed",
            method=method,
            path=path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        response.headers["X-Response-Time-ms"] = str(duration_ms)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds security-related HTTP headers."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'"
        )
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return response


class SizeLimitMiddleware(BaseHTTPMiddleware):
    """Enforces maximum request body size."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and settings.MAX_REQUEST_SIZE_MB > 0:
            max_bytes = settings.MAX_REQUEST_SIZE_MB * 1024 * 1024
            if int(content_length) > max_bytes:
                from app.core.exceptions import FileSizeExceededError

                raise FileSizeExceededError(
                    f"Request body exceeds {settings.MAX_REQUEST_SIZE_MB} MB limit",
                    details={"max_bytes": max_bytes, "actual_bytes": int(content_length)},
                )
        return await call_next(request)


class RequestIDMiddleware:
    """Pure ASGI middleware for early request ID generation and propagation."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers", [])
        rid = None
        for k, v in headers:
            if k.decode().lower() == "x-request-id":
                rid = v.decode()
                break
        if not rid:
            rid = str(uuid.uuid4())
        scope["headers"] = headers + [(b"x-request-id", rid.encode())]
        await self.app(scope, receive, send)


def register_middlewares(app):
    """Register all middleware on the FastAPI app in the correct order."""
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(SizeLimitMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestContextMiddleware)
