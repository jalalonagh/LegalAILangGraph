"""
Structured JSON logging with request context correlation.

Uses ``structlog`` for structured output and standard ``logging`` for
the OpenTelemetry bridge. Each log entry includes request_id, tenant_id,
user_id, and run_id when available.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog
from structlog.stdlib import ProcessorFormatter

from app.core.config import settings

# --- Context variables for request-scoped correlation IDs ---
_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
_tenant_id_ctx: ContextVar[str | None] = ContextVar("tenant_id", default=None)
_user_id_ctx: ContextVar[str | None] = ContextVar("user_id", default=None)
_run_id_ctx: ContextVar[str | None] = ContextVar("run_id", default=None)


def get_request_id() -> str | None:
    return _request_id_ctx.get()


def set_request_id(rid: str | None) -> None:
    _request_id_ctx.set(rid)


def get_tenant_id() -> str | None:
    return _tenant_id_ctx.get()


def set_tenant_id(tid: str | None) -> None:
    _tenant_id_ctx.set(tid)


def get_user_id() -> str | None:
    return _user_id_ctx.get()


def set_user_id(uid: str | None) -> None:
    _user_id_ctx.set(uid)


def get_run_id() -> str | None:
    return _run_id_ctx.get()


def set_run_id(rid: str | None) -> None:
    _run_id_ctx.set(rid)


def get_correlation_context() -> dict[str, str | None]:
    return {
        "request_id": get_request_id(),
        "tenant_id": get_tenant_id(),
        "user_id": get_user_id(),
        "run_id": get_run_id(),
    }


def _add_context_vars(logger: logging.Logger, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Inject request-scoped context variables into every log record."""
    for key in ("request_id", "tenant_id", "user_id", "run_id"):
        val = event_dict.get(key)
        if val is None:
            event_dict[key] = _get_ctx_var(key)
    return event_dict


_CTX_VAR_MAP = {
    "request_id": _request_id_ctx,
    "tenant_id": _tenant_id_ctx,
    "user_id": _user_id_ctx,
    "run_id": _run_id_ctx,
}


def _get_ctx_var(key: str) -> str | None:
    var = _CTX_VAR_MAP.get(key)
    if var is None:
        return None
    return var.get()


def _merge_context(event_dict: dict[str, Any]) -> dict[str, Any]:
    """Merge contextvars into the event dict, without overwriting explicit values."""
    for key in ("request_id", "tenant_id", "user_id", "run_id"):
        if key not in event_dict:
            event_dict[key] = _get_ctx_var(key)
    return event_dict


def configure_logging() -> structlog.BoundLogger:
    """Configure structured logging based on the environment."""
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _merge_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.LOG_FORMAT == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stdout,
    )

    # Configure stdlib loggers used by dependencies
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.addHandler(logging.NullHandler())
        lg.propagate = False

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first=True,
        wrapper_class_lookup=True,
    )

    return structlog.get_logger()


# Initialize on import
logger = configure_logging()
