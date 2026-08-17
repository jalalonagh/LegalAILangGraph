"""
FastAPI dependencies for authentication, database, and service resolution.
"""

from __future__ import annotations

from typing import Annotated, AsyncGenerator

from fastapi import Depends

from app.core.security import AuthenticatedServiceContext, get_auth_context_async
from app.infrastructure.database.session import get_db_session


# --- Authentication ---
def require_auth() -> Annotated[AuthenticatedServiceContext, Depends(get_auth_context_async)]:
    """Dependency that requires authentication. Raises 401 if missing/invalid."""
    return Depends(get_auth_context_async)


def require_admin() -> Annotated[AuthenticatedServiceContext, Depends]:
    """Dependency that requires an admin role."""
    async def _admin(
        ctx: AuthenticatedServiceContext = Depends(get_auth_context_async),
    ) -> AuthenticatedServiceContext:
        if ctx.user_role != "admin" and "*" not in ctx.permissions:
            from app.core.exceptions import AuthorizationError

            raise AuthorizationError("Admin privileges required")
        return ctx

    return Depends(_admin)


def require_permission(permission: str):
    """Returns a dependency that requires a specific permission."""
    async def _perm(
        ctx: AuthenticatedServiceContext = Depends(get_auth_context_async),
    ) -> AuthenticatedServiceContext:
        ctx.check_permission(permission)
        return ctx

    return Depends(_perm)


# --- Database ---
DbSession = Annotated[AsyncGenerator, Depends(get_db_session)]


# --- Service container (singletons) resolved per request ---
from app.services.service_container import get_service, ServiceContainer  # noqa: E402, F401

__all__ = [
    "require_auth",
    "require_admin",
    "require_permission",
    "DbSession",
    "get_service",
    "ServiceContainer",
]
