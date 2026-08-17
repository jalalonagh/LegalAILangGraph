"""
Security module: JWT validation, API key authentication, tenant resolution.

This module provides the trust boundary between the ASP.NET Core gateway
and the LangGraph service. It validates JWTs issued by the upstream
identity provider and resolves tenant/user context from verified claims.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from secrets import compare_digest
from typing import Any, Protocol

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext

from app.core.config import settings
from app.core.exceptions import AuthenticationError, AuthorizationError, ConfigError, TenantIsolationError
from app.core.logging import get_logger

logger = get_logger()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


class ServiceContext(Protocol):
    """Authenticated context for an internal service or user request."""

    tenant_id: str
    user_id: str
    user_role: str
    permissions: list[str]
    is_internal_service: bool


class AuthenticatedServiceContext:
    """Concrete authenticated context resolved from a verified token."""

    def __init__(
        self,
        tenant_id: str,
        user_id: str,
        user_role: str,
        permissions: list[str],
        is_internal_service: bool = False,
    ) -> None:
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.user_role = user_role
        self.permissions = permissions
        self.is_internal_service = is_internal_service

    def has_permission(self, permission: str) -> bool:
        """Check if the context has a specific permission."""
        if "*" in self.permissions or "admin" in self.permissions:
            return True
        return permission in self.permissions

    def check_permission(self, permission: str) -> None:
        if not self.has_permission(permission):
            raise AuthorizationError(
                f"Permission '{permission}' required",
                details={"required_permission": permission, "user_role": self.user_role},
            )

    def check_tenant(self, requested_tenant_id: str) -> None:
        """Ensure the requested tenant matches the authenticated tenant."""
        if self.tenant_id != requested_tenant_id:
            raise TenantIsolationError(
                "Access to tenant data is forbidden",
                details={"requested_tenant": requested_tenant_id, "authenticated_tenant": self.tenant_id},
            )


def _load_public_key() -> str | bytes:
    """Load the JWT public key from file or configuration."""
    if settings.JWT_PUBLIC_KEY_PATH:
        try:
            return Path(settings.JWT_PUBLIC_KEY_PATH).read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("jwt_public_key_load_failed", error=str(exc))
            raise ConfigError("Could not load JWT public key") from exc
    return settings.JWT_SECRET_KEY.get_secret_value()


def verify_jwt_token(token: str) -> dict[str, Any]:
    """Verify a JWT token and return its claims.

    Supports both RSA/EC public-key tokens (via JWT_PUBLIC_KEY_PATH) and
    symmetric HS256 tokens (via JWT_SECRET_KEY).
    """
    try:
        key = _load_public_key()
        payload = jwt.decode(
            token,
            key,
            algorithms=[settings.JWT_ALGORITHM],
            audience=settings.JWT_AUDIENCE,
            issuer=settings.JWT_ISSUER,
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        logger.warning("jwt_expired", error=str(exc))
        raise AuthenticationError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        logger.warning("jwt_invalid", error=str(exc))
        raise AuthenticationError(f"Invalid token: {exc}") from exc
    return payload


def verify_internal_api_key(key: str) -> bool:
    """Verify an internal service-to-service API key using constant-time comparison."""
    expected = settings.INTERNAL_API_KEY.get_secret_value()
    return compare_digest(key, expected)


def extract_claims(payload: dict[str, Any], is_internal: bool = False) -> AuthenticatedServiceContext:
    """Extract tenant/user/permission context from validated JWT claims."""
    tenant_id = payload.get("tenant_id") or payload.get("tid")
    user_id = payload.get("sub") or payload.get("user_id") or payload.get("uid")
    raw_roles = payload.get("role") or payload.get("roles")
    if isinstance(raw_roles, list):
        user_role = raw_roles[0] if raw_roles else "user"
    else:
        user_role = raw_roles or "user"
    permissions = payload.get("permissions", [])

    if not tenant_id:
        if is_internal:
            tenant_id = "internal"
        else:
            raise AuthenticationError("Token missing tenant_id claim")

    if not user_id:
        user_id = "internal_service" if is_internal else "anonymous"

    return AuthenticatedServiceContext(
        tenant_id=tenant_id,
        user_id=user_id,
        user_role=user_role,
        permissions=permissions if isinstance(permissions, list) else [],
        is_internal_service=is_internal,
    )


async def resolve_context(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = None,
    api_key: str | None = None,
) -> AuthenticatedServiceContext:
    """Resolve authentication context from JWT bearer token or internal API key.

    Priority:
    1. JWT bearer token (from Authorization header).
    2. Internal API key (from X-API-Key header) — for service-to-service calls.
    """
    if credentials is None:
        if api_key and verify_internal_api_key(api_key):
            return AuthenticatedServiceContext(
                tenant_id="internal",
                user_id="internal_service",
                user_role="admin",
                permissions=["admin", "*"],
                is_internal_service=True,
            )
        raise AuthenticationError("Missing or invalid authentication credentials")

    token = credentials.credentials
    payload = verify_jwt_token(token)
    aud_list = payload.get("aud")
    if isinstance(aud_list, list):
        is_internal = "internal" in aud_list
    elif isinstance(aud_list, str):
        is_internal = aud_list == "internal"
    else:
        is_internal = False
    return extract_claims(payload, is_internal=is_internal)


def get_auth_context(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthenticatedServiceContext:
    """Sync FastAPI dependency that returns an authenticated service context."""
    api_key = request.headers.get("X-API-Key")
    ctx = asyncio.run(resolve_context(request, credentials, api_key))
    _set_context_vars(ctx)
    return ctx


async def get_auth_context_async(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthenticatedServiceContext:
    """Async FastAPI dependency that returns an authenticated service context."""
    api_key = request.headers.get("X-API-Key")
    ctx = await resolve_context(request, credentials, api_key)
    _set_context_vars(ctx)
    return ctx


def _set_context_vars(ctx: AuthenticatedServiceContext) -> None:
    from app.core.logging import set_tenant_id, set_user_id

    set_tenant_id(ctx.tenant_id)
    set_user_id(ctx.user_id)


def create_dev_token(
    tenant_id: str = "demo",
    user_id: str = "dev-user",
    role: str = "admin",
    permissions: list[str] | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a self-signed JWT token for local development and testing.

    Only used in non-production environments. Never use in production.
    """
    if settings.is_production:
        raise ConfigError("Token generation is disabled in production")
    if expires_delta is None:
        expires_delta = timedelta(hours=8)
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "tid": tenant_id,
        "role": role,
        "permissions": permissions or ["*"],
        "aud": settings.JWT_AUDIENCE,
        "iss": settings.JWT_ISSUER,
        "iat": now,
        "exp": now + expires_delta,
    }
    secret = settings.JWT_SECRET_KEY.get_secret_value()
    return jwt.encode(payload, secret, algorithm=settings.JWT_ALGORITHM)


__all__ = [
    "ServiceContext",
    "AuthenticatedServiceContext",
    "verify_jwt_token",
    "verify_internal_api_key",
    "extract_claims",
    "resolve_context",
    "get_auth_context",
    "get_auth_context_async",
    "create_dev_token",
]
