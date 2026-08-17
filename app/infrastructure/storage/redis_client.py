"""
Redis client for caching, rate limiting, and distributed locks.
"""

from __future__ import annotations

import json
import pickle
from typing import Any

import redis.asyncio as redis
from redis.asyncio import Redis

from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import VectorStoreError

logger = get_logger()

_redis: Redis | None = None


def get_redis_client() -> Redis:
    """Return a singleton Redis client."""
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
        logger.info("redis_connected", url=settings.REDIS_URL)
    return _redis


async def get_redis() -> Redis:
    """Async context-friendly accessor."""
    return get_redis_client()


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.close()
        _redis = None
        logger.info("redis_disconnected")


# ------------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------------
CACHE_VERSION = "v1"


def _cache_key(key: str, tenant_id: str) -> str:
    return f"legalai:{CACHE_VERSION}:cache:{tenant_id}:{key}"


async def cache_get(key: str, tenant_id: str = "demo") -> Any | None:
    """Get a value from cache. Returns deserialized object or None."""
    if not settings.CACHE_ENABLED:
        return None
    client = get_redis_client()
    raw = await client.get(_cache_key(key, tenant_id))
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        try:
            return pickle.loads(raw.encode())
        except Exception:  # noqa: BLE001
            return None


async def cache_set(
    key: str,
    value: Any,
    tenant_id: str = "demo",
    ttl: int | None = None,
) -> None:
    """Set a value in cache with optional TTL."""
    if not settings.CACHE_ENABLED:
        return
    client = get_redis_client()
    ttl = ttl or settings.REDIS_CACHE_TTL_SECONDS
    try:
        raw = json.dumps(value, default=str)
    except (TypeError, ValueError):
        raw = pickle.dumps(value)
    await client.setex(_cache_key(key, tenant_id), ttl, raw)


async def cache_delete(key: str, tenant_id: str = "demo") -> None:
    """Delete a key from cache."""
    if not settings.CACHE_ENABLED:
        return
    client = get_redis_client()
    await client.delete(_cache_key(key, tenant_id))


# ------------------------------------------------------------------
# Distributed lock
# ------------------------------------------------------------------
class RedisLock:
    """Context manager for a distributed lock using Redis."""

    def __init__(self, name: str, timeout: float = 30.0, blocking_timeout: float = 10.0):
        self._name = f"legalai:lock:{name}"
        self._timeout = timeout
        self._blocking_timeout = blocking_timeout
        self._lock = None

    async def __aenter__(self):
        client = get_redis_client()
        self._lock = client.lock(self._name, timeout=self._timeout, blocking_timeout=self._blocking_timeout)
        await self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._lock is not None:
            await self._lock.release()


__all__ = [
    "get_redis_client",
    "get_redis",
    "close_redis",
    "cache_get",
    "cache_set",
    "cache_delete",
    "RedisLock",
]
