"""
Async database session and engine management.

Provides:
- An async SQLAlchemy engine with connection pooling.
- A session factory (async scoped sessions).
- A context manager for dependency injection.
- Base class import for model definitions.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool, Pool

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger()

# Use NullPool for PostgreSQL to avoid connection leaks across async workers
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine() -> AsyncEngine:
    """Create or return the singleton async engine."""
    global _engine
    if _engine is not None:
        return _engine

    db_url = settings.DATABASE_URL
    logger.info("database_connecting", url=db_url.split("@")[0] if "@" in db_url else "hidden")

    if db_url.startswith("sqlite"):
        _engine = create_async_engine(
            db_url,
            echo=settings.APP_DEBUG,
            future=True,
            poolclass=NullPool,
            connect_args={"check_same_thread": False} if db_url.startswith("sqlite+aiosqlite") else {},
        )
    else:
        _engine = create_async_engine(
            db_url,
            echo=settings.APP_DEBUG,
            future=True,
            poolclass=Pool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=300,
        )

    logger.info("database_engine_created")
    return _engine


def get_engine() -> AsyncEngine:
    """Return the singleton engine, creating it if necessary."""
    if _engine is None:
        return _build_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return a session factory bound to the engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


@asynccontextmanager
async def get_db_transaction() -> AsyncGenerator[AsyncSession, None]:
    """Context manager that yields a session within a transaction."""
    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            yield session


async def init_db() -> None:
    """Initialize the database (sync all tables). Use Alembic for production."""
    from app.infrastructure.database.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_tables_created")


async def close_engine() -> None:
    """Dispose of the engine on shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("database_engine_disposed")
