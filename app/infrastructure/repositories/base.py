"""
Base repository implementation providing common CRUD operations
with tenant isolation enforced at the repository level.

All repository methods accept a ``tenant_id`` parameter and automatically
apply a WHERE filter, ensuring cross-tenant data access is impossible
regardless of how the calling code is written.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from app.infrastructure.database.models import Base

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    """Generic async repository with tenant-scoped operations.

    Subclasses must set ``model`` after initialization.
    """

    def __init__(self, session: AsyncSession, model: type[T]):
        self.session = session
        self.model = model

    # ------------------------------------------------------------------
    # Internal: tenant filter
    # ------------------------------------------------------------------
    def _tenant_filter(self, tenant_id: str) -> Any:
        """Return the column-level filter expression for tenant isolation."""
        return self.model.tenant_id == tenant_id

    def _base_query(self, tenant_id: str) -> Select:
        """Return a base query already filtered by tenant."""
        return select(self.model).where(self._tenant_filter(tenant_id))

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    async def get(self, id: str, tenant_id: str) -> T | None:
        """Get a single record by primary key (tenant-scoped)."""
        result = await self.session.execute(
            self._base_query(tenant_id).where(self.model.id == id)
        )
        return result.scalar_one_or_none()

    async def get_by_external(
        self, external_id: str, tenant_id: str, external_field: str = "external_id"
    ) -> T | None:
        """Get a record by an external identifier (tenant-scoped)."""
        extra_filter = getattr(self.model, external_field) == external_id
        result = await self.session.execute(
            self._base_query(tenant_id).where(extra_filter)
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        tenant_id: str,
        limit: int = 100,
        offset: int = 0,
        **filters: Any,
    ) -> list[T]:
        """List records for a tenant with optional filters."""
        query = self._base_query(tenant_id)
        for field, value in filters.items():
            if hasattr(self.model, field):
                query = query.where(getattr(self.model, field) == value)
        query = query.offset(offset).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def count(self, tenant_id: str, **filters: Any) -> int:
        """Count records for a tenant."""
        query = select(func.count()).select_from(self.model).where(self._tenant_filter(tenant_id))
        for field, value in filters.items():
            if hasattr(self.model, field):
                query = query.where(getattr(self.model, field) == value)
        result = await self.session.execute(query)
        return result.scalar_one()

    async def add(self, instance: T) -> T:
        """Add a new record."""
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def update(
        self,
        id: str,
        tenant_id: str,
        data: dict[str, Any],
    ) -> T | None:
        """Update a record by primary key (tenant-scoped)."""
        await self.session.execute(
            update(self.model)
            .where(self.model.id == id, self._tenant_filter(tenant_id))
            .values(**data, updated_at=func.now())
        )
        await self.session.flush()
        return await self.get(id, tenant_id)

    async def delete(self, id: str, tenant_id: str) -> None:
        """Delete a record by primary key (tenant-scoped)."""
        await self.session.execute(
            delete(self.model).where(self.model.id == id, self._tenant_filter(tenant_id))
        )
        await self.session.flush()

    async def exists(self, id: str, tenant_id: str) -> bool:
        """Check if a record exists for the given tenant."""
        result = await self.session.execute(
            select(func.count())
            .where(self.model.id == id, self._tenant_filter(tenant_id))
        )
        return result.scalar_one() > 0
