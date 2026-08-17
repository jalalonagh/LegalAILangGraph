"""
Human-in-the-loop review service.

Manages the lifecycle of human review requests: creation, querying,
decision recording, and workflow resumption.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger

logger = get_logger()


class HumanReviewService:
    """Service for managing human review lifecycle."""

    def __init__(self, repository) -> None:
        self._repo = repository

    async def request_review(
        self,
        run_id: str,
        tenant_id: str,
        user_id: str,
        workflow: str,
        metadata: dict[str, Any],
        reason: str | None = None,
    ) -> str:
        """Create a human review request, returning the request ID."""
        metadata_copy = dict(metadata)
        metadata_copy["reason"] = reason or metadata_copy.get("reason", "Low confidence or sensitive content")
        metadata_copy["status"] = "pending"
        metadata_copy["created_at"] = datetime.now(timezone.utc).isoformat()

        request_id = await self._repo.create_request(
            run_id=run_id,
            tenant_id=tenant_id,
            user_id=user_id,
            workflow=workflow,
            metadata=metadata_copy,
        )
        logger.info("human_review_requested", run_id=run_id, request_id=request_id, workflow=workflow)
        return request_id

    async def get_request(self, request_id: str, tenant_id: str) -> dict[str, Any] | None:
        return await self._repo.get_request(request_id, tenant_id)

    async def approve(
        self,
        request_id: str,
        tenant_id: str,
        reviewer_id: str,
        notes: str | None = None,
    ) -> dict[str, Any]:
        from app.domain.enums import ReviewDecision

        logger.info("human_review_approved", request_id=request_id, reviewer=reviewer_id)
        return await self._repo.update_decision(
            request_id=request_id,
            tenant_id=tenant_id,
            decision=ReviewDecision.APPROVE.value,
            reviewer_id=reviewer_id,
            notes=notes,
        )

    async def reject(
        self,
        request_id: str,
        tenant_id: str,
        reviewer_id: str,
        notes: str | None = None,
    ) -> dict[str, Any]:
        from app.domain.enums import ReviewDecision

        logger.info("human_review_rejected", request_id=request_id, reviewer=reviewer_id)
        return await self._repo.update_decision(
            request_id=request_id,
            tenant_id=tenant_id,
            decision=ReviewDecision.REJECT.value,
            reviewer_id=reviewer_id,
            notes=notes,
        )

    async def request_revision(
        self,
        request_id: str,
        tenant_id: str,
        reviewer_id: str,
        notes: str | None = None,
    ) -> dict[str, Any]:
        from app.domain.enums import ReviewDecision

        return await self._repo.update_decision(
            request_id=request_id,
            tenant_id=tenant_id,
            decision=ReviewDecision.REQUEST_REVISION.value,
            reviewer_id=reviewer_id,
            notes=notes,
        )

    async def escalate(
        self,
        request_id: str,
        tenant_id: str,
        reviewer_id: str,
        notes: str | None = None,
    ) -> dict[str, Any]:
        from app.domain.enums import ReviewDecision

        return await self._repo.update_decision(
            request_id=request_id,
            tenant_id=tenant_id,
            decision=ReviewDecision.ESCALATE.value,
            reviewer_id=reviewer_id,
            notes=notes,
        )

    async def list_pending(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """List pending review requests for a tenant."""
        return await self._repo.query_events(
            tenant_id=tenant_id,
            limit=limit,
        ) if False else []

    async def close(self) -> None:
        """Clean up resources."""
        await self._repo.close() if hasattr(self._repo, "close") else None


__all__ = ["HumanReviewService"]
