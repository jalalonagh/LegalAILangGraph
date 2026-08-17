"""
Audit logging service.

Records every significant AI operation to the database. Does not store
raw sensitive content unless explicitly provided — stores hashes by default.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger
from app.domain.interfaces import AuditRepository
from app.domain.enums import AuditAction

logger = get_logger()


def _hash_content(content: str, algorithm: str = "sha256") -> str:
    """Produce a deterministic hash of content for audit logging."""
    if not content:
        return ""
    h = hashlib.new(algorithm)
    h.update(content.encode("utf-8"))
    return h.hexdigest()


class AuditService:
    """Service for recording audit events."""

    def __init__(self, repository: AuditRepository) -> None:
        self._repo = repository

    async def record_request(
        self,
        tenant_id: str,
        user_id: str,
        run_id: str,
        workflow: str,
        input_text: str,
        **extra: Any,
    ) -> None:
        await self._repo.record_event(
            event_type=AuditAction.REQUEST.value,
            tenant_id=tenant_id,
            user_id=user_id,
            run_id=run_id,
            workflow=workflow,
            input_hash=_hash_content(input_text),
            details=extra,
        )

    async def record_tool_call(
        self,
        tenant_id: str,
        user_id: str,
        run_id: str,
        workflow: str,
        agent: str,
        tool_name: str,
        tool_arguments: dict[str, Any],
        source_ids: list[str],
        **extra: Any,
    ) -> None:
        await self._repo.record_event(
            event_type=AuditAction.TOOL_CALL.value,
            tenant_id=tenant_id,
            user_id=user_id,
            run_id=run_id,
            workflow=workflow,
            agent=agent,
            tool=tool_name,
            tool_arguments_hash=_hash_content(json.dumps(tool_arguments, default=str)),
            source_ids=source_ids,
            details=extra,
        )

    async def record_llm_call(
        self,
        tenant_id: str,
        user_id: str,
        run_id: str,
        workflow: str,
        agent: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        input_hash: str,
        output_hash: str,
        citations: list[dict[str, Any]],
        confidence: float | None = None,
        **extra: Any,
    ) -> None:
        await self._repo.record_event(
            event_type=AuditAction.LLM_CALL.value,
            tenant_id=tenant_id,
            user_id=user_id,
            run_id=run_id,
            workflow=workflow,
            agent=agent,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_hash=input_hash,
            output_hash=output_hash,
            citations=citations,
            confidence=confidence,
            details=extra,
        )

    async def record_verification(
        self,
        tenant_id: str,
        user_id: str,
        run_id: str,
        workflow: str,
        verified: bool,
        confidence: float,
        issues: list[str],
        **extra: Any,
    ) -> None:
        await self._repo.record_event(
            event_type=AuditAction.VERIFY.value,
            tenant_id=tenant_id,
            user_id=user_id,
            run_id=run_id,
            workflow=workflow,
            confidence=confidence,
            result="verified" if verified else "failed",
            error="; ".join(issues) if issues else None,
            details=extra,
        )

    async def record_review(
        self,
        tenant_id: str,
        user_id: str,
        run_id: str,
        workflow: str,
        decision: str,
        reviewer_id: str,
        notes: str | None = None,
    ) -> None:
        from app.domain.enums import ReviewDecision

        audit_event_type = {
            ReviewDecision.APPROVE: AuditAction.REVIEW_APPROVE,
            ReviewDecision.REJECT: AuditAction.REVIEW_REJECT,
        }.get(decision, AuditAction.REVIEW_APPROVE)

        await self._repo.record_event(
            event_type=audit_event_type.value,
            tenant_id=tenant_id,
            user_id=user_id,
            run_id=run_id,
            workflow=workflow,
            result=decision,
            reviewer_id=reviewer_id,
            tool=notes,
        )

    async def record_error(
        self,
        tenant_id: str,
        user_id: str,
        run_id: str,
        workflow: str,
        node_id: str,
        error_code: str,
        error_message: str,
        **extra: Any,
    ) -> None:
        await self._repo.record_event(
            event_type="error",
            tenant_id=tenant_id,
            user_id=user_id,
            run_id=run_id,
            workflow=workflow,
            node_id=node_id,
            error=error_message,
            details={"error_code": error_code, **extra},
        )

    async def record_config_change(
        self,
        tenant_id: str,
        user_id: str,
        entity_type: str,
        entity_id: str,
        action: str,
        **extra: Any,
    ) -> None:
        await self._repo.record_event(
            event_type=AuditAction.CONFIG_CHANGE.value,
            tenant_id=tenant_id,
            user_id=user_id,
            run_id="",
            workflow=entity_type,
            result=action,
            details={"entity_id": entity_id, **extra},
        )

    async def query(
        self,
        tenant_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_type: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        return await self._repo.query_events(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            limit=limit,
        )


__all__ = ["AuditService", "_hash_content"]
