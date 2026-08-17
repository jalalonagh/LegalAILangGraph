"""
Usage tracking service for token accounting and cost control.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger
from app.domain.interfaces import UsageRepository

logger = get_logger()

# Approximate cost per 1K tokens (USD) by model — used for cost estimation
MODEL_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "qwen2.5:7b-instruct-q4_K_M": {"prompt": 0.0, "completion": 0.0},
    "qwen2.5:14b-instruct-q4_K_M": {"prompt": 0.0, "completion": 0.0},
    "qwen2.5:3b-instruct-q4_K_M": {"prompt": 0.0, "completion": 0.0},
}


def estimate_cost(
    model: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """Estimate cost in USD based on token usage and model pricing."""
    cost_info = MODEL_COSTS.get(model)
    if cost_info is None:
        return None
    prompt_cost = (prompt_tokens / 1000) * cost_info["prompt"]
    completion_cost = (completion_tokens / 1000) * cost_info["completion"]
    return round(prompt_cost + completion_cost, 6)


class UsageService:
    """Service for recording and querying usage metrics."""

    def __init__(self, repository: UsageRepository) -> None:
        self._repo = repository

    async def record_usage(
        self,
        tenant_id: str,
        user_id: str,
        run_id: str | None,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        workflow: str | None = None,
    ) -> None:
        cost = estimate_cost(model, provider, prompt_tokens, completion_tokens)
        await self._repo.record_usage(
            tenant_id=tenant_id,
            user_id=user_id,
            run_id=run_id,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            workflow=workflow,
        )

    async def get_usage(
        self,
        tenant_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        return await self._repo.get_usage(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    @staticmethod
    def summarize(usage_records: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate usage records into a summary."""
        total_tokens = 0
        total_cost = 0.0
        models_used: set[str] = set()
        workflows: dict[str, int] = {}

        for rec in usage_records:
            pt = rec.get("prompt_tokens", 0) or 0
            ct = rec.get("completion_tokens", 0) or 0
            total_tokens += pt + ct
            cost = rec.get("cost_usd")
            if cost is not None:
                total_cost += cost
            models_used.add(rec.get("model", "unknown"))
            wf = rec.get("workflow", "unknown")
            workflows[wf] = workflows.get(wf, 0) + 1

        return {
            "total_requests": len(usage_records),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "models_used": sorted(models_used),
            "workflows": workflows,
        }


__all__ = ["UsageService", "estimate_cost", "MODEL_COSTS"]
