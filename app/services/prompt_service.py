"""
Prompt management service with versioning support.

Prompts are stored in the database with version history. Only one
version can be active at a time per prompt name/workflow.
"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.infrastructure.repositories import PromptRepository, PromptVersionRepository

logger = get_logger()


class PromptService:
    """Manages prompt retrieval and versioning."""

    def __init__(
        self,
        prompt_repo: PromptRepository | None = None,
        version_repo: PromptVersionRepository | None = None,
    ) -> None:
        self._prompt_repo = prompt_repo
        self._version_repo = version_repo

    def _set_repos(self, prompt_repo, version_repo) -> None:
        self._prompt_repo = prompt_repo
        self._version_repo = version_repo

    async def get_prompt(
        self,
        name: str,
        workflow: str | None = None,
        tenant_id: str = "demo",
        version: str | None = None,
    ) -> str:
        """Get the active (or specified) prompt content for a named prompt.

        Falls back to a file-based prompt if the database entry is not found.
        """
        if self._prompt_repo and self._version_repo:
            prompt = await self._prompt_repo.get_by_name(name, tenant_id)
            if prompt is not None:
                if version:
                    pv = await self._version_repo.get_by_version(prompt.id, version, tenant_id)
                else:
                    pv = await self._version_repo.get_latest_active(prompt.id, tenant_id)
                if pv is not None:
                    return pv.content

        # Fallback: file-based prompts in app/agents/prompts/
        from pathlib import Path

        candidates = []
        if workflow:
            candidates.append(Path(f"app/agents/prompts/{workflow}/{name}.md"))
        candidates.append(Path(f"app/agents/prompts/{name}.md"))
        candidates.append(Path(f"app/agents/prompts/{name}.txt"))
        candidates.append(Path(f"prompts/{name}.md"))

        for path in candidates:
            if path.exists():
                return path.read_text(encoding="utf-8")

        logger.warning("prompt_not_found", name=name, workflow=workflow, tenant=tenant_id)
        return ""

    async def list_prompts(self, tenant_id: str = "demo") -> list[dict[str, Any]]:
        if not self._prompt_repo:
            return []
        prompts = await self._prompt_repo.list(tenant_id=tenant_id)
        result = []
        for p in prompts:
            result.append({
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "workflow": p.workflow,
                "agent": p.agent,
                "active_version_id": p.active_version_id,
                "created_at": p.created_at.isoformat(),
                "updated_at": p.updated_at.isoformat(),
            })
        return result

    async def create_prompt(
        self,
        name: str,
        content: str,
        tenant_id: str,
        version: str = "1.0",
        created_by: str = "system",
        workflow: str | None = None,
        agent: str | None = None,
        description: str = "",
    ) -> str:
        """Create a new prompt with its first version."""
        if not self._prompt_repo or not self._version_repo:
            raise RuntimeError("Prompt repository not initialized")

        prompt = await self._prompt_repo.add(
            type(self._prompt_repo.model)(
                name=name,
                description=description,
                workflow=workflow,
                agent=agent,
                tenant_id=tenant_id,
            )
        )
        await self._version_repo.create_version(
            prompt_id=prompt.id,
            content=content,
            version=version,
            created_by=created_by,
            tenant_id=tenant_id,
            status="active",
        )
        await self._prompt_repo.update(
            prompt.id,
            tenant_id,
            {"active_version_id": (await self._version_repo.get_latest_active(prompt.id, tenant_id)).id}
            if False
            else {},
        )  # no-op, version created above
        logger.info("prompt_created", name=name, version=version, tenant=tenant_id)
        return prompt.id

    async def create_new_version(
        self,
        prompt_id: str,
        content: str,
        version: str,
        tenant_id: str,
        created_by: str = "system",
    ) -> str:
        """Create a new version of an existing prompt and make it active."""
        if not self._version_repo:
            raise RuntimeError("Prompt repository not initialized")

        # Deactivate existing active versions
        await self._version_repo.session.execute(
            self._version_repo.model.__table__.update()
            .where(
                self._version_repo.model.prompt_id == prompt_id,
                self._version_repo.model.tenant_id == tenant_id,
                self._version_repo.model.is_active.is_(True),
            )
            .values(is_active=False, status="archived")
        )

        pv = await self._version_repo.create_version(
            prompt_id=prompt_id,
            content=content,
            version=version,
            created_by=created_by,
            tenant_id=tenant_id,
            status="active",
        )
        # Update prompt's active version
        await self._prompt_repo.update(
            prompt_id,
            tenant_id,
            {"active_version_id": pv.id},
        )
        logger.info("prompt_version_created", prompt_id=prompt_id, version=version, tenant=tenant_id)
        return pv.id

    async def render(
        self,
        name: str,
        variables: dict[str, str],
        workflow: str | None = None,
        tenant_id: str = "demo",
    ) -> str:
        """Get a prompt and render it with variables."""
        template = await self.get_prompt(name, workflow, tenant_id)
        if not template:
            return ""
        for key, value in variables.items():
            template = template.replace("{{" + key + "}}", str(value))
        return template


__all__ = ["PromptService"]
