"""
Agent manager for managing AI agent configurations.

Agents are dynamically configurable where safe. The manager loads
agent definitions from the database and provides tools for
runtime configuration.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.infrastructure.repositories import AgentRepository, PromptRepository, PromptVersionRepository

logger = get_logger()


class AgentManager:
    """Manages AI agents, their tools, prompts, and model assignments."""

    def __init__(self, session, container) -> None:
        self._session = session
        self._container = container
        self._agent_repo = AgentRepository(session)
        self._prompt_repo = PromptRepository(session)
        self._prompt_version_repo = PromptVersionRepository(session)

    async def list_agents(self, tenant_id: str) -> list[dict[str, Any]]:
        agents = await self._agent_repo.list_enabled(tenant_id)
        result = []
        for agent in agents:
            result.append({
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "enabled": agent.enabled,
                "model_name": agent.model_name,
                "tool_names": agent.tool_names,
                "temperature": agent.temperature,
                "max_iterations": agent.max_iterations,
                "timeout_seconds": agent.timeout_seconds,
                "risk_level": agent.risk_level,
                "human_review_policy": agent.human_review_policy,
                "created_at": agent.created_at.isoformat(),
                "updated_at": agent.updated_at.isoformat(),
            })
        return result

    async def get_agent_config(self, agent_name: str, tenant_id: str) -> dict[str, Any] | None:
        """Retrieve the full runtime configuration for an agent, including its prompt."""
        agent = await self._agent_repo.get_by_name(agent_name, tenant_id)
        if agent is None:
            return None

        prompt_content = ""
        if agent.prompt_id:
            prompt = await self._prompt_repo.get(agent.prompt_id, tenant_id)
            if prompt and prompt.active_version_id:
                version = await self._prompt_version_repo.get(prompt.active_version_id, tenant_id)
                if version:
                    prompt_content = version.content

        return {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "enabled": agent.enabled,
            "model": agent.model_name,
            "prompt": prompt_content,
            "tools": agent.tool_names,
            "temperature": agent.temperature,
            "max_iterations": agent.max_iterations,
            "timeout_seconds": agent.timeout_seconds,
            "risk_level": agent.risk_level,
            "human_review_policy": agent.human_review_policy,
        }

    async def resolve_provider_for_agent(self, agent_name: str, tenant_id: str):
        """Resolve the LLM provider configured for a given agent."""
        config = await self.get_agent_config(agent_name, tenant_id)
        if config is None:
            return None, 0.7, 10
        model_manager = self._container.get_model_manager(self._session)
        provider = await model_manager.get_provider(
            config["model"], tenant_id=tenant_id
        )
        return provider, config["temperature"], config["max_iterations"]

    async def get_tool_config(self, agent_name: str, tool_name: str, tenant_id: str) -> dict[str, Any] | None:
        """Get the configuration for a specific tool used by an agent."""
        from app.infrastructure.repositories import AgentToolRepository

        agent = await self._agent_repo.get_by_name(agent_name, tenant_id)
        if agent is None:
            return None

        tool_repo = AgentToolRepository(self._session)
        tools = await tool_repo.list_for_agent(agent.id, tenant_id)
        for tool in tools:
            if tool.tool_name == tool_name:
                return {
                    "name": tool.tool_name,
                    "authorization_required": tool.authorization_required,
                    "risk_level": tool.risk_level,
                    "timeout_seconds": tool.timeout_seconds,
                    "retry_count": tool.retry_count,
                    "audit_enabled": tool.audit_enabled,
                }
        return None

    async def list_all_tools(self, tenant_id: str) -> list[dict[str, Any]]:
        """List all available tools for the tenant (from tool registry)."""
        from app.agents.tools.registry import get_tool_registry

        registry = get_tool_registry()
        return registry.list_tools(tenant_id=tenant_id)

    async def close(self) -> None:
        pass


__all__ = ["AgentManager"]
