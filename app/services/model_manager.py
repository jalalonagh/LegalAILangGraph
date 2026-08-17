"""
Model manager for managing LLM model configurations.

Provides resolution of LLM providers by name, with fallback and
capability discovery.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.exceptions import ConfigError, NotFoundError
from app.core.logging import get_logger
from app.domain.enums import LLMProviderType
from app.infrastructure.llm.factory import LLMProviderFactoryImpl

logger = get_logger()


class ModelManager:
    """Manages AI model configurations and provider resolution."""

    def __init__(
        self,
        llm_factory: LLMProviderFactoryImpl,
        session=None,
    ) -> None:
        self._factory = llm_factory
        self._session = session

    async def get_provider(
        self,
        model_name: str,
        provider_type: str | None = None,
        tenant_id: str | None = None,
    ) -> Any:
        """Resolve an LLM provider for a given model name.

        If provider_type is not specified, it's inferred from model_name
        or from the default provider configuration.
        """
        if self._session and (provider_type is None or not provider_type.startswith("ollama")):
            from app.infrastructure.repositories import AIModelRepository

            repo = AIModelRepository(self._session)
            model = await repo.get_by_name(name=model_name, tenant_id=tenant_id or "demo")
            if model and model.enabled:
                provider_type = provider_type or model.provider
                return self._factory.create(
                    LLMProviderType(provider_type),
                    model_name=model.model_name,
                    base_url=model.api_endpoint,
                )

        if provider_type:
            try:
                pt = LLMProviderType(provider_type)
            except ValueError:
                raise ConfigError(f"Unknown provider type: {provider_type}")
        else:
            pt = LLMProviderType(settings.DEFAULT_LLM_PROVIDER)

        return self._factory.create(pt, model_name=model_name)

    async def get_default_provider(self) -> Any:
        return self._factory.get_default_provider()

    async def get_strong_provider(self) -> Any:
        return self._factory.get_strong_provider()

    async def get_fast_provider(self) -> Any:
        return self._factory.get_fast_provider()

    async def list_models(self, tenant_id: str = "demo", enabled_only: bool = True) -> list[dict[str, Any]]:
        if not self._session:
            return [
                {
                    "name": settings.OLLAMA_MODEL,
                    "provider": "ollama",
                    "model_name": settings.OLLAMA_MODEL,
                    "enabled": True,
                    "is_default": True,
                },
                {
                    "name": settings.OPENAI_MODEL,
                    "provider": "openai",
                    "model_name": settings.OPENAI_MODEL,
                    "enabled": settings.OPENAI_API_KEY is not None,
                    "is_default": False,
                },
            ]

        from app.infrastructure.repositories import AIModelRepository

        repo = AIModelRepository(self._session)
        models = await repo.list_enabled(tenant_id) if enabled_only else await repo.list(tenant_id=tenant_id)
        return [
            {
                "id": m.id,
                "name": m.name,
                "provider": m.provider,
                "model_name": m.model_name,
                "temperature": m.temperature,
                "max_tokens": m.max_tokens,
                "context_window": m.context_window,
                "enabled": m.enabled,
                "is_default": m.is_default,
                "capabilities": m.capabilities,
                "created_at": m.created_at.isoformat(),
                "updated_at": m.updated_at.isoformat(),
            }
            for m in models
        ]

    async def set_default_model(self, model_id: str, tenant_id: str) -> None:
        if not self._session:
            raise ConfigError("Database session required for model management")

        from app.infrastructure.repositories import AIModelRepository

        repo = AIModelRepository(self._session)
        model = await repo.get(model_id, tenant_id)
        if model is None:
            raise NotFoundError("Model not found", details={"model_id": model_id})

        await self._session.execute(
            "UPDATE ai_models SET is_default = false WHERE tenant_id = :tenant AND is_default = true",
            {"tenant": tenant_id},
        )
        model.is_default = True
        await self._session.flush()
        logger.info("model_default_set", model_id=model_id, tenant=tenant_id)

    async def close(self) -> None:
        pass


__all__ = ["ModelManager"]
