"""
LLM provider implementations and factory.

Supports: Ollama, OpenAI, and OpenAI-compatible local/self-hosted endpoints.
Each provider implements the :class:`LLMProvider` interface defined in
``app.domain.interfaces``.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.enums import LLMProviderType, RiskLevel
from app.domain.interfaces import LLMProvider, LLMProviderFactory

logger = get_logger()

# --- Usage callback handler for token accounting ---
class UsageCallbackHandler(BaseCallbackHandler):
    """Callback handler that extracts token usage from LLM calls."""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.calls: int = 0

    def on_llm_end(self, response, *args: Any, **kwargs: Any) -> None:
        self.calls += 1
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("usage", {}) or {}
            self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
            self.completion_tokens += usage.get("completion_tokens", 0) or 0
            self.total_tokens += usage.get("total_tokens", 0) or 0

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.calls = 0


class LLMResponseImpl:
    """Concrete LLM response."""

    def __init__(
        self,
        content: str,
        model: str,
        usage: dict[str, int],
        finish_reason: str | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.content = content
        self.model = model
        self.usage = usage
        self.finish_reason = finish_reason
        self.raw = raw or {}


def _to_langchain_messages(messages: list[dict[str, Any]]) -> list[BaseMessage]:
    """Convert dict messages to LangChain messages."""
    lc_messages: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle multimodal content (keep as string for safety)
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(text_parts)
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif role == "tool":
            lc_messages.append(ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", "")))
        else:
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


class OllamaLLMProvider(LLMProvider):
    """LLM provider backed by Ollama."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._base_url = base_url or settings.OLLAMA_BASE_URL
        self._model = model or settings.OLLAMA_MODEL
        self._client: ChatOllama | None = None
        self._kwargs = kwargs

    @property
    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.OLLAMA

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> ChatOllama:
        if self._client is None:
            self._client = ChatOllama(
                base_url=self._base_url,
                model=self._model,
                temperature=0.7,
                **self._kwargs,
            )
        return self._client

    async def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponseImpl:
        client = self._get_client()
        kwargs_combined = dict(temperature=temperature, **kwargs)
        if max_tokens:
            kwargs_combined["max_tokens"] = max_tokens
        if stop:
            kwargs_combined["stop"] = stop

        usage_handler = UsageCallbackHandler()
        client = client.bind(callbacks=[usage_handler])  # type: ignore
        lc_messages = _to_langchain_messages(messages)
        response = await client.ainvoke(lc_messages, config={"callbacks": [usage_handler]})

        content = response.content if isinstance(response.content, str) else str(response.content)
        usage = {
            "prompt_tokens": usage_handler.prompt_tokens,
            "completion_tokens": usage_handler.completion_tokens,
            "total_tokens": usage_handler.total_tokens,
        }
        finish = response.additional_kwargs.get("finish_reason") if hasattr(response, "additional_kwargs") else None
        return LLMResponseImpl(
            content=content,
            model=self._model,
            usage=usage,
            finish_reason=finish or "stop",
            raw={"provider": "ollama"},
        )

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        client = self._get_client()
        kwargs_combined = dict(temperature=temperature, **kwargs)
        if max_tokens:
            kwargs_combined["max_tokens"] = max_tokens
        lc_messages = _to_langchain_messages(messages)
        async for chunk in client.astream(lc_messages, config={"configurable": kwargs_combined}):
            if isinstance(chunk, AIMessageChunk):
                content = chunk.content
            else:
                content = chunk
            if isinstance(content, str) and content:
                yield content

    async def get_embedding(
        self,
        text: str | list[str],
        **kwargs: Any,
    ) -> list[float] | list[list[float]]:
        # Ollama doesn't have a native embedding endpoint in the LLM provider;
        # use the embedding provider instead. This is a fallback.
        from app.services.service_container import get_service

        embedding_provider = get_service().embedding_provider
        return embedding_provider.embed_text(text)

    async def close(self) -> None:
        self._client = None


class OpenAICompatibleProvider(LLMProvider):
    """LLM provider backed by OpenAI or any OpenAI-compatible API."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._model = model or settings.OPENAI_MODEL
        self._base_url = base_url or settings.OPENAI_BASE_URL
        self._api_key = api_key or (settings.OPENAI_API_KEY.get_secret_value() if settings.OPENAI_API_KEY else None)
        self._client: ChatOpenAI | None = None
        self._kwargs = kwargs

    @property
    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.OPENAI

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> ChatOpenAI:
        if self._client is None:
            kwargs: dict[str, Any] = {"model": self._model, "temperature": 0.7}
            kwargs.update(self._kwargs)
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = ChatOpenAI(**kwargs)
        return self._client

    async def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponseImpl:
        client = self._get_client()
        usage_handler = UsageCallbackHandler()
        kwargs_combined: dict[str, Any] = dict(temperature=temperature, **kwargs)
        if max_tokens:
            kwargs_combined["max_tokens"] = max_tokens
        if stop:
            kwargs_combined["stop"] = stop

        lc_messages = _to_langchain_messages(messages)
        response = await client.ainvoke(lc_messages, config={"callbacks": [usage_handler]})

        content = response.content if isinstance(response.content, str) else str(response.content)
        usage_dict = response.usage_metadata or {}
        usage = {
            "prompt_tokens": usage_dict.get("input_tokens", 0),
            "completion_tokens": usage_dict.get("output_tokens", 0),
            "total_tokens": usage_dict.get("total_tokens", 0),
        }
        finish = response.usage_metadata
        return LLMResponseImpl(
            content=content,
            model=self._model,
            usage=usage,
            finish_reason="stop",
            raw={"provider": "openai"},
        )

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        client = self._get_client()
        kwargs_combined: dict[str, Any] = dict(temperature=temperature, **kwargs)
        if max_tokens:
            kwargs_combined["max_tokens"] = max_tokens
        lc_messages = _to_langchain_messages(messages)
        async for chunk in client.astream(lc_messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if isinstance(content, str) and content:
                yield content

    async def get_embedding(
        self,
        text: str | list[str],
        **kwargs: Any,
    ) -> list[float] | list[list[float]]:
        from app.services.service_container import get_service

        return await get_service().embedding_provider.embed_text(text)

    async def close(self) -> None:
        self._client = None


class LLMProviderFactoryImpl(LLMProviderFactory):
    """Factory that creates LLM provider instances by type.

    The factory registers provider constructors keyed by
    :class:`LLMProviderType`. The service container pre-registers the
    available providers and configures them from environment settings.
    """

    def __init__(self) -> None:
        self._providers: dict[LLMProviderType, type[LLMProvider]] = {}
        self._instances: dict[LLMProviderType, LLMProvider] = {}
        self._container: Any = None

    def register_providers(self, container: Any) -> None:
        """Register provider classes and cache the container reference."""
        self._container = container
        self._providers[LLMProviderType.OLLAMA] = OllamaLLMProvider
        self._providers[LLMProviderType.OPENAI] = OpenAICompatibleProvider
        self._providers[LLMProviderType.OPENAI_COMPATIBLE] = OpenAICompatibleProvider

    def create(
        self,
        provider_type: LLMProviderType,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> LLMProvider:
        """Create or return a cached LLM provider instance."""
        if provider_type in self._instances:
            return self._instances[provider_type]

        cls = self._providers.get(provider_type)
        if cls is None:
            from app.core.exceptions import ConfigError

            raise ConfigError(f"Unsupported LLM provider: {provider_type}")

        if provider_type == LLMProviderType.OLLAMA:
            provider = cls(
                base_url=settings.OLLAMA_BASE_URL,
                model=model_name or settings.OLLAMA_MODEL,
                **kwargs,
            )
        elif provider_type in (LLMProviderType.OPENAI, LLMProviderType.OPENAI_COMPATIBLE):
            provider = cls(
                model=model_name or settings.OPENAI_MODEL,
                base_url=settings.OPENAI_BASE_URL,
                api_key=settings.OPENAI_API_KEY.get_secret_value() if settings.OPENAI_API_KEY else None,
                **kwargs,
            )
        else:
            provider = cls(**kwargs)

        self._instances[provider_type] = provider
        logger.info("llm_provider_created", provider=provider_type.value, model=provider.model_name)
        return provider

    def get_provider(
        self, provider_type: str | LLMProviderType, model_name: str | None = None
    ) -> LLMProvider:
        """Get a provider by string name (case-insensitive). Resolves 'strong'/'fast'."""
        if isinstance(provider_type, str):
            provider_type = LLMProviderType(provider_type.lower())

        if provider_type == LLMProviderType.OPENAI:
            model = model_name or _resolve_model_for_use("default")
            return self.create(LLMProviderType.OPENAI, model_name=model)
        elif provider_type == LLMProviderType.OLLAMA:
            model = model_name or _resolve_model_for_use("default")
            return self.create(LLMProviderType.OLLAMA, model_name=model)
        return self.create(provider_type, model_name)

    def get_default_provider(self) -> LLMProvider:
        """Return the default LLM provider configured at startup."""
        return self.get_provider(settings.DEFAULT_LLM_PROVIDER)

    def get_strong_provider(self) -> LLMProvider:
        """Return the strong/reasoning LLM provider (for verification and analysis)."""
        return self.get_provider(settings.STRONG_LLM_PROVIDER, model_name=_resolve_strong_model())

    def get_fast_provider(self) -> LLMProvider:
        """Return the fast/cheap LLM provider (for classification)."""
        return self.get_provider(settings.FAST_LLM_PROVIDER, model_name=_resolve_fast_model())

    def close_all(self) -> None:
        """Close all cached provider instances."""
        import asyncio

        async def _close_all() -> None:
            for provider in self._instances.values():
                await provider.close()

        asyncio.get_event_loop().run_until_complete(_close_all())
        self._instances.clear()


def _resolve_model_for_use(use: str) -> str:
    if settings.DEFAULT_LLM_PROVIDER == "openai":
        return settings.OPENAI_MODEL
    return settings.OLLAMA_MODEL


def _resolve_strong_model() -> str:
    if settings.STRONG_LLM_PROVIDER == "openai":
        return settings.STRONG_OPENAI_MODEL
    return settings.STRONG_OLLAMA_MODEL


def _resolve_fast_model() -> str:
    if settings.FAST_LLM_PROVIDER == "openai":
        return settings.FAST_OPENAI_MODEL
    return settings.FAST_OLLAMA_MODEL


__all__ = [
    "OllamaLLMProvider",
    "OpenAICompatibleProvider",
    "LLMProviderFactoryImpl",
    "LLMResponseImpl",
    "UsageCallbackHandler",
]
