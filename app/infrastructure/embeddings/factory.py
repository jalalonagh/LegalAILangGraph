"""
Embedding provider implementations and factory.

Supports:
- Sentence Transformers (local models)
- OpenAI embeddings
- Ollama embeddings
"""

from __future__ import annotations

from typing import Any

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.enums import EmbeddingProviderType
from app.domain.interfaces import EmbeddingProvider

logger = get_logger()


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""

    def __init__(
        self,
        model: str | None = None,
        device: str = "cpu",
    ) -> None:
        self._model = model or settings.EMBEDDING_MODEL
        self._device = device
        self._client: SentenceTransformerEmbeddings | None = None

    @property
    def provider_type(self) -> EmbeddingProviderType:
        return EmbeddingProviderType.SENTENCE_TRANSFORMERS

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        # Lazy-init to get dimension
        self._get_client()
        # langchain SentenceTransformerEmbeddings exposes _model with dim attribute
        model = self._client
        if model is not None and hasattr(model, "_model"):
            return getattr(model._model, "dim", settings.EMBEDDING_DIM)
        return settings.EMBEDDING_DIM

    def _get_client(self) -> SentenceTransformerEmbeddings:
        if self._client is None:
            try:
                self._client = SentenceTransformerEmbeddings(
                    model_name=self._model,
                    model_kwargs={"device": self._device},
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("sentence_transformer_load_failed", error=str(exc))
                raise
        return self._client

    async def embed_text(self, text: str | list[str]) -> list[float] | list[list[float]]:
        client = self._get_client()
        if isinstance(text, str):
            return client.embed_query(text)
        return await self.embed_documents(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        # Use batched embedding
        batch_size = 32
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_vectors = client.embed_documents(batch)
            vectors.extend(batch_vectors)
        return vectors

    async def close(self) -> None:
        self._client = None


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model or "text-embedding-3-small"
        self._api_key = api_key
        self._base_url = base_url
        self._client: OpenAIEmbeddings | None = None

    @property
    def provider_type(self) -> EmbeddingProviderType:
        return EmbeddingProviderType.OPENAI

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        if self._model == "text-embedding-3-small":
            return 1536
        return 1024

    def _get_client(self) -> OpenAIEmbeddings:
        if self._client is None:
            self._client = OpenAIEmbeddings(
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    async def embed_text(self, text: str | list[str]) -> list[float] | list[list[float]]:
        client = self._get_client()
        if isinstance(text, str):
            return client.embed_query(text)
        return await self.embed_documents(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        batch_size = 100
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors.extend(client.embed_documents(batch))
        return vectors

    async def close(self) -> None:
        self._client = None


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model or settings.OLLAMA_EMBEDDING_MODEL
        self._base_url = base_url or settings.OLLAMA_BASE_URL
        self._client: OllamaEmbeddings | None = None

    @property
    def provider_type(self) -> EmbeddingProviderType:
        return EmbeddingProviderType.OLLAMA

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        # bge-m3 typically 1024 dims
        return 1024

    def _get_client(self) -> OllamaEmbeddings:
        if self._client is None:
            self._client = OllamaEmbeddings(model=self._model, base_url=self._base_url)
        return self._client

    async def embed_text(self, text: str | list[str]) -> list[float] | list[list[float]]:
        client = self._get_client()
        if isinstance(text, str):
            return client.embed_query(text)
        return await self.embed_documents(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        batch_size = 32
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors.extend(client.embed_documents(batch))
        return vectors

    async def close(self) -> None:
        self._client = None


class EmbeddingProviderFactoryImpl:
    """Factory for creating embedding providers."""

    _provider_map: dict[EmbeddingProviderType, type[EmbeddingProvider]] = {
        EmbeddingProviderType.SENTENCE_TRANSFORMERS: SentenceTransformerEmbeddingProvider,
        EmbeddingProviderType.OPENAI: OpenAIEmbeddingProvider,
        EmbeddingProviderType.OLLAMA: OllamaEmbeddingProvider,
    }

    def create(
        self,
        provider_type: str | EmbeddingProviderType,
        model: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingProvider:
        if isinstance(provider_type, str):
            provider_type = EmbeddingProviderType(provider_type)

        cls = self._provider_map.get(provider_type)
        if cls is None:
            from app.core.exceptions import ConfigError

            raise ConfigError(f"Unsupported embedding provider: {provider_type}")

        if provider_type == EmbeddingProviderType.SENTENCE_TRANSFORMERS:
            return cls(
                model=model or settings.EMBEDDING_MODEL,
                device=settings.SENTENCE_TRANSFORMERS_DEVICE,
                **kwargs,
            )
        elif provider_type == EmbeddingProviderType.OPENAI:
            return cls(
                model=model,
                api_key=settings.OPENAI_API_KEY.get_secret_value() if settings.OPENAI_API_KEY else None,
                base_url=settings.OPENAI_BASE_URL,
                **kwargs,
            )
        elif provider_type == EmbeddingProviderType.OLLAMA:
            return cls(model=model or settings.OLLAMA_EMBEDDING_MODEL, **kwargs)
        return cls(**kwargs)


__all__ = [
    "SentenceTransformerEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "EmbeddingProviderFactoryImpl",
]
