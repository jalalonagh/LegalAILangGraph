"""
Reranker provider implementations and factory.

Supports:
- TEI (Text Embeddings Inference) reranker
- Cohere reranker
- Jina reranker
- BGE cross-encoder (local)
- None (passthrough)
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.enums import RerankerProviderType
from app.domain.interfaces import RerankerProvider, RerankResult

logger = get_logger()


class TEIRerankerProvider(RerankerProvider):
    """Reranker backed by a TEI (HuggingFace Text Embeddings Inference) server."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        top_k: int | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = (base_url or settings.RERANKER_BASE_URL).rstrip("/")
        self._model = model or settings.RERANKER_MODEL
        self._top_k = top_k or settings.RERANKER_TOP_K
        self._api_key = api_key
        self._timeout = timeout

    @property
    def provider_type(self) -> RerankerProviderType:
        return RerankerProviderType.TEI

    @property
    def top_k(self) -> int:
        return self._top_k

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[RerankResult]:
        import httpx

        k = top_k or self._top_k
        if not candidates:
            return []

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "query": query,
            "texts": [c.get("content", str(c)) for c in candidates],
            "top_k": k,
            "model": self._model,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}/rerank", json=payload, headers=headers)
            resp.raise_for_status()
            results = resp.json()

        reranked: list[RerankResult] = []
        for item in results.get("results", results) if isinstance(results, dict) else results:
            idx = item.get("index", 0)
            score = item.get("score", 0.0)
            if idx < len(candidates):
                cand = candidates[idx]
                reranked.append(
                    RerankResultImpl(
                        index=idx,
                        score=float(score),
                        content=cand.get("content", str(cand)),
                        metadata=cand.get("metadata", {}),
                    )
                )
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:k]

    async def close(self) -> None:
        pass


class CohereRerankerProvider(RerankerProvider):
    """Reranker backed by Cohere's rerank API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        top_k: int | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or settings.OPENAI_API_KEY.get_secret_value() if settings.OPENAI_API_KEY else None
        self._model = model or "rerank-multilingual-v3.0"
        self._top_k = top_k or settings.RERANKER_TOP_K
        self._base_url = base_url or "https://api.cohere.ai"
        self._timeout = timeout

    @property
    def provider_type(self) -> RerankerProviderType:
        return RerankerProviderType.COHENE

    @property
    def top_k(self) -> int:
        return self._top_k

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[RerankResult]:
        import httpx

        if not candidates or not self._api_key:
            return [
                RerankResultImpl(
                    index=i,
                    score=1.0 - (i * 0.001),
                    content=c.get("content", str(c)),
                    metadata=c.get("metadata", {}),
                )
                for i, c in enumerate(candidates)
            ]

        k = top_k or self._top_k
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "query": query,
            "documents": [c.get("content", str(c)) for c in candidates],
            "top_n": k,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}/v1/rerank", json=payload, headers=headers)
            resp.raise_for_status()
            results = resp.json()

        reranked: list[RerankResult] = []
        for item in results.get("results", []):
            idx = item.get("index", 0)
            score = item.get("relevance_score", 0.0)
            if idx < len(candidates):
                cand = candidates[idx]
                reranked.append(
                    RerankResultImpl(
                        index=idx,
                        score=float(score),
                        content=cand.get("content", str(cand)),
                        metadata=cand.get("metadata", {}),
                    )
                )
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:k]

    async def close(self) -> None:
        pass


class JinaRerankerProvider(RerankerProvider):
    """Reranker backed by Jina AI's rerank API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        top_k: int | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or settings.OPENAI_API_KEY.get_secret_value() if settings.OPENAI_API_KEY else None
        self._model = model or "jina-reranker-v2-base-multilingual"
        self._top_k = top_k or settings.RERANKER_TOP_K
        self._timeout = timeout

    @property
    def provider_type(self) -> RerankerProviderType:
        return RerankerProviderType.JINA

    @property
    def top_k(self) -> int:
        return self._top_k

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[RerankResult]:
        import httpx

        if not candidates:
            return []

        k = top_k or self._top_k
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "query": query,
            "documents": [c.get("content", str(c)) for c in candidates],
            "top_k": k,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post("https://api.jina.ai/v1/rerank", json=payload, headers=headers)
            resp.raise_for_status()
            results = resp.json()

        reranked: list[RerankResult] = []
        for item in results.get("results", []):
            idx = item.get("index", 0)
            score = item.get("relevance_score", 0.0)
            if idx < len(candidates):
                cand = candidates[idx]
                reranked.append(
                    RerankResultImpl(
                        index=idx,
                        score=float(score),
                        content=cand.get("content", str(cand)),
                        metadata=cand.get("metadata", {}),
                    )
                )
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:k]

    async def close(self) -> None:
        pass


class BGECrossEncoderRerankerProvider(RerankerProvider):
    """Local reranker using a BGE cross-encoder from sentence-transformers."""

    def __init__(
        self,
        model: str | None = None,
        top_k: int | None = None,
    ) -> None:
        self._model_name = model or "BAAI/bge-reranker-v2-m3"
        self._top_k = top_k or settings.RERANKER_TOP_K
        self._cross_encoder: Any | None = None

    @property
    def provider_type(self) -> RerankerProviderType:
        return RerankerProviderType.BGE

    @property
    def top_k(self) -> int:
        return self._top_k

    def _get_model(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self._model_name)
        return self._cross_encoder

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[RerankResult]:
        if not candidates:
            return []

        k = top_k or self._top_k
        model = self._get_model()
        pairs = [(query, c.get("content", str(c))) for c in candidates]

        # CrossEncoder.predict is CPU-bound; run in thread pool
        scores = await asyncio.to_thread(model.predict, pairs)

        scored = [
            RerankResultImpl(
                index=i,
                score=float(scores[i]),
                content=c.get("content", str(c)),
                metadata=c.get("metadata", {}),
            )
            for i, c in enumerate(candidates)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    async def close(self) -> None:
        self._cross_encoder = None


class NoOpRerankerProvider(RerankerProvider):
    """Passthrough reranker that returns candidates sorted by retrieval score."""

    def __init__(self, top_k: int | None = None) -> None:
        self._top_k = top_k or settings.RERANKER_TOP_K

    @property
    def provider_type(self) -> RerankerProviderType:
        return RerankerProviderType.NONE

    @property
    def top_k(self) -> int:
        return self._top_k

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[RerankResult]:
        k = top_k or self._top_k
        # Sort by existing score (descending)
        sorted_candidates = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
        return [
            RerankResultImpl(
                index=i,
                score=float(c.get("score", 0.0)),
                content=c.get("content", str(c)),
                metadata=c.get("metadata", {}),
            )
            for i, c in enumerate(sorted_candidates[:k])
        ]

    async def close(self) -> None:
        pass


class RerankResultImpl:
    """Concrete implementation of RerankResult."""

    def __init__(self, index: int, score: float, content: str, metadata: dict[str, Any]) -> None:
        self.index = index
        self.score = score
        self.content = content
        self.metadata = metadata


class RerankerProviderFactoryImpl:
    """Factory for creating reranker providers."""

    _provider_map: dict[str, type[RerankerProvider]] = {
        "tei": TEIRerankerProvider,
        "cohere": CohereRerankerProvider,
        "jina": JinaRerankerProvider,
        "bge": BGECrossEncoderRerankerProvider,
        "none": NoOpRerankerProvider,
    }

    def create(
        self,
        provider_type: str | RerankerProviderType,
        **kwargs: Any,
    ) -> RerankerProvider:
        key = provider_type.value if isinstance(provider_type, RerankerProviderType) else provider_type.lower()

        cls = self._provider_map.get(key)
        if cls is None:
            from app.core.exceptions import ConfigError

            raise ConfigError(f"Unsupported reranker provider: {provider_type}")

        if cls is NoOpRerankerProvider:
            return cls(top_k=kwargs.get("top_k"))
        return cls(**kwargs)


__all__ = [
    "TEIRerankerProvider",
    "CohereRerankerProvider",
    "JinaRerankerProvider",
    "BGECrossEncoderRerankerProvider",
    "NoOpRerankerProvider",
    "RerankerProviderFactoryImpl",
    "RerankResultImpl",
]
