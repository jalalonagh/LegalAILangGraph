"""
Qdrant vector store implementation.

Provides hybrid search (vector + keyword), tenant isolation via
collection-per-tenant or namespace-based filtering, and full
metadata filtering support.
"""

from __future__ import annotations

import time
from typing import Any

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, PayloadSchemaType, VectorParams

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.enums import AuthorityLevel
from app.domain.interfaces import VectorStore
from app.domain.value_objects import EvidenceChunk, ChunkMetadata, LegalCitation

logger = get_logger()

# Authority level ranking for scoring
AUTHORITY_WEIGHTS = {
    AuthorityLevel.LEVEL_1_OFFICIAL_LEGISLATION: 1.0,
    AuthorityLevel.LEVEL_2_OFFICIAL_COURT_DECISIONS: 0.95,
    AuthorityLevel.LEVEL_3_OFFICIAL_LEGAL_OPINIONS: 0.9,
    AuthorityLevel.LEVEL_4_TRUSTED_DATABASES: 0.8,
    AuthorityLevel.LEVEL_5_SECONDARY_SOURCES: 0.5,
    AuthorityLevel.LEVEL_6_GENERAL_WEB: 0.2,
}


class QdrantVectorStore(VectorStore):
    """Vector store backed by Qdrant with hybrid search support."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        grpc_port: int | None = None,
        prefer_grpc: bool = False,
    ) -> None:
        self._url = url or settings.VECTOR_DB_URL
        self._api_key = api_key or (
            settings.VECTOR_DB_API_KEY.get_secret_value() if settings.VECTOR_DB_API_KEY else None
        )
        self._grpc_port = grpc_port
        self._client: AsyncQdrantClient | None = None
        self._sync_client: QdrantClient | None = None

    def _get_async_client(self) -> AsyncQdrantClient:
        if self._client is None:
            opts: dict[str, Any] = {"url": self._url}
            if self._api_key:
                opts["api_key"] = self._api_key
            self._client = AsyncQdrantClient(**opts)
        return self._client

    def _get_sync_client(self) -> QdrantClient:
        if self._sync_client is None:
            opts: dict[str, Any] = {"url": self._url}
            if self._api_key:
                opts["api_key"] = self._api_key
            if self._grpc_port:
                opts["grpc_port"] = self._grpc_port
                opts["prefer_grpc"] = True
            self._sync_client = QdrantClient(**opts)
        return self._sync_client

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        **kwargs: Any,
    ) -> bool:
        client = self._get_async_client()
        exists = await client.collection_exists(collection_name)
        if exists:
            logger.info("qdrant_collection_exists", collection=collection_name)
            return False

        await client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "": VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                    on_disk_payload=True,
                )
            },
        )
        await client.create_payload_index(
            collection_name=collection_name,
            field="content",
            field_schema=PayloadSchemaType.TEXT,
        )
        logger.info("qdrant_collection_created", collection=collection_name, vector_size=vector_size)
        return True

    async def delete_collection(self, collection_name: str) -> None:
        client = self._get_async_client()
        await client.delete_collection(collection_name)
        logger.info("qdrant_collection_deleted", collection=collection_name)

    async def collection_exists(self, collection_name: str) -> bool:
        client = self._get_async_client()
        return await client.collection_exists(collection_name)

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------
    async def upsert(
        self,
        collection_name: str,
        points: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        if not points:
            return
        client = self._get_async_client()
        qdrant_points = []
        for point_id, vector, payload in points:
            payload_copy = dict(payload)
            qdrant_points.append(
                rest.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload_copy,
                )
            )
        await client.upsert(collection_name=collection_name, points=qdrant_points, wait=True)
        logger.info("qdrant_upserted", collection=collection_name, count=len(points))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> rest.Filter | None:
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, dict):
                conditions.append(rest.FieldCondition(key=key, match=rest.MatchValue(value=value)))
            elif isinstance(value, list):
                conditions.append(rest.FieldCondition(key=key, match=rest.MatchAny(any=value)))
            else:
                conditions.append(rest.FieldCondition(key=key, match=rest.MatchValue(value=value)))
        if not conditions:
            return None
        return rest.Filter(must=conditions)

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[EvidenceChunk]:
        client = self._get_async_client()
        qdrant_filter = self._build_filter(filters)
        results = await client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            filter=qdrant_filter,
            score_threshold=kwargs.get("score_threshold"),
        )

        evidence: list[EvidenceChunk] = []
        for hit in results:
            payload = hit.payload or {}
            evidence.append(self._hit_to_evidence(hit, payload, float(hit.score), "vector"))
        logger.info("qdrant_search_completed", collection=collection_name, hits=len(evidence))
        return evidence

    async def search_with_keyword(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[EvidenceChunk]:
        """Hybrid search: vector + keyword with RRF (Reciprocal Rank Fusion)."""
        client = self._get_async_client()
        qdrant_filter = self._build_filter(filters)

        vector_results: list = []
        if kwargs.get("query_vector"):
            vector_results = await client.search(
                collection_name=collection_name,
                query_vector=kwargs.get("query_vector"),
                limit=top_k * 2,
                with_payload=True,
                with_vectors=False,
                filter=qdrant_filter,
            )

        keyword_hits: list = []
        try:
            keyword_hits = await client.search(
                collection_name=collection_name,
                query_text=query,
                limit=top_k * 2,
                with_payload=True,
                with_vectors=False,
                filter=qdrant_filter,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("qdrant_keyword_search_fallback", error=str(exc))

        # RRF fusion
        rrf_k = 60
        scores: dict[str, float] = {}

        for rank, hit in enumerate(vector_results):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (rrf_k + rank)

        for rank, hit in enumerate(keyword_hits):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        all_hits = {hit.id: hit for hit in vector_results}
        for hit in keyword_hits:
            if hit.id not in all_hits:
                all_hits[hit.id] = hit

        evidence: list[EvidenceChunk] = []
        for pid, score in ranked:
            hit = all_hits.get(pid)
            if hit is None:
                continue
            payload = hit.payload or {}
            evidence.append(self._hit_to_evidence(hit, payload, score, "hybrid"))

        logger.info("qdrant_hybrid_search_completed", collection=collection_name, hits=len(evidence))
        return evidence

    def _hit_to_evidence(
        self,
        hit: Any,
        payload: dict[str, Any],
        score: float,
        retrieval_method: str,
    ) -> EvidenceChunk:
        metadata = ChunkMetadata(
            chunk_id=payload.get("chunk_id", hit.id),
            document_id=payload.get("document_id", ""),
            section_id=payload.get("section_id", ""),
            article_number=payload.get("article_number", ""),
            paragraph_number=payload.get("paragraph_number", ""),
            source=payload.get("source", ""),
            version=payload.get("version", ""),
            page=payload.get("page", 0),
            offset=payload.get("offset", 0),
            boundary=payload.get("boundary", "paragraph"),
            content=payload.get("content", ""),
        )
        try:
            auth_level = AuthorityLevel(
                payload.get("authority_level", AuthorityLevel.LEVEL_5_SECONDARY_SOURCES.value)
            )
        except ValueError:
            auth_level = AuthorityLevel.LEVEL_5_SECONDARY_SOURCES

        return EvidenceChunk(
            chunk_id=metadata.chunk_id,
            document_id=metadata.document_id,
            content=payload.get("content", ""),
            metadata=metadata,
            score=score,
            retrieval_method=retrieval_method,
            authority_level=auth_level,
            verified=False,
        )

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------
    async def delete_points(
        self,
        collection_name: str,
        point_ids: list[str],
    ) -> None:
        client = self._get_async_client()
        if point_ids:
            await client.delete(collection_name=collection_name, ids=point_ids, wait=True)

    async def count(self, collection_name: str, filters: dict[str, Any] | None = None) -> int:
        client = self._get_async_client()
        qdrant_filter = self._build_filter(filters)
        result = await client.count(collection_name=collection_name, filter=qdrant_filter)
        return result.count

    async def get_point(self, collection_name: str, point_id: str) -> dict[str, Any] | None:
        client = self._get_async_client()
        result = await client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not result:
            return None
        hit = result[0]
        return {"id": hit.id, "payload": hit.payload, "score": hit.score}

    async def get_info(self, collection_name: str) -> dict[str, Any]:
        client = self._get_async_client()
        info = await client.get_collection_info(collection_name=collection_name)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._sync_client = None

    def get_sync_client(self) -> QdrantClient:
        return self._get_sync_client()


__all__ = ["QdrantVectorStore", "AUTHORITY_WEIGHTS"]
