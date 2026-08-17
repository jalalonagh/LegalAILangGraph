"""
Knowledge base management service.

Handles document ingestion, chunking, embedding, and indexing pipeline:
Upload → Validation → Parsing → OCR → Cleaning → Metadata extraction
→ Legal structure extraction → Chunking → Embedding → Vector indexing
→ Keyword indexing → Verification.
"""

from __future__ import annotations

import asyncio
import tempfile
import os
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import (
    FileSizeExceededError,
    UnsupportedFileTypeError,
    VectorStoreError,
)
from app.core.logging import get_logger
from app.domain.enums import DocumentParseFormat
from app.domain.interfaces import DocumentParser, EmbeddingProvider, VectorStore
from app.domain.value_objects import DocumentMetadata, ChunkMetadata, classify_file_format
from app.rag.chunking.legal_chunker import LegalDocumentChunker
from app.rag.ingestion.pipeline import IngestionPipeline

logger = get_logger()


class KnowledgeBaseManager:
    """Manages knowledge bases, document ingestion, and re-indexing."""

    def __init__(
        self,
        session=None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        document_parser: DocumentParser | None = None,
    ) -> None:
        self._session = session
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._parser = document_parser
        self._chunker = LegalDocumentChunker()

    async def create_knowledge_base(
        self,
        name: str,
        description: str,
        vector_collection: str,
        jurisdiction: str = "us_federal",
        legal_domain: str = "general",
        embedding_model: str = "",
        embedding_dim: int = 1024,
        tenant_id: str = "demo",
    ) -> str:
        """Create a new knowledge base."""
        if not self._session:
            raise RuntimeError("Database session required")

        from app.infrastructure.repositories import KnowledgeBaseRepository

        repo = KnowledgeBaseRepository(self._session)
        existing = await repo.get_by_name(name, tenant_id)
        if existing:
            from app.core.exceptions import ConfigError

            raise ConfigError(f"Knowledge base '{name}' already exists")

        kb = await repo.add(
            type(repo.model)(
                name=name,
                description=description,
                vector_collection=vector_collection,
                jurisdiction=jurisdiction,
                legal_domain=legal_domain,
                embedding_model=embedding_model or settings.EMBEDDING_MODEL,
                embedding_dim=embedding_dim,
                enabled=True,
                tenant_id=tenant_id,
            )
        )
        await self._session.flush()

        # Create the vector collection
        if self._vector_store:
            await self._vector_store.create_collection(
                collection_name=vector_collection,
                vector_size=embedding_dim,
            )
        logger.info("knowledge_base_created", name=name, kb_id=kb.id, tenant=tenant_id)
        return kb.id

    async def get_knowledge_base(self, kb_id: str, tenant_id: str) -> dict[str, Any] | None:
        if not self._session:
            return None
        from app.infrastructure.repositories import KnowledgeBaseRepository, KnowledgeDocumentRepository

        kb_repo = KnowledgeBaseRepository(self._session)
        kb = await kb_repo.get(kb_id, tenant_id)
        if kb is None:
            return None

        doc_repo = KnowledgeDocumentRepository(self._session)
        doc_count = await doc_repo.count(tenant_id=tenant_id)
        chunk_count = 0
        if self._vector_store and await self._vector_store.collection_exists(kb.vector_collection):
            chunk_count = await self._vector_store.count(kb.vector_collection)

        return {
            "id": kb.id,
            "name": kb.name,
            "description": kb.description,
            "vector_collection": kb.vector_collection,
            "jurisdiction": kb.jurisdiction,
            "legal_domain": kb.legal_domain,
            "embedding_model": kb.embedding_model,
            "embedding_dim": kb.embedding_dim,
            "enabled": kb.enabled,
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "created_at": kb.created_at.isoformat(),
            "updated_at": kb.updated_at.isoformat(),
        }

    async def list_knowledge_bases(self, tenant_id: str) -> list[dict[str, Any]]:
        if not self._session:
            return []
        from app.infrastructure.repositories import KnowledgeBaseRepository

        repo = KnowledgeBaseRepository(self._session)
        kbs = await repo.list_enabled(tenant_id)
        result = []
        for kb in kbs:
            result.append(await self.get_knowledge_base(kb.id, tenant_id))
        return [r for r in result if r is not None]

    async def delete_knowledge_base(self, kb_id: str, tenant_id: str) -> None:
        if not self._session:
            raise RuntimeError("Database session required")
        from app.infrastructure.repositories import KnowledgeBaseRepository

        repo = KnowledgeBaseRepository(self._session)
        kb = await repo.get(kb_id, tenant_id)
        if kb is None:
            from app.core.exceptions import NotFoundError

            raise NotFoundError("Knowledge base not found", details={"kb_id": kb_id})

        # Delete vector collection
        if self._vector_store:
            try:
                await self._vector_store.delete_collection(kb.vector_collection)
            except Exception as exc:  # noqa: BLE001
                logger.warning("vector_collection_delete_failed", error=str(exc))

        await repo.delete(kb_id, tenant_id)
        logger.info("knowledge_base_deleted", kb_id=kb_id, tenant=tenant_id)

    async def ingest_document(
        self,
        kb_id: str,
        file_data: bytes,
        filename: str,
        mime_type: str,
        metadata: dict[str, Any] | None = None,
        tenant_id: str = "demo",
    ) -> dict[str, Any]:
        """Run the full ingestion pipeline for a document."""
        if not self._session:
            raise RuntimeError("Database session required")

        from app.infrastructure.repositories import KnowledgeBaseRepository, KnowledgeDocumentRepository

        # Validate file
        if len(file_data) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise FileSizeExceededError(
                f"File exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit",
                details={"size": len(file_data), "max": settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024},
            )

        fmt = classify_file_format(filename)
        if fmt is None:
            raise UnsupportedFileTypeError(
                "Unsupported file format",
                details={"filename": filename, "mime_type": mime_type},
            )

        kb = await KnowledgeBaseRepository(self._session).get(kb_id, tenant_id)
        if kb is None:
            from app.core.exceptions import NotFoundError

            raise NotFoundError("Knowledge base not found", details={"kb_id": kb_id})

        # Ensure vector collection exists
        if self._vector_store and not await self._vector_store.collection_exists(kb.vector_collection):
            await self._vector_store.create_collection(
                collection_name=kb.vector_collection,
                vector_size=kb.embedding_dim,
            )

        # Run ingestion pipeline
        pipeline = IngestionPipeline(
            parser=self._parser,
            embedding_provider=self._embedding_provider,
            vector_store=self._vector_store,
            chunker=self._chunker,
        )
        result = await pipeline.run(
            file_data=file_data,
            filename=filename,
            mime_type=mime_type,
            kb=kb,
            metadata=metadata or {},
            tenant_id=tenant_id,
            session=self._session,
        )
        return result

    async def reindex_knowledge_base(self, kb_id: str, tenant_id: str, force: bool = False) -> dict[str, Any]:
        """Re-index all documents in a knowledge base."""
        if not self._session:
            raise RuntimeError("Database session required")
        from app.infrastructure.repositories import KnowledgeDocumentRepository, KnowledgeChunkRepository

        doc_repo = KnowledgeDocumentRepository(self._session)
        documents = await doc_repo.list_by_status(
            status="indexed", kb_id=kb_id, tenant_id=tenant_id
        ) if not force else await doc_repo.list(tenant_id=tenant_id)

        chunk_repo = KnowledgeChunkRepository(self._session)
        # Delete existing chunks
        for doc in documents:
            await chunk_repo.delete_by_document(doc.id, tenant_id)

        # Re-ingest
        pipeline = IngestionPipeline(
            parser=self._parser,
            embedding_provider=self._embedding_provider,
            vector_store=self._vector_store,
            chunker=self._chunker,
        )
        results = []
        for doc in documents:
            try:
                with open(doc.file_path, "rb") as f:
                    file_data = f.read()
                result = await pipeline.run(
                    file_data=file_data,
                    filename=doc.file_name,
                    mime_type=doc.mime_type,
                    kb=kb_repo_get_kb(self._session, kb_id),
                    metadata=doc.metadata_ or {},
                    tenant_id=tenant_id,
                    session=self._session,
                )
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.error("reindex_document_failed", doc_id=doc.id, error=str(exc))
                results.append({"document_id": doc.id, "status": "failed", "error": str(exc)})

        return {
            "total_documents": len(documents),
            "results": results,
        }

    async def delete_document(self, doc_id: str, tenant_id: str) -> None:
        if not self._session:
            raise RuntimeError("Database session required")
        from app.infrastructure.repositories import KnowledgeDocumentRepository, KnowledgeChunkRepository

        doc_repo = KnowledgeDocumentRepository(self._session)
        chunk_repo = KnowledgeChunkRepository(self._session)
        doc = await doc_repo.get(doc_id, tenant_id)
        if doc is None:
            from app.core.exceptions import NotFoundError

            raise NotFoundError("Document not found", details={"doc_id": doc_id})

        # Delete vectors
        if self._vector_store:
            chunks = await chunk_repo.list(tenant_id=tenant_id)
            chunk_ids = [c.vector_id for c in chunks if c.document_id == doc_id]
            if chunk_ids:
                await self._vector_store.delete_points(doc.kb_id, chunk_ids)

        await doc_repo.delete(doc_id, tenant_id)
        logger.info("document_deleted", doc_id=doc_id, tenant=tenant_id)

    async def close(self) -> None:
        pass


def _get_doc_repo(session, kb_id, tenant_id):
    pass


async def kb_repo_get_kb(session, kb_id):
    from app.infrastructure.repositories import KnowledgeBaseRepository

    return await KnowledgeBaseRepository(session).get(kb_id, "")


__all__ = ["KnowledgeBaseManager"]
