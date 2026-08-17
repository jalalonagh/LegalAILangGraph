"""
Service container for dependency injection.

Provides a singleton service locator that lazily constructs and caches
infrastructure services (LLM providers, vector store, repositories, etc.).
This is the composition root for the application.
"""

from __future__ import annotations

import asyncio
from functools import cached_property
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.interfaces import (
    AuditRepository,
    EmbeddingProvider,
    HumanReviewRepository,
    LegalDocumentRepository,
    LLMProviderFactory,
    OCRProvider,
    RerankerProvider,
    UsageRepository,
    UserDocumentRepository,
    VectorStore,
)
from app.domain.value_objects import DocumentMetadata

logger = get_logger()


class ServiceContainer:
    """Central service container that resolves all infrastructure dependencies."""

    def __init__(self) -> None:
        self._initialized = False
        self._lock = asyncio.Lock()
        self._overrides: dict[type, Any] = {}

    # ------------------------------------------------------------------
    # Lazy singletons via cached_property
    # ------------------------------------------------------------------
    @cached_property
    def llm_factory(self) -> LLMProviderFactory:
        from app.infrastructure.llm.factory import LLMProviderFactoryImpl

        factory = self._overrides.get(LLMProviderFactory) or LLMProviderFactoryImpl()
        factory.register_providers(self)
        return factory

    @cached_property
    def embedding_provider(self) -> EmbeddingProvider:
        from app.infrastructure.embeddings.factory import EmbeddingProviderFactoryImpl

        factory = EmbeddingProviderFactoryImpl()
        return factory.create(settings.EMBEDDING_PROVIDER)

    @cached_property
    def reranker_provider(self) -> RerankerProvider | None:
        from app.infrastructure.reranking.factory import RerankerProviderFactoryImpl

        if not settings.RERANKER_PROVIDER or settings.RERANKER_PROVIDER == "none":
            return None
        factory = RerankerProviderFactoryImpl()
        return factory.create(settings.RERANKER_PROVIDER)

    @cached_property
    def vector_store(self) -> VectorStore:
        from app.infrastructure.vectorstore.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(
            url=settings.VECTOR_DB_URL,
            api_key=settings.VECTOR_DB_API_KEY.get_secret_value() if settings.VECTOR_DB_API_KEY else None,
            grpc_port=settings.VECTOR_DB_GRPC_PORT,
        )

    @cached_property
    def ocr_provider(self) -> OCRProvider | None:
        if not settings.FEATURE_OCR:
            return None
        from app.infrastructure.external_services.ocr_provider import TesseractOCRProvider

        return TesseractOCRProvider(
            language=settings.OCR_LANGUAGE,
            tesseract_cmd=settings.TESSERACT_CMD,
        )

    @cached_property
    def document_parser(self):
        from app.infrastructure.external_services.document_parser import UniversalDocumentParser

        return UniversalDocumentParser(ocr_provider=self.ocr_provider)

    @cached_property
    def memory_services(self):
        from app.memory.case_memory import CaseMemoryService
        from app.memory.conversation_memory import ConversationMemoryService
        from app.memory.user_memory import UserMemoryService

        return {
            "conversation": ConversationMemoryService(self),
            "case": CaseMemoryService(self),
            "user": UserMemoryService(self),
        }

    # ------------------------------------------------------------------
    # Repositories (need a DB session, so we provide factory methods)
    # ------------------------------------------------------------------
    def get_legal_document_repository(self, session) -> LegalDocumentRepository:
        from app.infrastructure.repositories.legal_document_repo import LegalDocumentRepositoryImpl

        return self._overrides.get(LegalDocumentRepository) or LegalDocumentRepositoryImpl(
            session=session, vector_store=self.vector_store
        )

    def get_user_document_repository(self, session) -> UserDocumentRepository:
        from app.infrastructure.repositories.user_document_repo import UserDocumentRepositoryImpl

        return self._overrides.get(UserDocumentRepository) or UserDocumentRepositoryImpl(
            session=session, vector_store=self.vector_store
        )

    def get_audit_repository(self, session) -> AuditRepository:
        from app.infrastructure.repositories import AuditRepositoryImpl

        return self._overrides.get(AuditRepository) or AuditRepositoryImpl(session)

    def get_usage_repository(self, session) -> UsageRepository:
        from app.infrastructure.repositories import UsageRepositoryImpl

        return self._overrides.get(UsageRepository) or UsageRepositoryImpl(session)

    def get_human_review_repository(self, session) -> HumanReviewRepository:
        from app.infrastructure.repositories import HumanReviewRepositoryImpl

        return self._overrides.get(HumanReviewRepository) or HumanReviewRepositoryImpl(session)

    # ------------------------------------------------------------------
    # Prompt / Model / Agent / KB loaders
    # ------------------------------------------------------------------
    def get_prompt_service(self, session):
        from app.services.prompt_service import PromptService

        return PromptService(session)

    def get_model_manager(self, session):
        from app.services.model_manager import ModelManager

        return ModelManager(self.llm_factory, session)

    def get_agent_manager(self, session):
        from app.services.agent_manager import AgentManager

        return AgentManager(session, self)

    def get_knowledge_base_manager(self, session):
        from app.services.knowledge_base_manager import KnowledgeBaseManager

        return KnowledgeBaseManager(
            session=session,
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            document_parser=self.document_parser,
        )

    def get_audit_service(self, session):
        from app.audit.audit_service import AuditService

        return AuditService(self.get_audit_repository(session))

    def get_usage_service(self, session):
        from app.audit.usage_service import UsageService

        return UsageService(self.get_usage_repository(session))

    def get_human_review_service(self, session):
        from app.services.human_review_service import HumanReviewService

        return HumanReviewService(self.get_human_review_repository(session))

    # ------------------------------------------------------------------
    # Override support for testing
    # ------------------------------------------------------------------
    def override(self, cls: type, instance: Any) -> None:
        """Override a service for testing."""
        self._overrides[cls] = instance

    def reset(self) -> None:
        """Clear all cached properties and overrides (for testing)."""
        for attr in list(self.__dict__.keys()):
            if not attr.startswith("_"):
                delattr(self, attr)
        self._overrides.clear()


_container: ServiceContainer | None = None


def get_service() -> ServiceContainer:
    """Return the singleton service container."""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


__all__ = ["ServiceContainer", "get_service"]

# Avoid circular import for DocumentMetadata
del DocumentMetadata
