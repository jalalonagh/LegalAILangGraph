"""
Domain interfaces (abstract base classes) for all external providers and
core services. These define the contracts that infrastructure implementations
must satisfy, following the dependency inversion principle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Protocol

from app.domain.enums import (
    DocumentParseFormat,
    EmbeddingProviderType,
    LLMProviderType,
    RerankerProviderType,
)
from app.domain.value_objects import ChunkMetadata, DocumentMetadata, LegalCitation, EvidenceChunk


# =============================================================================
# LLM Providers
# =============================================================================
class LLMMessage(Protocol):
    """A single message in an LLM conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | list[Any]


class LLMResponse(Protocol):
    """A response from an LLM provider."""

    content: str
    usage: dict[str, int]
    model: str
    finish_reason: str | None
    raw: dict[str, Any]


class LLMProvider(ABC):
    """Abstract interface for LLM providers (Ollama, OpenAI, etc.)."""

    @property
    @abstractmethod
    def provider_type(self) -> LLMProviderType:
        """The type of this LLM provider."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model name for this provider."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a single response from the LLM."""

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the LLM."""

    @abstractmethod
    async def get_embedding(
        self,
        text: str | list[str],
        **kwargs: Any,
    ) -> list[float] | list[list[float]]:
        """Get embeddings for text (some providers support this natively)."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""


class LLMProviderFactory(ABC):
    """Factory for creating LLM providers by type."""

    @abstractmethod
    def create(
        self,
        provider_type: LLMProviderType,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> LLMProvider:
        """Create a provider instance."""


# =============================================================================
# Embedding Providers
# =============================================================================
class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @property
    @abstractmethod
    def provider_type(self) -> EmbeddingProviderType:
        """The type of this embedding provider."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The embedding model name."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of embeddings produced."""

    @abstractmethod
    async def embed_text(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Embed text(s) into vector space."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""


# =============================================================================
# Reranker Providers
# =============================================================================
class RerankResult(Protocol):
    """Result of a reranking operation."""

    index: int
    score: float
    content: str
    metadata: dict[str, Any]


class RerankerProvider(ABC):
    """Abstract interface for reranker providers."""

    @property
    @abstractmethod
    def provider_type(self) -> RerankerProviderType:
        """The type of this reranker provider."""

    @property
    @abstractmethod
    def top_k(self) -> int:
        """Number of top results to return."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[RerankResult]:
        """Rerank candidates based on relevance to query."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""


# =============================================================================
# OCR Providers
# =============================================================================
class OCRProvider(ABC):
    """Abstract interface for OCR providers."""

    @abstractmethod
    async def extract_text(
        self,
        file_path: str,
        language: str = "eng",
        dpi: int = 300,
        **kwargs: Any,
    ) -> str:
        """Extract text from a scanned PDF/image using OCR."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""


# =============================================================================
# Document Parsers
# =============================================================================
class ParsedDocument(Protocol):
    """A parsed document with text content and metadata."""

    text: str
    metadata: dict[str, Any]
    pages: list[str] | None
    tables: list[list[list[str]]] | None


class DocumentParser(ABC):
    """Abstract interface for document parsers."""

    @property
    @abstractmethod
    def supported_formats(self) -> list[DocumentParseFormat]:
        """File formats this parser supports."""

    @abstractmethod
    async def parse(
        self,
        file_path: str,
        mime_type: str,
        fmt: DocumentParseFormat,
        **kwargs: Any,
    ) -> ParsedDocument:
        """Parse a document and extract text + metadata."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""


# =============================================================================
# Vector Store
# =============================================================================
class VectorStore(ABC):
    """Abstract interface for vector databases (Qdrant, etc.)."""

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        **kwargs: Any,
    ) -> bool:
        """Create a vector collection."""

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""

    @abstractmethod
    async def upsert(
        self,
        collection_name: str,
        points: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        """Upsert points into a collection."""

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[EvidenceChunk]:
        """Search for similar vectors."""

    @abstractmethod
    async def search_with_keyword(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[EvidenceChunk]:
        """Hybrid search: vector + keyword."""

    @abstractmethod
    async def delete_points(
        self,
        collection_name: str,
        point_ids: list[str],
    ) -> None:
        """Delete points by ID."""

    @abstractmethod
    async def count(self, collection_name: str, filters: dict[str, Any] | None = None) -> int:
        """Count points in a collection."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""


# =============================================================================
# Repositories
# =============================================================================
class LegalDocumentRepository(ABC):
    """Abstract repository for legal documents (laws, regulations, cases)."""

    @abstractmethod
    async def search_documents(
        self,
        query: str,
        tenant_id: str,
        document_types: list[str] | None = None,
        jurisdiction: str | None = None,
        legal_domain: str | None = None,
        top_k: int = 50,
        **kwargs: Any,
    ) -> list[DocumentMetadata]:
        """Search legal documents by query and filters."""

    @abstractmethod
    async def get_document(self, document_id: str, tenant_id: str) -> DocumentMetadata | None:
        """Get a document by ID."""

    @abstractmethod
    async def get_article(
        self,
        document_id: str,
        article_number: str,
        tenant_id: str,
    ) -> list[ChunkMetadata] | None:
        """Get a specific article from a document."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""


class UserDocumentRepository(ABC):
    """Abstract repository for user-uploaded documents."""

    @abstractmethod
    async def store_document(self, metadata: DocumentMetadata, content: str) -> str:
        """Store a user document and return its ID."""

    @abstractmethod
    async def search_documents(
        self,
        query: str,
        tenant_id: str,
        user_id: str | None = None,
        top_k: int = 50,
        **kwargs: Any,
    ) -> list[DocumentMetadata]:
        """Search user documents."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""


class AuditRepository(ABC):
    """Abstract repository for audit events."""

    @abstractmethod
    async def record_event(
        self,
        event_type: str,
        tenant_id: str,
        user_id: str,
        run_id: str,
        **fields: Any,
    ) -> None:
        """Record an audit event."""

    @abstractmethod
    async def query_events(
        self,
        tenant_id: str | None = None,
        start_time: Any | None = None,
        end_time: Any | None = None,
        event_type: str | None = None,
        limit: int = 1000,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Query audit events."""


class UsageRepository(ABC):
    """Abstract repository for usage metrics."""

    @abstractmethod
    async def record_usage(
        self,
        tenant_id: str,
        user_id: str,
        run_id: str,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float | None = None,
    ) -> None:
        """Record usage metrics."""

    @abstractmethod
    async def get_usage(
        self,
        tenant_id: str | None = None,
        start_time: Any | None = None,
        end_time: Any | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Query usage metrics."""


class HumanReviewRepository(ABC):
    """Abstract repository for human review requests."""

    @abstractmethod
    async def create_request(
        self,
        run_id: str,
        tenant_id: str,
        user_id: str,
        workflow: str,
        metadata: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """Create a human review request. Returns the request ID."""

    @abstractmethod
    async def get_request(self, request_id: str, tenant_id: str) -> dict[str, Any] | None:
        """Get a review request by ID."""

    @abstractmethod
    async def update_decision(
        self,
        request_id: str,
        tenant_id: str,
        decision: str,
        reviewer_id: str,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Update the decision on a review request."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""


__all__ = [
    # LLM
    "LLMMessage",
    "LLMResponse",
    "LLMProvider",
    "LLMProviderFactory",
    # Embeddings
    "EmbeddingProvider",
    # Reranker
    "RerankResult",
    "RerankerProvider",
    # OCR
    "OCRProvider",
    # Document parser
    "ParsedDocument",
    "DocumentParser",
    # Vector store
    "VectorStore",
    # Repositories
    "LegalDocumentRepository",
    "UserDocumentRepository",
    "AuditRepository",
    "UsageRepository",
    "HumanReviewRepository",
]
