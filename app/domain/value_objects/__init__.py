"""Domain value objects for the Legal AI platform."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path

from app.domain.enums import (
    AuthorityLevel,
    DocumentParseFormat,
    DocumentType,
    Jurisdiction,
    LegalDomain,
    RiskLevel,
    SourceStatus,
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TenantId:
    value: str

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(("tenant", self.value))


@dataclass(frozen=True)
class UserId:
    value: str

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class CaseId:
    value: str

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class DocumentId:
    value: str

    def __str__(self) -> str:
        return self.value

    @classmethod
    def generate(cls) -> "DocumentId":
        return cls(str(uuid.uuid4()))


@dataclass(frozen=True)
class ChunkId:
    value: str

    def __str__(self) -> str:
        return self.value

    @classmethod
    def generate(cls) -> "ChunkId":
        return cls(str(uuid.uuid4()))


@dataclass(frozen=True)
class CitationId:
    value: str

    def __str__(self) -> str:
        return self.value

    @classmethod
    def generate(cls) -> "CitationId":
        return cls(f"cit_{uuid.uuid4().hex[:12]}")


@dataclass(frozen=True)
class ArticleNumber:
    value: str

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class LegalJurisdiction:
    """Normalized jurisdiction key, e.g. 'us_ca' for California, 'eu' for EU."""

    jurisdiction: Jurisdiction = Jurisdiction.GENERIC
    country_code: str = ""
    region_code: str = ""
    language: str = "en"

    @classmethod
    def from_code(cls, code: str, language: str = "en") -> "LegalJurisdiction":
        """Parse a jurisdiction code like 'us-ca' or 'eu-en'."""
        parts = code.replace("-", "_").split("_")
        if len(parts) == 2:
            country, region = parts
        elif len(parts) == 1:
            country, region = parts[0], ""
        else:
            country, region = parts[0], "_".join(parts[1:])

        mapping = {
            "us": {
                "federal": Jurisdiction.US_FEDERAL,
                "ca": Jurisdiction.US_CA,
                "ny": Jurisdiction.US_NY,
            },
            "eu": {"": Jurisdiction.EU},
            "uk": {"": Jurisdiction.UK},
        }
        country_map = mapping.get(country.lower(), {})
        juris = country_map.get(region.lower(), Jurisdiction.GENERIC)
        return cls(
            jurisdiction=juris,
            country_code=country.lower(),
            region_code=region.lower(),
            language=language,
        )

    def to_code(self) -> str:
        parts = [self.country_code]
        if self.region_code:
            parts.append(self.region_code)
        return "-".join(parts)


@dataclass(frozen=True)
class DocumentMetadata:
    """Normalized metadata for a legal document."""

    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    document_type: DocumentType = DocumentType.USER_DOCUMENT
    authority: str = ""
    jurisdiction: str = ""
    publication_date: datetime | None = None
    effective_date: datetime | None = None
    expiration_date: datetime | None = None
    status: SourceStatus = SourceStatus.CURRENT
    source_url: str = ""
    source_identifier: str = ""
    version: str = "1.0"
    language: str = "en"
    legal_domain: LegalDomain = LegalDomain.GENERAL
    authority_level: AuthorityLevel = AuthorityLevel.LEVEL_5_SECONDARY_SOURCES
    file_path: str = ""
    file_name: str = ""
    file_size_bytes: int = 0
    mime_type: str = ""
    page_count: int | None = None
    tenant_id: str = ""
    external_document_id: str = ""

    def to_json(self) -> dict[str, str | int | float | bool | None]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "document_type": self.document_type.value,
            "authority": self.authority,
            "jurisdiction": self.jurisdiction,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "status": self.status.value,
            "source_url": self.source_url,
            "source_identifier": self.source_identifier,
            "version": self.version,
            "language": self.language,
            "legal_domain": self.legal_domain.value,
            "authority_level": self.authority_level.value,
            "file_name": self.file_name,
            "file_size_bytes": self.file_size_bytes,
            "mime_type": self.mime_type,
            "page_count": self.page_count,
            "tenant_id": self.tenant_id,
            "external_document_id": self.external_document_id,
        }


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata retained with each chunk for high-quality citations."""

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    section_id: str = ""
    article_number: str = ""
    paragraph_number: str = ""
    source: str = ""
    version: str = ""
    page: int = 0
    offset: int = 0
    boundary: str = "paragraph"
    content: str = ""

    def to_json(self) -> dict[str, str | int]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "section_id": self.section_id,
            "article_number": self.article_number,
            "paragraph_number": self.paragraph_number,
            "source": self.source,
            "version": self.version,
            "page": self.page,
            "offset": self.offset,
            "boundary": self.boundary,
        }


@dataclass(frozen=True)
class EvidenceChunk:
    """A single retrieved chunk of evidence with provenance."""

    chunk_id: str
    document_id: str
    content: str
    metadata: ChunkMetadata
    score: float = 0.0
    retrieval_method: str = "vector"
    authority_level: AuthorityLevel = AuthorityLevel.LEVEL_5_SECONDARY_SOURCES
    verified: bool = False

    def to_json(self) -> dict[str, str | float | bool | dict]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content_snippet": self.content[:250],
            "score": self.score,
            "retrieval_method": self.retrieval_method,
            "authority_level": self.authority_level.value,
            "verified": self.verified,
            "metadata": self.metadata.to_json(),
        }


@dataclass(frozen=True)
class LegalCitation:
    """Stable, first-class legal citation."""

    citation_id: str = field(default_factory=lambda: f"cit_{uuid.uuid4().hex[:12]}")
    document_id: str = ""
    article_number: str = ""
    section: str = ""
    title: str = ""
    source: str = ""
    source_url: str = ""
    relevance_score: float = 0.0
    authority_level: AuthorityLevel = AuthorityLevel.LEVEL_5_SECONDARY_SOURCES
    jurisdiction: str = ""
    publication_date: datetime | None = None
    effective_date: datetime | None = None
    status: SourceStatus = SourceStatus.CURRENT
    verified: bool = False
    retrieval_method: str = "vector"

    def to_json(self) -> dict[str, str | float | bool | None]:
        return {
            "citation_id": self.citation_id,
            "document_id": self.document_id,
            "article_number": self.article_number,
            "section": self.section,
            "title": self.title,
            "source": self.source,
            "source_url": self.source_url,
            "relevance_score": self.relevance_score,
            "authority_level": self.authority_level.value,
            "jurisdiction": self.jurisdiction,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "status": self.status.value,
            "verified": self.verified,
            "retrieval_method": self.retrieval_method,
        }


@dataclass(frozen=True)
class Claim:
    """A factual/legal claim that maps to evidence and citations."""

    text: str
    kind: str = "fact"  # fact, legal_authority, inference, assumption, recommendation
    confidence: float = 0.0
    evidence_chunk_ids: list[str] = field(default_factory=list)
    citation_ids: list[str] = field(default_factory=list)
    support_score: float = 0.0
    unsupported: bool = False
    contradiction: bool = False
    explanation: str = ""

    def to_json(self) -> dict[str, str | float | list[str] | bool]:
        return {
            "text": self.text,
            "kind": self.kind,
            "confidence": self.confidence,
            "evidence_chunk_ids": self.evidence_chunk_ids,
            "citation_ids": self.citation_ids,
            "support_score": self.support_score,
            "unsupported": self.unsupported,
            "contradiction": self.contradiction,
            "explanation": self.explanation,
        }


def classify_file_format(filename: str) -> DocumentParseFormat | None:
    """Classify a file by its extension, returning the parse format or None."""
    ext = Path(filename).suffix.lower()
    mapping = {
        ".pdf": DocumentParseFormat.PDF,
        ".docx": DocumentParseFormat.DOCX,
        ".doc": DocumentParseFormat.DOC,
        ".txt": DocumentParseFormat.TXT,
        ".html": DocumentParseFormat.HTML,
        ".htm": DocumentParseFormat.HTML,
        ".rtf": DocumentParseFormat.RTF,
        ".xlsx": DocumentParseFormat.XLSX,
        ".xls": DocumentParseFormat.XLS,
        ".pptx": DocumentParseFormat.PPTX,
        ".csv": DocumentParseFormat.CSV,
    }
    return mapping.get(ext)
