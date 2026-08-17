"""
Domain entities - core business objects that represent legal concepts
and workflow states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.domain.enums import IntentType, LegalDomain, RiskLevel, WorkflowType


@dataclass
class LegalIssue:
    """A legal issue identified during analysis."""

    title: str
    description: str = ""
    category: str = "general"
    applicable_rules: list[str] = field(default_factory=list)
    relevant_facts: list[str] = field(default_factory=list)
    confidence: float = 0.0
    supporting_citations: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Structured result of a legal analysis."""

    summary: str = ""
    legal_issues: list[LegalIssue] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    authorities: list[str] = field(default_factory=list)
    arguments: list[str] = field(default_factory=list)
    counterarguments: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    requires_human_review: bool = False
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Result of intent and domain classification."""

    intent: IntentType = IntentType.GENERAL_INQUIRY
    legal_domain: LegalDomain = LegalDomain.GENERAL
    workflow: WorkflowType = WorkflowType.GENERAL
    risk_level: RiskLevel = RiskLevel.LOW
    confidence: float = 0.0
    reasoning: str = ""
    required_tools: list[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Result of verifying an AI-generated legal answer."""

    verified: bool = False
    confidence: float = 0.0
    issues: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    citations_verified: bool = False
    retried: bool = False
    retry_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DraftingResult:
    """Result of a drafting operation."""

    document_id: str
    title: str | None = None
    content: str = ""
    document_type: str = "legal_document"
    confidence: float = 0.0
    requires_human_review: bool = False
    disclaimer: str | None = None
    facts_used: list[str] = field(default_factory=list)
    authorities_used: list[str] = field(default_factory=list)


@dataclass
class DocumentAnalysisResult:
    """Result of a document analysis operation."""

    classification: dict[str, Any] = field(default_factory=dict)
    entities: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    summary: str | None = None
    key_clauses: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    risks: list[dict[str, Any]] = field(default_factory=list)
    page_count: int | None = None
    confidence: float = 0.0


@dataclass
class ContractAnalysisResult:
    """Result of a contract analysis operation."""

    parties: list[dict[str, Any]] = field(default_factory=list)
    obligations: list[dict[str, Any]] = field(default_factory=list)
    rights: list[dict[str, Any]] = field(default_factory=list)
    termination: dict[str, Any] = field(default_factory=dict)
    penalties: list[dict[str, Any]] = field(default_factory=list)
    payment: dict[str, Any] = field(default_factory=dict)
    liability: dict[str, Any] = field(default_factory=dict)
    confidentiality: dict[str, Any] = field(default_factory=dict)
    ip: dict[str, Any] = field(default_factory=dict)
    dispute_resolution: dict[str, Any] = field(default_factory=dict)
    jurisdiction: dict[str, Any] = field(default_factory=dict)
    renewal: dict[str, Any] = field(default_factory=dict)
    force_majeure: dict[str, Any] = field(default_factory=dict)
    missing_clauses: list[str] = field(default_factory=list)
    contradictory_clauses: list[dict[str, Any]] = field(default_factory=list)
    high_risk_clauses: list[dict[str, Any]] = field(default_factory=list)
    summary: str | None = None
    confidence: float = 0.0
    requires_human_review: bool = False


@dataclass
class CaseAnalysisResult:
    """Result of a case analysis operation."""

    parties: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    legal_issues: list[str] = field(default_factory=list)
    applicable_laws: list[dict[str, Any]] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    risk_assessment: dict[str, Any] = field(default_factory=dict)
    possible_arguments: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    requires_human_review: bool = False
    summary: str | None = None


@dataclass
class SearchResult:
    """Result of a retrieval search."""

    chunks: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0
    took_ms: float = 0.0
    retriever_used: str = ""
    confidence: float = 0.0
    query: str = ""


@dataclass
class ToolExecutionResult:
    """Result of a tool execution."""

    tool_name: str
    success: bool
    output: dict[str, Any]
    error: str | None = None
    latency_ms: float = 0.0
    citation_count: int = 0
    confidence: float = 0.0


@dataclass
class Entity:
    """An extracted legal entity."""

    name: str
    entity_type: str
    value: str | None = None
    confidence: float = 0.0
    source: str = "extracted"
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }


@dataclass
class TimelineEvent:
    """A timeline event extracted from documents."""

    event: str
    date: str | None = None
    date_precision: str = "day"  # day, month, year, unknown
    parties: list[str] = field(default_factory=list)
    description: str = ""
    source_document: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "date": self.date,
            "date_precision": self.date_precision,
            "parties": self.parties,
            "description": self.description,
            "source_document": self.source_document,
            "confidence": self.confidence,
        }


__all__ = [
    "LegalIssue",
    "AnalysisResult",
    "ClassificationResult",
    "VerificationResult",
    "DraftingResult",
    "DocumentAnalysisResult",
    "ContractAnalysisResult",
    "CaseAnalysisResult",
    "SearchResult",
    "ToolExecutionResult",
    "Entity",
    "TimelineEvent",
]
