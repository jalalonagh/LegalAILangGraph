"""
Pydantic v2 schemas for the Legal AI API.

These models define all request/response contracts. They are intentionally
kept separate from the database models to allow independent evolution
and to avoid leaking sensitive fields to clients.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# Base
# =============================================================================

class BaseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BaseResponseSchema(BaseModel):
    model_config = ConfigDict(extra="allow")


class PaginatedResponse[T: BaseResponseSchema](BaseResponseSchema):
    items: list[T] = Field(default_factory=list)
    total: int = 0
    limit: int = 100
    offset: int = 0


# =============================================================================
# Common sub-schemas
# =============================================================================

class CitationOut(BaseResponseSchema):
    citation_id: str
    document_id: str
    article_number: str | None = None
    section: str | None = None
    title: str | None = None
    source: str | None = None
    source_url: str | None = None
    relevance_score: float | None = None
    authority_level: str | None = None
    jurisdiction: str | None = None
    verified: bool = False
    retrieval_method: str | None = None


class EvidenceChunkOut(BaseResponseSchema):
    chunk_id: str
    document_id: str
    content: str
    score: float
    retrieval_method: str
    authority_level: str
    verified: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimOut(BaseResponseSchema):
    text: str
    kind: str = "fact"
    confidence: float
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    support_score: float = 0.0
    unsupported: bool = False
    contradiction: bool = False
    explanation: str | None = None


class VerificationDetail(BaseResponseSchema):
    verified: bool
    confidence: float
    issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    citations_verified: bool
    retried: bool = False
    retry_count: int = 0


class LegalResponse(BaseResponseSchema):
    answer: str
    summary: str | None = None
    legal_issues: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    authorities: list[str] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    arguments: list[str] = Field(default_factory=list)
    counterarguments: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    requires_human_review: bool = False
    verification: VerificationDetail | None = None
    evidence: list[EvidenceChunkOut] = Field(default_factory=list)
    claims: list[ClaimOut] = Field(default_factory=list)
    run_id: str | None = None


class ErrorResponse(BaseResponseSchema):
    error: dict[str, Any]


class HealthResponse(BaseResponseSchema):
    status: str = "healthy"
    timestamp: datetime
    version: str
    checks: dict[str, str] = Field(default_factory=dict)


# =============================================================================
# Chat
# =============================================================================

class ChatMessage(BaseSchema):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str
    name: str | None = None


class ChatRequest(BaseSchema):
    messages: list[ChatMessage]
    conversation_id: str | None = None
    case_id: str | None = None
    tenant_id: str | None = None
    workflow: str | None = None
    tools: list[str] | None = None
    temperature: float | None = None
    max_iterations: int | None = None
    enable_streaming: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseResponseSchema):
    answer: str
    conversation_id: str
    run_id: str
    requires_human_review: bool = False
    human_review_request_id: str | None = None
    legal_response: LegalResponse | None = None


class StreamEvent(BaseSchema):
    event: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


# =============================================================================
# Legal Research
# =============================================================================

class ResearchRequest(BaseSchema):
    question: str
    jurisdiction: str = "us_federal"
    legal_domains: list[str] | None = None
    max_sources: int = 50
    include_user_documents: bool = True
    include_web: bool = False
    temperature: float | None = None
    enable_streaming: bool = False
    case_id: str | None = None
    tenant_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchResponse(BaseResponseSchema):
    legal_response: LegalResponse


# =============================================================================
# Document Analysis
# =============================================================================

class DocumentAnalysisRequest(BaseSchema):
    document_text: str | None = None
    document_url: str | None = None
    mime_type: str | None = None
    filename: str | None = None
    extract_entities: bool = True
    extract_timeline: bool = True
    extract_citations: bool = True
    summarize: bool = True
    classify: bool = True
    risk_detection: bool = True
    tenant_id: str | None = None
    case_id: str | None = None
    enable_streaming: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentAnalysisResponse(BaseResponseSchema):
    classification: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    entities: list[dict[str, Any]] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    summary: str | None = None
    key_clauses: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    risks: list[dict[str, Any]] = Field(default_factory=list)
    run_id: str | None = None
    confidence: float = 0.0


# =============================================================================
# Contract Analysis
# =============================================================================

class ContractAnalysisRequest(BaseSchema):
    contract_text: str | None = None
    document_url: str | None = None
    mime_type: str | None = None
    filename: str | None = None
    tenant_id: str | None = None
    case_id: str | None = None
    extract_parties: bool = True
    extract_obligations: bool = True
    extract_rights: bool = True
    check_termination: bool = True
    check_penalties: bool = True
    check_liability: bool = True
    check_confidentiality: bool = True
    check_ip: bool = True
    check_dispute_resolution: bool = True
    check_jurisdiction: bool = True
    check_renewal: bool = True
    check_force_majeure: bool = True
    check_missing_clauses: bool = True
    detect_contradictions: bool = True
    high_risk_clauses: bool = True
    enable_streaming: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContractClause(BaseResponseSchema):
    name: str
    text: str
    risk_level: str = "low"
    present: bool = True
    explanation: str | None = None
    recommendation: str | None = None


class ContractAnalysisResponse(BaseResponseSchema):
    parties: list[dict[str, Any]] = Field(default_factory=list)
    obligations: list[dict[str, Any]] = Field(default_factory=list)
    rights: list[dict[str, Any]] = Field(default_factory=list)
    termination: dict[str, Any] = Field(default_factory=dict)
    penalties: list[dict[str, Any]] = Field(default_factory=list)
    payment: dict[str, Any] = Field(default_factory=dict)
    liability: dict[str, Any] = Field(default_factory=dict)
    confidentiality: dict[str, Any] = Field(default_factory=dict)
    ip: dict[str, Any] = Field(default_factory=dict)
    dispute_resolution: dict[str, Any] = Field(default_factory=dict)
    jurisdiction: dict[str, Any] = Field(default_factory=dict)
    renewal: dict[str, Any] = Field(default_factory=dict)
    force_majeure: dict[str, Any] = Field(default_factory=dict)
    missing_clauses: list[str] = Field(default_factory=list)
    contradictory_clauses: list[dict[str, Any]] = Field(default_factory=list)
    high_risk_clauses: list[ContractClause] = Field(default_factory=list)
    all_clauses: list[ContractClause] = Field(default_factory=list)
    summary: str | None = None
    confidence: float = 0.0
    requires_human_review: bool = False
    run_id: str | None = None


# =============================================================================
# Case Analysis
# =============================================================================

class CaseAnalysisRequest(BaseSchema):
    case_facts: str | None = None
    case_documents: list[dict[str, Any]] | None = None
    claims: list[str] | None = None
    defenses: list[str] | None = None
    relevant_laws: list[str] | None = None
    jurisdiction: str = "us_federal"
    enable_streaming: bool = False
    tenant_id: str | None = None
    case_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseAnalysisResponse(BaseResponseSchema):
    parties: list[dict[str, Any]] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    legal_issues: list[str] = Field(default_factory=list)
    applicable_laws: list[dict[str, Any]] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    risk_assessment: dict[str, Any] = Field(default_factory=dict)
    possible_arguments: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    requires_human_review: bool = False
    run_id: str | None = None
    summary: str | None = None


# =============================================================================
# Drafting
# =============================================================================

class DraftingRequest(BaseSchema):
    document_type: str = "legal_letter"
    title: str | None = None
    prompt_content: str
    facts: list[str] | None = None
    authorities: list[str] | None = None
    template: str | None = None
    tone: str = "formal"
    jurisdiction: str = "us_federal"
    enable_streaming: bool = False
    tenant_id: str | None = None
    case_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DraftingResponse(BaseResponseSchema):
    document_id: str
    title: str | None = None
    content: str
    document_type: str
    confidence: float = 0.0
    requires_human_review: bool = False
    run_id: str | None = None
    disclaimer: str | None = None


# =============================================================================
# Citation Verification
# =============================================================================

class CitationVerifyRequest(BaseSchema):
    citations: list[dict[str, Any]]
    verify_existence: bool = True
    verify_relevance: bool = True
    verify_authority: bool = True
    include_jurisdiction_check: bool = True
    tenant_id: str | None = None


class CitationVerifyResponse(BaseResponseSchema):
    verified: bool
    confidence: float
    issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    citations_verified: bool
    details: list[dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# Retrieval Search
# =============================================================================

class SearchRequest(BaseSchema):
    query: str
    knowledge_base_ids: list[str] | None = None
    retriever: str | None = None
    top_k: int = 10
    filters: dict[str, Any] | None = None
    rerank: bool = True
    include_user_docs: bool = True
    tenant_id: str | None = None
    case_id: str | None = None


class SearchResult(BaseResponseSchema):
    chunks: list[EvidenceChunkOut]
    query: str
    total: int
    took_ms: float
    retriever_used: str
    confidence: float = 0.0


# =============================================================================
# Document Comparison
# =============================================================================

class CompareDocumentsRequest(BaseSchema):
    document_a: str
    document_b: str
    document_a_id: str | None = None
    document_b_id: str | None = None
    focus_areas: list[str] | None = None
    tenant_id: str | None = None
    enable_streaming: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompareDocumentsResponse(BaseResponseSchema):
    run_id: str | None = None
    comparison: str
    differences: list[dict[str, Any]] = Field(default_factory=list)
    similarities: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    summary: str | None = None


# =============================================================================
# Run Management
# =============================================================================

class RunStateResponse(BaseResponseSchema):
    run_id: str
    thread_id: str
    status: str
    created_at: datetime
    updated_at: datetime | None = None
    current_node: str | None = None
    next_node: str | None = None
    state: dict[str, Any] | None = None
    values: dict[str, Any] | None = None
    interrupts: list[dict[str, Any]] = Field(default_factory=list)


class ResumeRequest(BaseSchema):
    input: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Human Review
# =============================================================================

class HumanReviewResponse(BaseResponseSchema):
    request_id: str
    status: str
    run_id: str
    workflow: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class HumanReviewDecision(BaseSchema):
    decision: str = "approve"
    notes: str | None = None
    reviewer_id: str | None = None


# =============================================================================
# Admin: Models
# =============================================================================

class ModelCreate(BaseSchema):
    name: str
    provider: str
    model_name: str
    api_endpoint: str | None = None
    api_key_ref: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    context_window: int | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)


class ModelOut(BaseResponseSchema):
    id: str
    name: str
    provider: str
    model_name: str
    api_endpoint: str | None = None
    api_key_ref: str | None = None
    temperature: float
    max_tokens: int | None = None
    context_window: int | None = None
    enabled: bool
    is_default: bool
    capabilities: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Admin: Prompts
# =============================================================================

class PromptCreate(BaseSchema):
    name: str
    description: str = ""
    workflow: str | None = None
    agent: str | None = None


class PromptVersionCreate(BaseSchema):
    content: str
    version: str
    created_by: str = "system"
    status: str = "active"


class PromptOut(BaseResponseSchema):
    id: str
    name: str
    description: str
    workflow: str | None = None
    agent: str | None = None
    active_version_id: str | None = None
    versions: list["PromptVersionOut"] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class PromptVersionOut(BaseResponseSchema):
    id: str
    prompt_id: str
    version: str
    content: str
    status: str
    created_by: str
    is_active: bool
    created_at: datetime


# =============================================================================
# Admin: Knowledge Base
# =============================================================================

class KnowledgeBaseCreate(BaseSchema):
    name: str
    description: str = ""
    vector_collection: str
    jurisdiction: str = "us_federal"
    legal_domain: str = "general"
    embedding_model: str
    embedding_dim: int = 1024


class KnowledgeBaseOut(BaseResponseSchema):
    id: str
    name: str
    description: str
    vector_collection: str
    jurisdiction: str
    legal_domain: str
    embedding_model: str
    embedding_dim: int
    enabled: bool
    document_count: int = 0
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


class DocumentUploadResponse(BaseResponseSchema):
    document_id: str
    status: str
    message: str
    chunk_count: int = 0


# =============================================================================
# Admin: Workflow, Agent, Tool, Retriever, Reranker
# =============================================================================

class WorkflowOut(BaseResponseSchema):
    id: str
    name: str
    workflow_type: str
    description: str
    config: dict[str, Any]
    enabled: bool
    created_at: datetime
    updated_at: datetime


class WorkflowCreate(BaseSchema):
    name: str
    workflow_type: str
    description: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class AgentOut(BaseResponseSchema):
    id: str
    name: str
    description: str
    enabled: bool
    model_name: str
    tool_names: list[str]
    temperature: float
    max_iterations: int
    timeout_seconds: int
    risk_level: str
    created_at: datetime
    updated_at: datetime


class ToolOut(BaseResponseSchema):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    authorization_required: bool
    risk_level: str
    timeout_seconds: int
    retry_count: int


class RetrieverOut(BaseResponseSchema):
    id: str
    name: str
    config: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class RerankerOut(BaseResponseSchema):
    id: str
    name: str
    provider: str
    model: str
    base_url: str | None = None
    top_k: int
    created_at: datetime
    updated_at: datetime


class AuditEventOut(BaseResponseSchema):
    id: str
    tenant_id: str
    user_id: str
    event_type: str
    workflow: str | None = None
    agent: str | None = None
    run_id: str | None = None
    model: str | None = None
    confidence: float | None = None
    result: str | None = None
    error: str | None = None
    created_at: datetime


class UsageSummary(BaseResponseSchema):
    period: str
    tenant_id: str
    total_requests: int
    total_tokens: int
    total_cost_usd: float
    models_used: list[str]
    workflows: dict[str, int]


# =============================================================================
# Health
# =============================================================================

class DependencyHealth(BaseResponseSchema):
    name: str
    status: str
    detail: str | None = None


__all__ = [
    # Base
    "BaseSchema", "BaseResponseSchema", "PaginatedResponse",
    # Common
    "CitationOut", "EvidenceChunkOut", "ClaimOut", "VerificationDetail",
    "LegalResponse", "ErrorResponse", "HealthResponse", "DependencyHealth",
    # Chat
    "ChatMessage", "ChatRequest", "ChatResponse", "StreamEvent",
    # Research
    "ResearchRequest", "ResearchResponse",
    # Document
    "DocumentAnalysisRequest", "DocumentAnalysisResponse",
    # Contract
    "ContractAnalysisRequest", "ContractAnalysisResponse", "ContractClause",
    # Case
    "CaseAnalysisRequest", "CaseAnalysisResponse",
    # Drafting
    "DraftingRequest", "DraftingResponse",
    # Citations
    "CitationVerifyRequest", "CitationVerifyResponse",
    # Search
    "SearchRequest", "SearchResult",
    # Compare
    "CompareDocumentsRequest", "CompareDocumentsResponse",
    # Run
    "RunStateResponse", "ResumeRequest",
    # Human review
    "HumanReviewResponse", "HumanReviewDecision",
    # Admin
    "ModelCreate", "ModelOut", "PromptCreate", "PromptVersionCreate",
    "PromptOut", "PromptVersionOut", "KnowledgeBaseCreate",
    "KnowledgeBaseOut", "DocumentUploadResponse", "WorkflowOut",
    "WorkflowCreate", "AgentOut", "ToolOut", "RetrieverOut",
    "RerankerOut", "AuditEventOut", "UsageSummary",
]
