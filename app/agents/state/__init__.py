"""
LangGraph state definitions for the Legal AI platform.

The state is a single Pydantic model attached to the root graph.
Subgraphs operate on slices of this state, and the state includes
all the fields needed for legal reasoning, citation, verification,
and audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langgraph.channels import AnyChannel
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.pregel import Pregel
from pydantic import BaseModel, Field

from app.core.config import settings
from app.domain.value_objects import EvidenceChunk, LegalCitation, Claim


class LegalAIState(BaseModel, frozen=True):
    """
    The canonical LangGraph state for the Legal AI platform.

    All subgraphs share this state model. Fields are optional and default
    to their natural types so that partial states are valid during
    progressive refinement.
    """

    # ---- Request context ----
    request_id: str = ""
    tenant_id: str = "demo"
    user_id: str = "anonymous"
    case_id: str | None = None
    conversation_id: str | None = None
    client_request_id: str | None = None

    # ---- Input ----
    input_messages: list[dict[str, Any]] = Field(default_factory=list)
    question: str = ""
    query_text: str = ""

    # ---- Classification ----
    intent: str = "general"
    legal_domain: str = "general"
    jurisdiction: str = "us_federal"
    risk_level: str = "low"
    workflow: str = "general"
    classification_confidence: float = 0.0
    classification_reasoning: str = ""

    # ---- Context / Documents ----
    document_ids: list[str] = Field(default_factory=list)
    case_facts: list[str] = Field(default_factory=list)
    case_documents: list[dict[str, Any]] = Field(default_factory=list)

    # ---- Retrieval ----
    search_queries: list[str] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    candidate_documents: list[dict[str, Any]] = Field(default_factory=list)
    evidence_set: list[dict[str, Any]] = Field(default_factory=list)

    # ---- Legal reasoning ----
    legal_issues: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    authorities: list[str] = Field(default_factory=list)
    arguments: list[str] = Field(default_factory=list)
    counterarguments: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)

    # ---- Citations ----
    citations: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)

    # ---- Verification ----
    verification: dict[str, Any] = Field(default_factory=dict)
    verification_passed: bool = False
    verification_confidence: float = 0.0
    verification_retry_count: int = 0
    verification_issues: list[str] = Field(default_factory=list)

    # ---- Drafting ----
    draft_content: str = ""
    draft_title: str = ""
    draft_document_id: str = ""
    draft_document_type: str = "legal_document"

    # ---- Document analysis ----
    document_analysis: dict[str, Any] = Field(default_factory=dict)
    extracted_entities: list[dict[str, Any]] = Field(default_factory=list)
    extracted_timeline: list[dict[str, Any]] = Field(default_factory=list)
    document_summary: str = ""
    document_classification: dict[str, Any] = Field(default_factory=dict)
    document_risks: list[dict[str, Any]] = Field(default_factory=list)
    key_clauses: list[dict[str, Any]] = Field(default_factory=list)

    # ---- Contract analysis ----
    contract_analysis: dict[str, Any] = Field(default_factory=dict)
    contract_parties: list[dict[str, Any]] = Field(default_factory=list)
    contract_obligations: list[dict[str, Any]] = Field(default_factory=list)
    contract_clauses: list[dict[str, Any]] = Field(default_factory=list)
    missing_clauses: list[str] = Field(default_factory=list)
    contradictory_clauses: list[dict[str, Any]] = Field(default_factory=list)

    # ---- Case analysis ----
    case_analysis: dict[str, Any] = Field(default_factory=dict)
    case_parties: list[dict[str, Any]] = Field(default_factory=list)
    case_claims: list[str] = Field(default_factory=list)
    case_defenses: list[str] = Field(default_factory=list)
    possible_arguments: list[dict[str, Any]] = Field(default_factory=list)

    # ---- Document comparison ----
    comparison_result: dict[str, Any] = Field(default_factory=dict)
    document_differences: list[dict[str, Any]] = Field(default_factory=list)

    # ---- Output ----
    answer: str = ""
    summary: str = ""
    final_response: dict[str, Any] = Field(default_factory=dict)

    # ---- Execution control ----
    confidence: float = 0.0
    requires_human_review: bool = False
    human_review_request_id: str | None = None
    human_review_approved: bool = False

    # ---- Error handling ----
    error: str | None = None
    error_code: str | None = None
    retry_count: int = 0
    max_retries: int = settings.MAX_VERIFICATION_RETRIES

    # ---- Tool results ----
    tool_results: list[dict[str, Any]] = Field(default_factory=list)

    # ---- Metadata ----
    started_at: datetime | None = None
    completed_at: datetime | None = None
    model_used: str = ""
    models_used: list[str] = Field(default_factory=list)

    # ---- Streaming markers ----
    current_node: str = ""
    current_tool: str = ""

    def update(self, **kwargs: Any) -> "LegalAIState":
        """Return a new state with the given fields updated."""
        data = self.model_dump()
        for k, v in kwargs.items():
            if k in data:
                data[k] = v
        return LegalAIState(**data)


# Convenience: default state factory
def create_initial_state(
    question: str = "",
    tenant_id: str = "demo",
    user_id: str = "anonymous",
    case_id: str | None = None,
    conversation_id: str | None = None,
    request_id: str | None = None,
    **kwargs: Any,
) -> LegalAIState:
    import uuid

    from datetime import timezone

    return LegalAIState(
        request_id=request_id or str(uuid.uuid4()),
        tenant_id=tenant_id,
        user_id=user_id,
        case_id=case_id,
        conversation_id=conversation_id or str(uuid.uuid4()),
        client_request_id=kwargs.get("client_request_id"),
        question=question,
        query_text=question,
        input_messages=kwargs.get("messages", []),
        document_ids=kwargs.get("document_ids", []),
        case_facts=kwargs.get("case_facts", []),
        case_documents=kwargs.get("case_documents", []),
        workflow=kwargs.get("workflow", ""),
        started_at=datetime.now(timezone.utc),
    )


__all__ = ["LegalAIState", "create_initial_state"]
