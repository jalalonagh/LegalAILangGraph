"""
SQLAlchemy database models for the Legal AI service metadata.

These models represent the administrative/metadata layer (models, prompts,
agents, knowledge bases, audit, usage, etc.). Legal source documents are
stored in the vector database (Qdrant). PostgreSQL via LangGraph is used
for checkpoint persistence and these metadata tables.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.domain.enums import (
    AuthorityLevel,
    DocumentType,
    EvaluationMetric,
    Jurisdiction,
    LegalDomain,
    LLMProviderType,
    OCRProviderType,
    EmbeddingProviderType,
    RiskLevel,
    RerankerProviderType,
    ReviewDecision,
    SourceStatus,
    ToolRiskLevel,
    VerificationStatus,
    WorkflowType,
)

# =============================================================================
# Base
# =============================================================================

class Base(AsyncAttrs, DeclarativeBase):
    __abstract__ = True

    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now()"),
        onupdate=text("now()"),
        nullable=False,
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


# =============================================================================
# Models - AIModel
# =============================================================================

class AIModel(Base):
    __tablename__ = "ai_models"
    __table_args__ = (
        Index("idx_ai_models_tenant", "tenant_id"),
        Index("idx_ai_models_provider", "provider"),
        UniqueConstraint("tenant_id", "name", "provider", name="uq_ai_model_tenant_name"),
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    api_endpoint: Mapped[str | None] = mapped_column(String(512), default=None)
    api_key_ref: Mapped[str | None] = mapped_column(String(255), default=None)
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    max_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    context_window: Mapped[int | None] = mapped_column(Integer, default=None)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    capabilities: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    def __repr__(self) -> str:
        return f"<AIModel {self.provider}:{self.model_name} (tenant={self.tenant_id})>"


# =============================================================================
# Prompts
# =============================================================================

class Prompt(Base):
    __tablename__ = "prompts"
    __table_args__ = (
        Index("idx_prompts_tenant", "tenant_id"),
        Index("idx_prompts_workflow", "workflow"),
        UniqueConstraint("tenant_id", "name", name="uq_prompt_tenant_name"),
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    workflow: Mapped[str | None] = mapped_column(String(100), default=None)
    agent: Mapped[str | None] = mapped_column(String(100), default=None)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    active_version_id: Mapped[str | None] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("prompt_versions.id"), default=None
    )


class PromptVersion(Base):
    __tablename__ = "prompt_versions"
    __table_args__ = (
        Index("idx_prompt_versions_prompt", "prompt_id"),
        Index("idx_prompt_versions_tenant", "tenant_id"),
    )

    prompt_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("prompts.id"), nullable=False
    )
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="draft")
    created_by: Mapped[str] = mapped_column(String(128), default="system")
    updated_by: Mapped[str] = mapped_column(String(128), default="system")
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)

    def __repr__(self) -> str:
        return f"<PromptVersion v{self.version} for prompt {self.prompt_id}>"


# =============================================================================
# Agents
# =============================================================================

class Agent(Base):
    __tablename__ = "agents"
    __table_args__ = (
        Index("idx_agents_tenant", "tenant_id"),
        UniqueConstraint("tenant_id", "name", name="uq_agent_tenant_name"),
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt_id: Mapped[str | None] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("prompts.id"), default=None
    )
    tool_names: Mapped[list[str]] = mapped_column(JSON, default=list)
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    max_iterations: Mapped[int] = mapped_column(Integer, default=10)
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=300)
    risk_level: Mapped[str] = mapped_column(String(50), default="medium")
    human_review_policy: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)


# =============================================================================
# Agent-Tool association
# =============================================================================

class AgentTool(Base):
    __tablename__ = "agent_tools"
    __table_args__ = (
        Index("idx_agent_tools_tenant", "tenant_id"),
        UniqueConstraint("agent_id", "tool_name", name="uq_agent_tool"),
    )

    agent_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("agents.id"), nullable=False
    )
    tool_name: Mapped[str] = mapped_column(String(255), nullable=False)
    authorization_required: Mapped[bool] = mapped_column(Boolean, default=True)
    risk_level: Mapped[str] = mapped_column(String(50), default="low")
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=60)
    retry_count: Mapped[int] = mapped_column(Integer, default=3)
    audit_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)


# =============================================================================
# Workflows
# =============================================================================

class Workflow(Base):
    __tablename__ = "workflows"
    __table_args__ = (
        Index("idx_workflows_tenant", "tenant_id"),
        UniqueConstraint("tenant_id", "name", name="uq_workflow_tenant_name"),
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    workflow_type: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)


# =============================================================================
# Knowledge Bases & Documents
# =============================================================================

class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    __table_args__ = (
        Index("idx_kb_tenant", "tenant_id"),
        UniqueConstraint("tenant_id", "name", name="uq_kb_tenant_name"),
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    vector_collection: Mapped[str] = mapped_column(String(255), nullable=False)
    jurisdiction: Mapped[str] = mapped_column(String(50), default="us_federal")
    legal_domain: Mapped[str] = mapped_column(String(50), default="general")
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"
    __table_args__ = (
        Index("idx_kdoc_kb", "kb_id"),
        Index("idx_kdoc_tenant", "tenant_id"),
        Index("idx_kdoc_status", "status"),
        UniqueConstraint("kb_id", "external_id", name="uq_kbdoc_kb_external"),
    )

    kb_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("knowledge_bases.id"), nullable=False
    )
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[str] = mapped_column(String(255), default="manual")
    status: Mapped[str] = mapped_column(
        String(50), default="uploaded", nullable=False
    )
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int | None] = mapped_column(Integer, default=None)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_reason: Mapped[str | None] = mapped_column(Text, default=None)
    metadata_: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    kb: Mapped["KnowledgeBase"] = relationship("KnowledgeBase", lazy="selectin")


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        Index("idx_kchunk_kdoc", "document_id"),
        Index("idx_kchunk_tenant", "tenant_id"),
        Index("idx_kchunk_kb", "kb_id"),
        Index("idx_kchunk_vector", "vector_id"),
    )

    kb_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("knowledge_bases.id"), nullable=False
    )
    document_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("knowledge_documents.id"), nullable=False
    )
    vector_id: Mapped[str] = mapped_column(String(128), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    section_id: Mapped[str] = mapped_column(String(128), default="")
    article_number: Mapped[str] = mapped_column(String(64), default="")
    paragraph_number: Mapped[str] = mapped_column(String(64), default="")
    page_number: Mapped[int] = mapped_column(Integer, default=0)
    offset: Mapped[int] = mapped_column(Integer, default=0)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    embedding_preview: Mapped[str] = mapped_column(String(255), default="")
    metadata_: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)


# =============================================================================
# Retrieval & Reranker Configuration
# =============================================================================

class RetrievalConfiguration(Base):
    __tablename__ = "retrieval_configs"
    __table_args__ = (Index("idx_retrieval_tenant", "tenant_id"),)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    kb_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    top_k: Mapped[int] = mapped_column(Integer, default=50)
    vector_weight: Mapped[float] = mapped_column(Float, default=0.7)
    keyword_weight: Mapped[float] = mapped_column(Float, default=0.3)
    use_reranker: Mapped[bool] = mapped_column(Boolean, default=True)
    reranker_id: Mapped[str | None] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("reranker_configs.id"), default=None
    )
    min_score: Mapped[float] = mapped_column(Float, default=0.0)
    filters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)


class RerankerConfiguration(Base):
    __tablename__ = "reranker_configs"
    __table_args__ = (Index("idx_rerank_tenant", "tenant_id"),)

    name: Mapped[str] = mapped_column(String(255), nullable=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    base_url: Mapped[str | None] = mapped_column(String(512), default=None)
    api_key_ref: Mapped[str | None] = mapped_column(String(255), default=None)
    top_k: Mapped[int] = mapped_column(Integer, default=10)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)


# =============================================================================
# Evaluation
# =============================================================================

class EvaluationDataset(Base):
    __tablename__ = "evaluation_datasets"
    __table_args__ = (Index("idx_eval_dataset_tenant", "tenant_id"),)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    workflow_type: Mapped[str] = mapped_column(String(100), nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="en")
    test_cases: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index("idx_eval_run_dataset", "dataset_id"),
        Index("idx_eval_run_tenant", "tenant_id"),
        Index("idx_eval_run_created", "created_at"),
    )

    dataset_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("evaluation_datasets.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    results: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    duration_seconds: Mapped[float | None] = mapped_column(Float, default=None)
    created_by: Mapped[str] = mapped_column(String(128), default="system")
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)


# =============================================================================
# Audit
# =============================================================================

class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (
        Index("idx_audit_tenant", "tenant_id"),
        Index("idx_audit_user", "user_id"),
        Index("idx_audit_run", "run_id"),
        Index("idx_audit_created", "created_at"),
        Index("idx_audit_event_type", "event_type"),
    )

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    case_id: Mapped[str | None] = mapped_column(String(128), default=None)
    conversation_id: Mapped[str | None] = mapped_column(String(128), default=None)
    run_id: Mapped[str | None] = mapped_column(String(128), default=None)
    node_id: Mapped[str | None] = mapped_column(String(128), default=None)
    workflow: Mapped[str | None] = mapped_column(String(100), default=None)
    agent: Mapped[str | None] = mapped_column(String(128), default=None)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str | None] = mapped_column(String(255), default=None)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    input_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    output_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    source_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    human_review: Mapped[bool] = mapped_column(Boolean, default=False)
    result: Mapped[str | None] = mapped_column(String(50), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


# =============================================================================
# Usage
# =============================================================================

class UsageRecord(Base):
    __tablename__ = "usage_records"
    __table_args__ = (
        Index("idx_usage_tenant", "tenant_id"),
        Index("idx_usage_created", "created_at"),
        Index("idx_usage_run", "run_id"),
    )

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(128), default=None)
    workflow: Mapped[str | None] = mapped_column(String(100), default=None)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float | None] = mapped_column(Float, default=None)
    metadata_: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


# =============================================================================
# Human Review
# =============================================================================

class HumanReviewRequest(Base):
    __tablename__ = "human_review_requests"
    __table_args__ = (
        Index("idx_review_run", "run_id"),
        Index("idx_review_tenant", "tenant_id"),
        Index("idx_review_status", "status"),
    )

    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    workflow: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    metadata_: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    decision: Mapped[str | None] = mapped_column(String(50), default=None)
    reviewer_id: Mapped[str | None] = mapped_column(String(128), default=None)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    approved: Mapped[bool | None] = mapped_column(Boolean, default=None)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    resolved_by: Mapped[str | None] = mapped_column(String(128), default=None)


# =============================================================================
# Conversations
# =============================================================================

class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        Index("idx_conv_tenant", "tenant_id"),
        Index("idx_conv_user", "user_id"),
        Index("idx_conv_case", "case_id"),
        UniqueConstraint("tenant_id", "external_id", name="uq_conv_tenant_external"),
    )

    external_id: Mapped[str] = mapped_column(String(128), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False)
    case_id: Mapped[str | None] = mapped_column(String(128), default=None)
    title: Mapped[str] = mapped_column(String(512), default="")
    status: Mapped[str] = mapped_column(String(50), default="active")
    metadata_: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)

    messages: Mapped[list["ConversationMessage"]] = relationship(
        "ConversationMessage", back_populates="conversation", cascade="all, delete-orphan"
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    __table_args__ = (
        Index("idx_msg_conv", "conversation_id"),
        Index("idx_msg_tenant", "tenant_id"),
    )

    conversation_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("conversations.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)

    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")


# =============================================================================
# User Memory (long-term)
# =============================================================================

class UserMemoryEntry(Base):
    __tablename__ = "user_memory_entries"
    __table_args__ = (
        Index("idx_ume_tenant_user", "tenant_id", "user_id"),
        UniqueConstraint("tenant_id", "user_id", "key", name="uq_user_mem_tenant_user_key"),
    )

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(100), default="preference")
    metadata_: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


# =============================================================================
# Case Memory
# =============================================================================

class CaseMemoryEntry(Base):
    __tablename__ = "case_memory_entries"
    __table_args__ = (
        Index("idx_cme_tenant_case", "tenant_id", "case_id"),
        Index("idx_cme_tenant", "tenant_id"),
    )

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    case_id: Mapped[str] = mapped_column(String(128), nullable=False)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(100), default="fact")
    source_type: Mapped[str] = mapped_column(String(100), default="extracted")
    metadata_: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
