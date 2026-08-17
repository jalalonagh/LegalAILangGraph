"""
Domain enums for the Legal AI platform.
"""

from __future__ import annotations

import enum


class WorkflowType(str, enum.Enum):
    LEGAL_QA = "legal_qa"
    LEGAL_RESEARCH = "legal_research"
    DOCUMENT_ANALYSIS = "document_analysis"
    CONTRACT_ANALYSIS = "contract_analysis"
    CASE_ANALYSIS = "case_analysis"
    LEGAL_DRAFTING = "legal_drafting"
    DOCUMENT_COMPARISON = "document_comparison"
    SUMMARIZATION = "summarization"
    GENERAL = "general"


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IntentType(str, enum.Enum):
    QUESTION_ANSWERING = "question_answering"
    RESEARCH = "research"
    DOCUMENT_ANALYSIS = "document_analysis"
    CONTRACT_REVIEW = "contract_review"
    CASE_REVIEW = "case_review"
    DRAFTING = "drafting"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    GENERAL_INQUIRY = "general_inquiry"


class DocumentType(str, enum.Enum):
    LAW = "law"
    REGULATION = "regulation"
    COURT_DECISION = "court_decision"
    LEGAL_OPINION = "legal_opinion"
    CONTRACT = "contract"
    MEMORANDUM = "memorandum"
    LETTER = "letter"
    NOTICE = "notice"
    MOTION = "motion"
    BRIEF = "brief"
    STATUTE = "statute"
    CASE_FILE = "case_file"
    SECONDARY_SOURCE = "secondary_source"
    USER_DOCUMENT = "user_document"


class AuthorityLevel(str, enum.Enum):
    LEVEL_1_OFFICIAL_LEGISLATION = "level_1_official_legislation"
    LEVEL_2_OFFICIAL_COURT_DECISIONS = "level_2_official_court_decisions"
    LEVEL_3_OFFICIAL_LEGAL_OPINIONS = "level_3_official_legal_opinions"
    LEVEL_4_TRUSTED_DATABASES = "level_4_trusted_databases"
    LEVEL_5_SECONDARY_SOURCES = "level_5_secondary_sources"
    LEVEL_6_GENERAL_WEB = "level_6_general_web"


class SourceStatus(str, enum.Enum):
    CURRENT = "current"
    REPEALED = "repealed"
    EXPIRED = "expired"
    DRAFT = "draft"
    PROPOSED = "proposed"
    CONTESTED = "contested"


class LegalDomain(str, enum.Enum):
    GENERAL = "general"
    CONTRACT = "contract"
    CORPORATE = "corporate"
    EMPLOYMENT = "employment"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    REAL_ESTATE = "real_estate"
    TAX = "tax"
    IMMIGRATION = "immigration"
    CRIMINAL = "criminal"
    FAMILY = "family"
    PERSONAL_INJURY = "personal_injury"
    HEALTHCARE = "healthcare"
    ANTITRUST = "antitrust"
    BANKRUPTCY = "bankruptcy"
    INTERNATIONAL = "international"
    EU = "eu"
    INTERNATIONAL_ARBITRATION = "international_arbitration"


class Jurisdiction(str, enum.Enum):
    US_FEDERAL = "us_federal"
    US_CA = "us_ca"  # California
    US_NY = "us_ny"  # New York
    EU = "eu"
    UK = "uk"
    GENERIC = "generic"


class ToolRiskLevel(str, enum.Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryType(str, enum.Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    CASE_MEMORY = "case_memory"
    USER_MEMORY = "user_memory"


class VerificationStatus(str, enum.Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    RETRY_SCHEDULED = "retry_scheduled"


class ReviewDecision(str, enum.Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_REVISION = "request_revision"
    ESCALATE = "escalate"


class EventSource(str, enum.Enum):
    SYSTEM = "system"
    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    LLM = "llm"
    RECOVERY = "recovery"
    EXTERNAL = "external"


class EventType(str, enum.Enum):
    RUN_STARTED = "run_started"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    TOKEN = "token"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    CITATION_FOUND = "citation_found"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    RUN_COMPLETED = "run_completed"
    ERROR = "error"


class CacheStrategy(str, enum.Enum):
    NONE = "none"
    SEMANTIC = "semantic"
    EXACT = "exact"
    HYBRID = "hybrid"


class RerankerProviderType(str, enum.Enum):
    TEI = "tei"
    COHERE = "cohere"
    JINA = "jina"
    BGE = "bge"
    NONE = "none"


class LLMProviderType(str, enum.Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class EmbeddingProviderType(str, enum.Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    OLLAMA = "ollama"
    TFIDF = "tfidf"


class OCRProviderType(str, enum.Enum):
    TESSERACT = "tesseract"
    OCRMYPDF = "ocrmypdf"
    NONE = "none"


class DocumentParseFormat(str, enum.Enum):
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"
    HTML = "html"
    RTF = "rtf"
    XLSX = "xlsx"
    XLS = "xls"
    PPTX = "pptx"
    CSV = "csv"


class ChunkBoundary(str, enum.Enum):
    """Preferred boundaries for legal-aware chunking."""
    DOCUMENT = "document"
    SECTION = "section"
    CHAPTER = "chapter"
    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    CLAUSE = "clause"
    SUBCLAUSE = "subclause"


class EvaluationMetric(str, enum.Enum):
    RETRIEVAL_RECALL = "retrieval_recall"
    RETRIEVAL_PRECISION = "retrieval_precision"
    CITATION_ACCURACY = "citation_accuracy"
    CITATION_COVERAGE = "citation_coverage"
    ANSWER_FAITHFULNESS = "answer_faithfulness"
    HALLUCINATION_RATE = "hallucination_rate"
    LATENCY = "latency"
    TOKEN_COST = "token_cost"
    TOOL_SUCCESS_RATE = "tool_success_rate"
    WORKFLOW_SUCCESS_RATE = "workflow_success_rate"
    CONFIDENCE_CALIBRATION = "confidence_calibration"


class AuditAction(str, enum.Enum):
    REQUEST = "request"
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    REVIEW_APPROVE = "review_approve"
    REVIEW_REJECT = "review_reject"
    VERIFY = "verify"
    RETRIEVE = "retrieve"
    INGEST = "ingest"
    CONFIG_CHANGE = "config_change"
    RATE_LIMIT = "rate_limit"
