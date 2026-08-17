"""
Custom exception hierarchy for the Legal AI platform.

All application errors derive from :class:`LegalAIError`. Each exception
carries a machine-readable ``code``, a human-readable ``message``, and
an optional ``details`` dict. Sensitive internal details are never
leaked to API clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LegalAIError(Exception):
    """Base exception for all application errors."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    status_code: int = 500

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, str | dict[str, Any]]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class ConfigError(LegalAIError):
    """Raised on configuration or environment errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(code="CONFIG_ERROR", message=message, details=details or {}, status_code=500)


class AuthenticationError(LegalAIError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed", details: dict[str, Any] | None = None) -> None:
        super().__init__(code="AUTH_FAILED", message=message, details=details or {}, status_code=401)


class AuthorizationError(LegalAIError):
    """Raised when an authenticated user lacks permission."""

    def __init__(self, message: str = "Insufficient permissions", details: dict[str, Any] | None = None) -> None:
        super().__init__(code="FORBIDDEN", message=message, details=details or {}, status_code=403)


class TenantIsolationError(LegalAIError):
    """Raised when a tenant accesses another tenant's data."""

    def __init__(self, message: str = "Tenant isolation violation", details: dict[str, Any] | None = None) -> None:
        super().__init__(code="TENANT_ISOLATION_VIOLATION", message=message, details=details or {}, status_code=403)


class NotFoundError(LegalAIError):
    """Raised when a requested resource is not found."""

    def __init__(self, message: str = "Resource not found", details: dict[str, Any] | None = None) -> None:
        super().__init__(code="NOT_FOUND", message=message, details=details or {}, status_code=404)


class ValidationError(LegalAIError):
    """Raised on input validation errors."""

    def __init__(self, message: str = "Validation error", details: dict[str, Any] | None = None) -> None:
        super().__init__(code="VALIDATION_ERROR", message=message, details=details or {}, status_code=422)


class RateLimitExceededError(LegalAIError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", details: dict[str, Any] | None = None) -> None:
        super().__init__(code="RATE_LIMIT_EXCEEDED", message=message, details=details or {}, status_code=429)


class FileSizeExceededError(LegalAIError):
    """Raised when an uploaded file exceeds the size limit."""

    def __init__(self, message: str = "File size exceeded", details: dict[str, Any] | None = None) -> None:
        super().__init__(code="FILE_TOO_LARGE", message=message, details=details or {}, status_code=413)


class UnsupportedFileTypeError(LegalAIError):
    """Raised when an uploaded file type is not supported."""

    def __init__(self, message: str = "Unsupported file type", details: dict[str, Any] | None = None) -> None:
        super().__init__(code="UNSUPPORTED_FILE_TYPE", message=message, details=details or {}, status_code=415)


class LLMProviderError(LegalAIError):
    """Raised when an LLM provider call fails."""

    def __init__(self, message: str = "LLM provider error", details: dict[str, Any] | None = None) -> None:
        super().__init__(code="LLM_PROVIDER_ERROR", message=message, details=details or {}, status_code=502)


class VectorStoreError(LegalAIError):
    """Raised when vector store operations fail."""

    def __init__(self, message: str = "Vector store error", details: dict[str, Any] | None = None) -> None:
        super().__init__(code="VECTOR_STORE_ERROR", message=message, details=details or {}, status_code=502)


class WorkflowError(LegalAIError):
    """Raised when a workflow execution fails."""

    def __init__(self, message: str = "Workflow execution error", details: dict[str, Any] | None = None) -> None:
        super().__init__(code="WORKFLOW_ERROR", message=message, details=details or {}, status_code=500)


class VerificationError(LegalAIError):
    """Raised when legal verification fails."""

    def __init__(self, message: str = "Verification failed", details: dict[str, Any] | None = None) -> None:
        super().__init__(code="VERIFICATION_FAILED", message=message, details=details or {}, status_code=500)


class CancellationError(LegalAIError):
    """Raised when a workflow run is cancelled."""

    def __init__(self, message: str = "Operation cancelled", details: dict[str, Any] | None = None) -> None:
        super().__init__(code="CANCELLED", message=message, details=details or {}, status_code=499)


class PromptInjectionError(LegalAIError):
    """Raised when prompt injection is detected."""

    def __init__(self, message: str = "Prompt injection detected", details: dict[str, Any] | None = None) -> None:
        super().__init__(code="PROMPT_INJECTION_DETECTED", message=message, details=details or {}, status_code=400)
