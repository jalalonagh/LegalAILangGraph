"""
Core configuration for the Legal AI Assistant platform.

All configuration is loaded from environment variables (with optional .env file)
and validated using Pydantic v2 settings. No hard-coded secrets.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic.networks import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- Application ---
    APP_ENV: str = Field(default="development")
    APP_DEBUG: bool = Field(default=False)
    PROJECT_NAME: str = Field(default="Legal AI Assistant")
    VERSION: str = Field(default="0.1.0")
    RELOAD: bool = Field(default=False)
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_V1_PREFIX: str = Field(default="/api/v1")
    RATE_LIMIT_ENABLED: bool = Field(default=True)
    RATE_LIMIT_DEFAULT: str = Field(default="100/minute")
    MAX_REQUEST_SIZE_MB: int = Field(default=100)
    MAX_UPLOAD_SIZE_MB: int = Field(default=50)

    # --- Authentication ---
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_AUDIENCE: str = Field(default="legal-ai-internal")
    JWT_ISSUER: str = Field(default="legal-ai")
    JWT_PUBLIC_KEY_PATH: str | None = Field(default=None)
    JWT_PRIVATE_KEY_PATH: str | None = Field(default=None)
    JWT_SECRET_KEY: SecretStr = Field(default=SecretStr("change-me-in-production"))
    JWT_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    INTERNAL_API_KEY: SecretStr = Field(default=SecretStr("change-me-internal-api-key"))
    TRUSTED_SERVICE_URLS: str = Field(default="http://localhost:8000")

    # --- Database ---
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://legalai:legalai@postgres:5432/legalai"
    )

    # --- Redis ---
    REDIS_URL: str = Field(default="redis://redis:6379/0")
    REDIS_CACHE_TTL_SECONDS: int = Field(default=3600)
    CACHE_ENABLED: bool = Field(default=True)

    # --- Vector DB ---
    VECTOR_DB_URL: str = Field(default="http://qdrant:6333")
    VECTOR_DB_GRPC_PORT: int = Field(default=6334)
    VECTOR_DB_API_KEY: SecretStr | None = Field(default=None)
    VECTOR_DB_ENABLED: bool = Field(default=True)
    VECTOR_DB_RERANK_TOP_K: int = Field(default=100)
    EMBEDDING_DIM: int = Field(default=1024)

    # --- LLM Providers ---
    DEFAULT_LLM_PROVIDER: str = Field(default="ollama")
    OLLAMA_BASE_URL: str = Field(default="http://ollama:11434")
    OLLAMA_MODEL: str = Field(default="qwen2.5:7b-instruct-q4_K_M")
    OLLAMA_EMBEDDING_MODEL: str = Field(default="bge-m3:latest")
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1")
    OPENAI_API_KEY: SecretStr | None = Field(default=None)
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")
    OPENAI_API_KEY_ENV: str = Field(default="OPENAI_API_KEY")
    STRONG_LLM_PROVIDER: str = Field(default="openai")
    STRONG_OPENAI_MODEL: str = Field(default="gpt-4o")
    STRONG_OLLAMA_MODEL: str = Field(default="qwen2.5:14b-instruct-q4_K_M")
    FAST_LLM_PROVIDER: str = Field(default="ollama")
    FAST_OLLAMA_MODEL: str = Field(default="qwen2.5:3b-instruct-q4_K_M")
    FAST_OPENAI_MODEL: str = Field(default="gpt-3.5-turbo")

    # --- Reranker ---
    RERANKER_PROVIDER: str = Field(default="tei")
    RERANKER_BASE_URL: str = Field(default="http://tei-reranker:8080")
    RERANKER_MODEL: str = Field(default="bge-reranker-v2-m3")
    RERANKER_TOP_K: int = Field(default=10)

    # --- Embeddings ---
    EMBEDDING_PROVIDER: str = Field(default="sentence_transformers")
    EMBEDDING_MODEL: str = Field(default="intfloat/multilingual-e5-large")
    SENTENCE_TRANSFORMERS_DEVICE: str = Field(default="cpu")

    # --- OCR ---
    OCR_PROVIDER: str = Field(default="tesseract")
    TESSERACT_CMD: str = Field(default="tesseract")
    OCR_LANGUAGE: str = Field(default="eng+deu+fra+spa")
    POPPLER_PATH: str | None = Field(default=None)

    # --- Telemetry ---
    LANGSMITH_ENABLED: bool = Field(default=False)
    LANGSMITH_API_KEY: SecretStr | None = Field(default=None)
    LANGSMITH_PROJECT: str = Field(default="legal-ai")
    OTEL_EXPORTER_OTLP_ENABLED: bool = Field(default=False)
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(default="http://localhost:4317")
    OTEL_SERVICE_NAME: str = Field(default="legal-ai-service")
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json")
    TRACING_ENABLED: bool = Field(default=False)

    # --- Legal / Domain ---
    LEGAL_JURISDICTION: str = Field(default="us")
    LEGAL_LANGUAGE: str = Field(default="en")
    LEGAL_DISCLAIMER_PATH: str | None = Field(default=None)
    TENANTS: str = Field(default="demo,acme,beta")

    # --- Feature Flags ---
    FEATURE_WEB_SEARCH: bool = Field(default=False)
    FEATURE_OCR: bool = Field(default=True)
    FEATURE_SEMANIC_CACHE: bool = Field(default=True)
    FEATURE_HUMAN_IN_THE_LOOP: bool = Field(default=True)
    FEATURE_AUDIT_LOGGING: bool = Field(default=True)

    # --- Legal safety defaults ---
    DEFAULT_CONFIDENCE_THRESHOLD: float = Field(default=0.85)
    HUMAN_REVIEW_CONFIDENCE_THRESHOLD: float = Field(default=0.70)
    MAX_VERIFICATION_RETRIES: int = Field(default=2)
    MAX_GRAPH_ITERATIONS: int = Field(default=10)
    MAX_RETRIEVAL_RESULTS: int = Field(default=50)

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def tenant_list(self) -> list[str]:
        return [t.strip() for t in self.TENANTS.split(",") if t.strip()]

    @property
    def trusted_service_urls(self) -> list[str]:
        return [u.strip() for u in self.TRUSTED_SERVICE_URLS.split(",") if u.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
