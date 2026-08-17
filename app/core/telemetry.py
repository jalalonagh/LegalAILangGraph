"""
OpenTelemetry instrumentation setup.

Provides a configurable bridge between LangSmith (when enabled) and
OpenTelemetry exporters. Tracing is disabled by default; it only activates
when ``TRACING_ENABLED`` or ``OTEL_EXPORTER_OTLP_ENABLED`` is set.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanHTTPExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ALWAYS_ON, TraceOptions, TraceState
from opentelemetry.trace import Link, Span, get_tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger()


def _build_resource() -> Resource:
    return Resource.create(
        {
            "service.name": settings.OTEL_SERVICE_NAME,
            "service.version": settings.VERSION,
            "environment": settings.APP_ENV,
        }
    )


def init_tracing() -> TracerProvider | None:
    """Initialize OpenTelemetry tracing. Returns the provider or None if disabled."""
    if not (settings.TRACING_ENABLED or settings.OTEL_EXPORTER_OTLP_ENABLED):
        logger.info("tracing_disabled")
        return None

    resource = _build_resource()
    provider = TracerProvider(resource=resource, sampler=_resolve_sampler())
    provider.add_span_processor(BatchSpanProcessor(_build_exporter()))

    trace.set_tracer_provider(provider)
    trace.use_span_processor(BatchSpanProcessor(_build_exporter()))

    logger.info("tracing_initialized", endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)
    return provider


def _resolve_sampler():
    """Return a sampler - always-on in production, parent-based otherwise."""
    if settings.is_production:
        return ALWAYS_ON
    return ALWAYS_ON


def _build_exporter():
    """Build the OTLP exporter (HTTP preferred, with gRPC fallback)."""
    endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT
    if endpoint.startswith("http"):
        return OTLPSpanHTTPExporter(endpoint=endpoint)
    return OTLPSpanExporter(endpoint=endpoint, insecure=True)


def get_tracer(name: str = "legal-ai"):
    """Get a tracer instance, or a no-op tracer if tracing is disabled."""
    tp = trace.get_tracer_provider()
    if isinstance(tp, trace.ProxyTracerProvider) and tp._tracer_provider is None:
        # NoopTracerProvider
        return _NoopTracer()
    return trace.get_tracer(name)


class _NoopTracer:
    """No-op tracer used when OpenTelemetry is disabled."""

    def start_as_current_span(self, *args, **kwargs):
        from contextlib import nullcontext

        return nullcontext()


def instrument_app(app):
    """Instrument a FastAPI application with OpenTelemetry middleware."""
    if not (settings.TRACING_ENABLED or settings.OTEL_EXPORTER_OTLP_ENABLED):
        return
    try:
        FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())
        logger.info("fastapi_instrumentation_enabled")
    except Exception as exc:
        logger.warning("fastapi_instrumentation_failed", error=str(exc))


# Metrics helpers via Prometheus
from prometheus_client import Counter, Histogram, Gauge  # noqa: E402

# --- Standard metrics ---
request_counter = Counter(
    "legalai_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)

active_runs = Gauge("legalai_active_runs", "Number of currently active workflow runs")

run_counter = Counter(
    "legalai_runs_total",
    "Workflow run counts",
    ["workflow", "status"],
)

workflow_latency = Histogram(
    "legalai_workflow_duration_seconds",
    "Workflow execution latency in seconds",
    ["workflow"],
)

node_latency = Histogram(
    "legalai_node_duration_seconds",
    "Node execution latency in seconds",
    ["node"],
)

llm_latency = Histogram(
    "legalai_llm_duration_seconds",
    "LLM call latency in seconds",
    ["model", "provider"],
)

llm_tokens = Counter(
    "legalai_llm_tokens_total",
    "Total tokens consumed",
    ["direction", "model"],
)

retrieval_latency = Histogram(
    "legalai_retrieval_duration_seconds",
    "Retrieval latency in seconds",
    ["retriever"],
)

tool_latency = Histogram(
    "legalai_tool_duration_seconds",
    "Tool execution latency in seconds",
    ["tool"],
)

verification_failures = Counter(
    "legalai_verification_failures_total",
    "Number of verification failures",
    ["workflow"],
)

human_reviews = Counter(
    "legalai_human_reviews_total",
    "Number of human reviews triggered",
    ["workflow"],
)

verification_accuracy = Histogram(
    "legalai_verification_accuracy",
    "Distribution of verification confidence scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


def init_prometheus(app):
    """Mount Prometheus metrics endpoint."""
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app)
        logger.info("prometheus_instrumentator_enabled")
    except ImportError:
        logger.info("prometheus_fastapi_instrumentator_not_installed")
