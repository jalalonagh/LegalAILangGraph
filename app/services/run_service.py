"""
Run service for managing LangGraph workflow execution.

Provides:
- Thread/run creation with UUID-based identifiers
- Async execution with configurable checkpoints
- State inspection and resume
- Streaming with SSE-compatible event emission
- Cancellation support
- Audit integration
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.pregel import Pregel

from app.agents.state import LegalAIState, create_initial_state
from app.core.config import settings
from app.core.exceptions import WorkflowError, CancellationError, LLMProviderError
from app.core.logging import get_logger, set_run_id
from app.core.telemetry import active_runs, run_counter, workflow_latency
from app.services.service_container import get_service

logger = get_logger()


class CheckpointManager:
    """Manages LangGraph checkpoint storage using PostgreSQL.

    Uses langgraph-checkpoint-postgres for production persistence.
    Falls back to an in-memory checkpoint store for development/testing
    when PostgreSQL is unavailable.
    """

    def __init__(self, db_config=None):
        self._db_config = db_config
        self._saver: BaseCheckpointSaver | None = None
        self._fallback: BaseCheckpointSaver | None = None

    async def get_saver(self) -> BaseCheckpointSaver:
        """Return a checkpoint saver, preferring PostgreSQL."""
        if self._saver is not None:
            return self._saver

        # Try PostgreSQL-backed checkpoint store
        try:
            from langgraph.checkpoint.postgres.aio import PostgresSaver

            from app.infrastructure.database.session import get_engine

            engine = get_engine()
            self._saver = await PostgresSaver.create(engine, max_recent_writes=10, max_length=2_000_000_000)
            await self._saver.setup()
            logger.info("checkpoint_store_postgres")
            return self._saver
        except Exception as exc:  # noqa: BLE001
            logger.warning("postgres_checkpoint_failed_fallback_memory", error=str(exc))
            # Fallback to in-memory
            from langgraph.checkpoint.memory import InMemoryCheckpointSaver

            self._fallback = InMemoryCheckpointSaver()
            self._saver = self._fallback
            return self._saver

    async def close(self) -> None:
        if self._saver and hasattr(self._saver, "close"):
            await self._saver.close()


class RunService:
    """Orchestrates LangGraph workflow execution, persistence, and streaming."""

    def __init__(self) -> None:
        self._container = get_service()
        self._checkpoint_manager = CheckpointManager()
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, run_id: str) -> asyncio.Lock:
        if run_id not in self._locks:
            self._locks[run_id] = asyncio.Lock()
        return self._locks[run_id]

    async def create_run(
        self,
        question: str,
        workflow_type: str = "general",
        tenant_id: str = "demo",
        user_id: str = "anonymous",
        case_id: str | None = None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Create a new run and return the run ID."""
        run_id = str(uuid.uuid4())
        thread_id = conversation_id or str(uuid.uuid4())

        set_run_id(run_id)

        state = create_initial_state(
            question=question,
            tenant_id=tenant_id,
            user_id=user_id,
            case_id=case_id,
            conversation_id=conversation_id,
            request_id=request_id or run_id,
            messages=messages or [],
            workflow=workflow_type,
            **kwargs,
        )

        logger.info("run_created", run_id=run_id, thread_id=thread_id, workflow=workflow_type, tenant=tenant_id)
        return run_id

    async def execute(
        self,
        run_id: str,
        question: str,
        workflow_type: str = "general",
        tenant_id: str = "demo",
        user_id: str = "anonymous",
        case_id: str | None = None,
        conversation_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a workflow synchronously and return the final state."""
        thread_id = conversation_id or run_id
        set_run_id(run_id)

        saver = await self._checkpoint_manager.get_saver()
        config = {
            "configurable": {"thread_id": thread_id, "run_id": run_id},
            "recursion_limit": settings.MAX_GRAPH_ITERATIONS,
        }

        active_runs.inc()
        start_time = datetime.now(timezone.utc)
        run_counter.labels(workflow=workflow_type, status="started").inc()

        try:
            graph = await self._build_graph(workflow_type)
            # Check for pending interrupts
            state = graph.get_state(config)

            if state.next:
                # Resume from interrupted state
                logger.info("run_resuming", run_id=run_id, next_nodes=list(state.next))
                result = await graph.ainvoke(None, config)
            else:
                # Fresh execution
                initial_state = create_initial_state(
                    question=question,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    case_id=case_id,
                    conversation_id=conversation_id or thread_id,
                    request_id=run_id,
                    messages=messages or [],
                    workflow=workflow_type,
                    **kwargs,
                )
                result = await graph.ainvoke(initial_state.model_dump(), config)

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            workflow_latency.labels(workflow=workflow_type).observe(duration)
            run_counter.labels(workflow=workflow_type, status="completed").inc()
            logger.info("run_completed", run_id=run_id, duration_seconds=duration)

            # Record audit + usage
            await self._record_audit_and_usage(run_id, workflow_type, tenant_id, user_id, result)

            return self._extract_response(result)

        except asyncio.CancelledError:
            run_counter.labels(workflow=workflow_type, status="cancelled").inc()
            logger.warning("run_cancelled", run_id=run_id)
            raise CancellationError("Workflow execution cancelled")
        except Exception as exc:  # noqa: BLE001
            run_counter.labels(workflow=workflow_type, status="failed").inc()
            logger.error("run_failed", run_id=run_id, error=str(exc), exc_info=True)
            raise WorkflowError(f"Workflow '{workflow_type}' failed: {exc}") from exc
        finally:
            active_runs.dec()
            set_run_id(None)

    async def execute_stream(
        self,
        run_id: str,
        question: str,
        workflow_type: str = "general",
        tenant_id: str = "demo",
        user_id: str = "anonymous",
        case_id: str | None = None,
        conversation_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Execute a workflow with streaming event emission."""
        thread_id = conversation_id or run_id
        set_run_id(run_id)

        saver = await self._checkpoint_manager.get_saver()
        config = {
            "configurable": {"thread_id": thread_id, "run_id": run_id},
            "recursion_limit": settings.MAX_GRAPH_ITERATIONS,
        }

        active_runs.inc()
        start_time = datetime.now(timezone.utc)

        yield {"event": "run_started", "data": {"run_id": run_id, "workflow": workflow_type}, "timestamp": datetime.now(timezone.utc).isoformat()}

        try:
            graph = await self._build_graph(workflow_type)
            initial_state = create_initial_state(
                question=question,
                tenant_id=tenant_id,
                user_id=user_id,
                case_id=case_id,
                conversation_id=conversation_id or thread_id,
                request_id=run_id,
                messages=messages or [],
                workflow=workflow_type,
                **kwargs,
            )

            async for event in graph.astream_events(
                initial_state.model_dump(),
                config,
                version="v1",
            ):
                event_data = self._normalize_event(event)
                if event_data is not None:
                    yield event_data

            # Final state
            final_state = graph.get_state(config)
            final_data = self._extract_response(final_state.values)
            yield {
                "event": "run_completed",
                "data": final_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            workflow_latency.labels(workflow=workflow_type).observe(duration)
            run_counter.labels(workflow=workflow_type, status="completed").inc()
            await self._record_audit_and_usage(run_id, workflow_type, tenant_id, user_id, final_state.values)

        except asyncio.CancelledError:
            yield {"event": "error", "data": {"error": "Workflow cancelled", "run_id": run_id}, "timestamp": datetime.now(timezone.utc).isoformat()}
            run_counter.labels(workflow=workflow_type, status="cancelled").inc()
            raise CancellationError("Workflow execution cancelled")
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": {"error": str(exc), "run_id": run_id}, "timestamp": datetime.now(timezone.utc).isoformat()}
            run_counter.labels(workflow=workflow_type, status="failed").inc()
            logger.error("run_failed", run_id=run_id, error=str(exc), exc_info=True)
            raise WorkflowError(f"Workflow '{workflow_type}' failed: {exc}") from exc
        finally:
            active_runs.dec()
            set_run_id(None)

    async def get_run_state(self, run_id: str, thread_id: str) -> dict[str, Any]:
        """Get the current state of a run."""
        graph = await self._build_graph("general")
        config = {"configurable": {"thread_id": thread_id, "run_id": run_id}}
        state = graph.get_state(config)

        return {
            "run_id": run_id,
            "thread_id": thread_id,
            "status": "active" if state.next else "completed",
            "current_node": state.values.get("current_node", "") if state.values else "",
            "next_node": list(state.next) if state.next else [],
            "state": state.values or {},
            "values": state.values or {},
            "interrupts": state.interrupts if hasattr(state, "interrupts") else [],
        }

    async def resume_run(
        self,
        run_id: str,
        thread_id: str,
        workflow_type: str = "general",
        **resume_input: Any,
    ) -> dict[str, Any]:
        """Resume an interrupted run by providing the required interrupt input."""
        saver = await self._checkpoint_manager.get_saver()
        config = {
            "configurable": {"thread_id": thread_id, "run_id": run_id},
            "recursion_limit": settings.MAX_GRAPH_ITERATIONS,
        }

        graph = await self._build_graph(workflow_type)
        result = await graph.ainvoke(resume_input or None, config)
        return self._extract_response(result)

    async def cancel_run(self, run_id: str) -> bool:
        """Cancel a running task by run_id."""
        task = self._running_tasks.get(run_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    async def list_runs(
        self,
        tenant_id: str,
        limit: int = 50,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """List recent runs for a tenant (from audit events)."""
        from app.infrastructure.database.session import get_db_transaction
        from app.services.service_container import get_service

        container = get_service()
        async with get_db_transaction() as session:
            audit = container.get_audit_service(session)
            events = await audit.query(
                tenant_id=tenant_id,
                event_type="request",
                limit=limit,
            )
            return [
                {
                    "run_id": e.get("run_id"),
                    "workflow": e.get("workflow"),
                    "user_id": e.get("user_id"),
                    "created_at": e.get("created_at"),
                    "result": e.get("result"),
                }
                for e in events
            ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _build_graph(self, workflow_type: str):
        """Build (or retrieve cached) the LangGraph for a given workflow type."""
        from app.agents.graphs.root_graph import RootGraphBuilder

        builder = RootGraphBuilder(self._container)
        return await builder.build(workflow_type)

    def _normalize_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a LangGraph stream event to our stable event schema."""
        event_type = event.get("event", "")
        data = event.get("data", {})

        if event_type == "on_chat_model_stream":
            chunk = data.get("chunk")
            if isinstance(chunk, str):
                return {"event": "token", "data": {"content": chunk}, "timestamp": datetime.now(timezone.utc).isoformat()}
        elif event_type == "on_tool_start":
            return {"event": "tool_started", "data": {"tool": data.get("name"), "input": data.get("input")}, "timestamp": datetime.now(timezone.utc).isoformat()}
        elif event_type == "on_tool_end":
            return {"event": "tool_completed", "data": {"tool": data.get("name"), "output": str(data.get("output"))[:500]}, "timestamp": datetime.now(timezone.utc).isoformat()}
        elif event_type == "on_chain_start":
            node = data.get("name", "") or event.get("name", "")
            return {"event": "node_started", "data": {"node": node}, "timestamp": datetime.now(timezone.utc).isoformat()}
        elif event_type == "on_chain_end":
            node = data.get("name", "") or event.get("name", "")
            output = data.get("output", {})
            return {"event": "node_completed", "data": {"node": node, "output": str(output)[:500]}, "timestamp": datetime.now(timezone.utc).isoformat()}
        return None

    def _extract_response(self, state_data: dict[str, Any]) -> dict[str, Any]:
        """Extract the final response structure from state data."""
        return {
            "run_id": state_data.get("request_id", ""),
            "workflow": state_data.get("workflow", "general"),
            "answer": state_data.get("answer", ""),
            "summary": state_data.get("summary", ""),
            "confidence": state_data.get("confidence", 0.0),
            "legal_issues": state_data.get("legal_issues", []),
            "facts": state_data.get("facts", []),
            "assumptions": state_data.get("assumptions", []),
            "authorities": state_data.get("authorities", []),
            "citations": state_data.get("citations", []),
            "arguments": state_data.get("arguments", []),
            "counterarguments": state_data.get("counterarguments", []),
            "risks": state_data.get("risks", []),
            "uncertainties": state_data.get("uncertainties", []),
            "requires_human_review": state_data.get("requires_human_review", False),
            "human_review_request_id": state_data.get("human_review_request_id"),
            "verification": state_data.get("verification", {}),
            "model_used": state_data.get("model_used", ""),
            "models_used": state_data.get("models_used", []),
            "document_analysis": state_data.get("document_analysis", {}),
            "contract_analysis": state_data.get("contract_analysis", {}),
            "case_analysis": state_data.get("case_analysis", {}),
            "comparison_result": state_data.get("comparison_result", {}),
            "claims": state_data.get("claims", []),
            "evidence": state_data.get("evidence_set", []),
            "error": state_data.get("error"),
        }

    async def _record_audit_and_usage(self, run_id, workflow, tenant_id, user_id, result) -> None:
        """Record audit events and usage metrics after a run."""
        try:
            from app.infrastructure.database.session import get_db_transaction

            async with get_db_transaction() as session:
                container = get_service()
                audit = container.get_audit_service(session)
                await audit.record_request(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    run_id=run_id,
                    workflow=workflow,
                    input_text=result.get("answer", ""),
                    confidence=result.get("confidence"),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("audit_record_failed", error=str(exc))


# Singleton
_run_service: RunService | None = None


def get_run_service() -> RunService:
    global _run_service
    if _run_service is None:
        _run_service = RunService()
    return _run_service


__all__ = ["RunService", "CheckpointManager", "get_run_service"]
