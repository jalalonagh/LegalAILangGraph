"""
Root graph builder.

Constructs the main LangGraph workflow that routes requests to the
appropriate subgraph based on intent classification. The root graph
handles: request validation, intent/domain classification, risk
assessment, and routing.

Subgraphs:
  - Legal Q&A (legal_qa)
  - Legal Research (legal_research)
  - Document Analysis (document_analysis)
  - Contract Analysis (contract_analysis)
  - Case Analysis (case_analysis)
  - Legal Drafting (legal_drafting)
  - Document Comparison (document_comparison)
  - Summarization (summarization)
  - General (general)

Uses conditional routing and LangGraph interrupts for human review.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.constants import START
from langgraph.errors import GraphInterrupt
from langgraph.types import Send, interrupt
from langgraph.types import Checkpointer

from app.agents.state import LegalAIState
from app.core.config import settings
from app.core.exceptions import (
    CancellationError,
    LLMProviderError,
    LegalAIError,
    WorkflowError,
)
from app.core.logging import get_logger
from app.agents.routing import IntentClassifier, RiskAssessor, WorkflowRouter
from app.agents.nodes.base_nodes import (
    validate_request_node,
    load_context_node,
    classify_intent_node,
    classify_domain_node,
    assess_risk_node,
    route_workflow_node,
    audit_request_node,
    final_response_node,
)

logger = get_logger()


class RootGraphBuilder:
    """Builder for the root LangGraph workflow.

    Uses lazy subgraph construction: subgraphs are only built when
    first needed, and cached for reuse.
    """

    def __init__(self, container) -> None:
        self._container = container
        self._graphs: dict[str, CompiledStateGraph] = {}
        self._lock = asyncio.Lock()

    async def build(self, workflow_type: str) -> CompiledStateGraph:
        """Build and return the compiled root graph for a workflow type."""
        key = workflow_type or "general"
        if key in self._graphs:
            return self._graphs[key]

        # Build the root graph with all routing nodes
        graph = self._build_root_graph()
        compiled = graph.compile()

        self._graphs[key] = compiled
        logger.info("graph_compiled", graph=key, nodes=len(graph.nodes))
        return compiled

    async def build_subgraph(self, workflow_type: str) -> CompiledStateGraph:
        """Build a specific subgraph (lazy)."""
        builders = {
            "legal_research": self._build_legal_research_graph,
            "legal_qa": self._build_qa_graph,
            "document_analysis": self._build_document_analysis_graph,
            "contract_analysis": self._build_contract_analysis_graph,
            "case_analysis": self._build_case_analysis_graph,
            "legal_drafting": self._build_drafting_graph,
            "document_comparison": self._build_comparison_graph,
            "summarization": self._build_summarization_graph,
            "general": self._build_general_graph,
        }

        builder = builders.get(workflow_type, builders["general"])
        return builder()

    # ------------------------------------------------------------------
    # Root graph
    # ------------------------------------------------------------------
    def _build_root_graph(self) -> StateGraph:
        """Build the root routing graph."""
        workflow = StateGraph(LegalAIState)

        # Nodes
        workflow.add_node("validate_request", validate_request_node(self._container))
        workflow.add_node("load_context", load_context_node(self._container))
        workflow.add_node("classify_intent", classify_intent_node(self._container))
        workflow.add_node("classify_domain", classify_domain_node(self._container))
        workflow.add_node("assess_risk", assess_risk_node(self._container))
        workflow.add_node("route_workflow", route_workflow_node(self._container))
        workflow.add_node("audit_request", audit_request_node(self._container))

        # Subgraph nodes (added as compiled subgraphs)
        subgraph_names = [
            "legal_research", "legal_qa", "document_analysis",
            "contract_analysis", "case_analysis", "legal_drafting",
            "document_comparison", "summarization", "general",
        ]
        for name in subgraph_names:
            # Use a lazy subgraph runner
            workflow.add_node(name, self._make_subgraph_runner(name))

        # Final response
        workflow.add_node("final_response", final_response_node(self._container))

        # Edges
        workflow.add_edge(START, "validate_request")
        workflow.add_edge("validate_request", "load_context")
        workflow.add_edge("load_context", "classify_intent")
        workflow.add_edge("classify_intent", "classify_domain")
        workflow.add_edge("classify_domain", "assess_risk")
        workflow.add_edge("assess_risk", "route_workflow")
        workflow.add_edge("route_workflow", "audit_request")
        workflow.add_conditional_edges(
            "route_workflow",
            self._route_from_router,
            {name: name for name in subgraph_names} | {"__end__": END},
        )
        workflow.add_edge("legal_research", "final_response")
        workflow.add_edge("legal_qa", "final_response")
        workflow.add_edge("document_analysis", "final_response")
        workflow.add_edge("contract_analysis", "final_response")
        workflow.add_edge("case_analysis", "final_response")
        workflow.add_edge("legal_drafting", "final_response")
        workflow.add_edge("document_comparison", "final_response")
        workflow.add_edge("summarization", "final_response")
        workflow.add_edge("general", "final_response")
        workflow.add_edge("audit_request", END)  # audit is fire-and-forget after routing

        return workflow

    def _route_from_router(self, state: LegalAIState) -> str:
        """Conditional edge function: route to the appropriate subgraph."""
        wf = state.workflow
        if wf in ("legal_research", "legal_qa", "document_analysis",
                  "contract_analysis", "case_analysis", "legal_drafting",
                  "document_comparison", "summarization", "general"):
            return wf
        return "general"

    def _make_subgraph_runner(self, workflow_type: str):
        """Create a node function that runs a subgraph."""
        async def _run(state: LegalAIState) -> LegalAIState:
            graph = await self.build_subgraph(workflow_type)
            config = {
                "configurable": {"thread_id": str(uuid.uuid4()), "parent_run_id": state.request_id},
                "recursion_limit": settings.MAX_GRAPH_ITERATIONS,
            }
            # The subgraph operates on a slice of the parent state
            state_dict = state.model_dump()
            result = await graph.ainvoke(state_dict, config)
            if isinstance(result, dict):
                merged = state.model_dump()
                merged.update(result)
                return LegalAIState(**merged)
            return result

        _run.__name__ = workflow_type
        return _run

    # ------------------------------------------------------------------
    # Subgraph builders (each builds a complete subgraph)
    # ------------------------------------------------------------------
    def _build_legal_research_graph(self) -> CompiledStateGraph:
        from app.agents.graphs.legal_research_graph import LegalResearchGraphBuilder

        builder = LegalResearchGraphBuilder(self._container)
        return builder.build()

    def _build_qa_graph(self) -> CompiledStateGraph:
        from app.agents.graphs.qa_graph import LegalQAGraphBuilder

        builder = LegalQAGraphBuilder(self._container)
        return builder.build()

    def _build_document_analysis_graph(self) -> CompiledStateGraph:
        from app.agents.graphs.document_analysis_graph import DocumentAnalysisGraphBuilder

        builder = DocumentAnalysisGraphBuilder(self._container)
        return builder.build()

    def _build_contract_analysis_graph(self) -> CompiledStateGraph:
        from app.agents.graphs.contract_analysis_graph import ContractAnalysisGraphBuilder

        builder = ContractAnalysisGraphBuilder(self._container)
        return builder.build()

    def _build_case_analysis_graph(self) -> CompiledStateGraph:
        from app.agents.graphs.case_analysis_graph import CaseAnalysisGraphBuilder

        builder = CaseAnalysisGraphBuilder(self._container)
        return builder.build()

    def _build_drafting_graph(self) -> CompiledStateGraph:
        from app.agents.graphs.drafting_graph import DraftingGraphBuilder

        builder = DraftingGraphBuilder(self._container)
        return builder.build()

    def _build_comparison_graph(self) -> CompiledStateGraph:
        from app.agents.graphs.comparison_graph import ComparisonGraphBuilder

        builder = ComparisonGraphBuilder(self._container)
        return builder.build()

    def _build_summarization_graph(self) -> CompiledStateGraph:
        from app.agents.graphs.summarization_graph import SummarizationGraphBuilder

        builder = SummarizationGraphBuilder(self._container)
        return builder.build()

    def _build_general_graph(self) -> CompiledStateGraph:
        from app.agents.graphs.general_graph import GeneralGraphBuilder

        builder = GeneralGraphBuilder(self._container)
        return builder.build()


__all__ = ["RootGraphBuilder"]
