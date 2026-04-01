"""
LangGraph agent graph for VoltEdge warranty claims.

Topology:
  START → router
  router → planner
  planner → (conditional) → claim_state_updater | empathy_node | simple nodes |
                            policy_checker | evidence_planner |
                            vision_analysis | claim_validator | agent_respond |
                            claim_decision | END
  claim_state_updater → planner
  empathy_node → planner
  policy_checker → planner
  evidence_planner → planner
  vision_analysis → claim_validator → planner
  agent_respond → END
  claim_decision → END
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.graph import END, StateGraph

from agent.nodes.agent_respond import agent_respond_node
from agent.nodes.claim_decision import claim_decision_node
from agent.nodes.claim_state_updater import claim_state_updater_node
from agent.nodes.claim_validator import claim_validator_node
from agent.nodes.empathy import empathy_node
from agent.nodes.evidence_planner import evidence_planner_node
from agent.nodes.planner import planner_node
from agent.nodes.policy_checker import policy_checker_node
from agent.nodes.router import router_node
from agent.nodes.simple_nodes import (
    cancellation_node,
    confirmation_handler_node,
    escalation_node,
    fallback_node,
    feedback_node,
    greeting_node,
    post_resolution_close_node,
    status_node,
)
from agent.nodes.vision_analysis import vision_analysis_node
from models.state import AgentState
from observability.tracing import start_span

logger = logging.getLogger(__name__)


def _instrument_node(node_name: str, fn):
    def _wrapped(state: AgentState):
        attributes = {
            "agent.node": node_name,
            "session.id": state.get("session_id", ""),
        }
        with start_span(f"node.{node_name}", attributes=attributes) as span:
            result = fn(state)
            if isinstance(result, dict):
                span.set_attribute("agent.node.output_keys", sorted(result.keys()))
            return result

    return _wrapped

# ── Routing functions (conditional edges) ────────────────────────────────────


def _route_from_router(state: AgentState) -> str:
    return "planner"


def _route_from_empathy(state: AgentState) -> str:
    return "planner"


def _route_from_planner(state: AgentState) -> str:
    next_node = state.get("next_node") or "agent_respond"
    valid_targets = {
        "confirmation_handler",
        "greeting_node",
        "fallback_node",
        "escalation_node",
        "cancellation_node",
        "status_node",
        "post_resolution_close_node",
        "feedback_node",
        "empathy_node",
        "claim_state_updater",
        "policy_checker",
        "evidence_planner",
        "vision_analysis",
        "claim_validator",
        "agent_respond",
        "claim_decision",
        "END",
    }
    if next_node not in valid_targets:
        return "agent_respond"
    return END if next_node == "END" else next_node


# ── Graph builder ─────────────────────────────────────────────────────────────


def _build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Register all nodes
    g.add_node("router", _instrument_node("router", router_node))
    g.add_node("claim_state_updater", _instrument_node("claim_state_updater", claim_state_updater_node))
    g.add_node("confirmation_handler", _instrument_node("confirmation_handler", confirmation_handler_node))
    g.add_node("greeting_node", _instrument_node("greeting_node", greeting_node))
    g.add_node("fallback_node", _instrument_node("fallback_node", fallback_node))
    g.add_node("escalation_node", _instrument_node("escalation_node", escalation_node))
    g.add_node("cancellation_node", _instrument_node("cancellation_node", cancellation_node))
    g.add_node("status_node", _instrument_node("status_node", status_node))
    g.add_node("post_resolution_close_node", _instrument_node("post_resolution_close_node", post_resolution_close_node))
    g.add_node("feedback_node", _instrument_node("feedback_node", feedback_node))
    g.add_node("empathy_node", _instrument_node("empathy_node", empathy_node))
    g.add_node("planner", _instrument_node("planner", planner_node))
    g.add_node("policy_checker", _instrument_node("policy_checker", policy_checker_node))
    g.add_node("evidence_planner", _instrument_node("evidence_planner", evidence_planner_node))
    g.add_node("vision_analysis", _instrument_node("vision_analysis", vision_analysis_node))
    g.add_node("claim_validator", _instrument_node("claim_validator", claim_validator_node))
    g.add_node("agent_respond", _instrument_node("agent_respond", agent_respond_node))
    g.add_node("claim_decision", _instrument_node("claim_decision", claim_decision_node))

    # Entry flow
    g.set_entry_point("router")

    # Router always defers execution selection to planner.
    g.add_conditional_edges(
        "router",
        _route_from_router,
        {"planner": "planner"},
    )

    # Terminal simple nodes
    g.add_edge("confirmation_handler", END)
    g.add_edge("greeting_node", END)
    g.add_edge("fallback_node", END)
    g.add_edge("escalation_node", END)
    g.add_edge("cancellation_node", END)
    g.add_edge("status_node", END)
    g.add_edge("post_resolution_close_node", END)
    g.add_edge("feedback_node", END)

    # Empathy always defers execution decisions to planner.
    g.add_conditional_edges(
        "empathy_node",
        _route_from_empathy,
        {"planner": "planner"},
    )

    g.add_edge("claim_state_updater", "planner")

    # Planner owns all post-router execution decisions.
    g.add_conditional_edges(
        "planner",
        _route_from_planner,
        {
            "confirmation_handler": "confirmation_handler",
            "greeting_node": "greeting_node",
            "fallback_node": "fallback_node",
            "escalation_node": "escalation_node",
            "cancellation_node": "cancellation_node",
            "status_node": "status_node",
            "post_resolution_close_node": "post_resolution_close_node",
            "feedback_node": "feedback_node",
            "empathy_node": "empathy_node",
            "claim_state_updater": "claim_state_updater",
            "policy_checker": "policy_checker",
            "evidence_planner": "evidence_planner",
            "vision_analysis": "vision_analysis",
            "claim_validator": "claim_validator",
            "agent_respond": "agent_respond",
            "claim_decision": "claim_decision",
            END: END,
        },
    )

    g.add_edge("policy_checker", "planner")
    g.add_edge("evidence_planner", "planner")
    g.add_edge("vision_analysis", "claim_validator")
    g.add_edge("claim_validator", "planner")
    g.add_edge("agent_respond", END)
    g.add_edge("claim_decision", END)

    return g


@lru_cache(maxsize=1)
def get_compiled_graph():
    """Return the compiled LangGraph graph. Cached after first build."""
    graph = _build_graph().compile()
    logger.info("LangGraph agent graph compiled")
    return graph
