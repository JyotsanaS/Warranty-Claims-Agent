"""
LangGraph agent graph for VoltEdge warranty claims.

Topology:
  START → context_extractor → claim_extractor → router
  router → (conditional) → one of:
      confirmation_handler | greeting_node | fallback_node |
      escalation_node | cancellation_node | status_node |
      empathy_node → policy_checker |
      policy_checker
  policy_checker → (conditional) → evidence_planner → vision_analysis → claim_validator → agent_respond
                                 → agent_respond
  agent_respond → (conditional) → claim_decision → END
                               → END
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.graph import END, StateGraph

from agent.nodes.agent_respond import agent_respond_node
from agent.nodes.claim_decision import claim_decision_node
from agent.nodes.claim_extractor import claim_extractor_node
from agent.nodes.claim_validator import claim_validator_node
from agent.nodes.context_extractor import context_extractor_node
from agent.nodes.empathy import empathy_node
from agent.nodes.evidence_planner import evidence_planner_node
from agent.nodes.policy_checker import policy_checker_node
from agent.nodes.router import router_node
from agent.nodes.simple_nodes import (
    cancellation_node,
    confirmation_handler_node,
    escalation_node,
    fallback_node,
    greeting_node,
    status_node,
)
from agent.nodes.vision_analysis import vision_analysis_node
from models.state import AgentState

logger = logging.getLogger(__name__)

# ── Routing functions (conditional edges) ────────────────────────────────────


def _route_from_router(state: AgentState) -> str:
    if state.get("pending_switch_confirmation"):
        return "confirmation_handler"

    intent = state.get("intent", "issue")
    return {
        "greeting": "greeting_node",
        "out_of_scope": "fallback_node",
        "escalation": "escalation_node",
        "cancellation": "cancellation_node",
        "status_query": "status_node",
        "frustration": "empathy_node",
    }.get(intent, "policy_checker")


def _route_from_empathy(state: AgentState) -> str:
    """Skip policy_checker when there are no claims to process."""
    user_claims: list[dict] = state.get("user_claims", [])
    has_pending = any(c.get("policy_coverage") is None for c in user_claims)
    return "policy_checker" if has_pending else "agent_respond"


def _route_from_policy_checker(state: AgentState) -> str:
    has_image = bool(state.get("image_local_path"))
    if not has_image:
        return "agent_respond"

    user_claims: list[dict] = state.get("user_claims", [])
    policy_valid = [
        c for c in user_claims
        if (c.get("policy_coverage") or {}).get("covered")
        and c.get("visual_checks") is not None  # checks not yet planned
    ]
    if policy_valid or any(c.get("visual_checks") for c in user_claims):
        return "evidence_planner"

    return "agent_respond"


def _route_from_agent_respond(state: AgentState) -> str:
    """decision_gate: if all claims have a final verdict, emit claim_decision."""
    user_claims: list[dict] = state.get("user_claims", [])
    if not user_claims:
        return END

    final_verdicts = {"approved", "rejected", "escalated"}
    all_resolved = all(c.get("claim_verdict") in final_verdicts for c in user_claims)
    return "claim_decision" if all_resolved else END


# ── Graph builder ─────────────────────────────────────────────────────────────


def _build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Register all nodes
    g.add_node("context_extractor", context_extractor_node)
    g.add_node("claim_extractor", claim_extractor_node)
    g.add_node("router", router_node)
    g.add_node("confirmation_handler", confirmation_handler_node)
    g.add_node("greeting_node", greeting_node)
    g.add_node("fallback_node", fallback_node)
    g.add_node("escalation_node", escalation_node)
    g.add_node("cancellation_node", cancellation_node)
    g.add_node("status_node", status_node)
    g.add_node("empathy_node", empathy_node)
    g.add_node("policy_checker", policy_checker_node)
    g.add_node("evidence_planner", evidence_planner_node)
    g.add_node("vision_analysis", vision_analysis_node)
    g.add_node("claim_validator", claim_validator_node)
    g.add_node("agent_respond", agent_respond_node)
    g.add_node("claim_decision", claim_decision_node)

    # Entry flow
    g.set_entry_point("context_extractor")
    g.add_edge("context_extractor", "claim_extractor")
    g.add_edge("claim_extractor", "router")

    # Router fan-out (conditional)
    g.add_conditional_edges(
        "router",
        _route_from_router,
        {
            "confirmation_handler": "confirmation_handler",
            "greeting_node": "greeting_node",
            "fallback_node": "fallback_node",
            "escalation_node": "escalation_node",
            "cancellation_node": "cancellation_node",
            "status_node": "status_node",
            "empathy_node": "empathy_node",
            "policy_checker": "policy_checker",
        },
    )

    # Terminal simple nodes
    g.add_edge("confirmation_handler", END)
    g.add_edge("greeting_node", END)
    g.add_edge("fallback_node", END)
    g.add_edge("escalation_node", END)
    g.add_edge("cancellation_node", END)
    g.add_edge("status_node", END)

    # Empathy → policy_checker only when there are unchecked claims, else skip to agent_respond
    g.add_conditional_edges(
        "empathy_node",
        _route_from_empathy,
        {"policy_checker": "policy_checker", "agent_respond": "agent_respond"},
    )

    # Policy checker fan-out (conditional on image + coverage)
    g.add_conditional_edges(
        "policy_checker",
        _route_from_policy_checker,
        {
            "evidence_planner": "evidence_planner",
            "agent_respond": "agent_respond",
        },
    )

    # Vision pipeline
    g.add_edge("evidence_planner", "vision_analysis")
    g.add_edge("vision_analysis", "claim_validator")
    g.add_edge("claim_validator", "agent_respond")

    # Decision gate (conditional after agent_respond)
    g.add_conditional_edges(
        "agent_respond",
        _route_from_agent_respond,
        {
            "claim_decision": "claim_decision",
            END: END,
        },
    )

    g.add_edge("claim_decision", END)

    return g


@lru_cache(maxsize=1)
def get_compiled_graph():
    """Return the compiled LangGraph graph. Cached after first build."""
    graph = _build_graph().compile()
    logger.info("LangGraph agent graph compiled")
    return graph
