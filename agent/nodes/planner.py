"""
Planner node — converts the current state into an explicit execution plan and
the next node to execute.
"""
from __future__ import annotations

from models.state import AgentState
from agent.nodes.simple_nodes import is_negative_reply

_SIMPLE_INTENT_ROUTES = {
    "greeting": "greeting_node",
    "out_of_scope": "fallback_node",
    "escalation": "escalation_node",
    "cancellation": "cancellation_node",
    "status_query": "status_node",
}

_CLAIM_STATE_INTENTS = {"claim", "issue", "evidence", "clarification", "policy"}


def _has_unassessed_claims(user_claims: list[dict]) -> bool:
    return any(c.get("policy_coverage") is None for c in user_claims)


def _has_final_verdicts(user_claims: list[dict]) -> bool:
    if not user_claims:
        return False
    final_verdicts = {"approved", "rejected", "escalated"}
    return all(c.get("claim_verdict") in final_verdicts for c in user_claims)


def _has_policy_grounding(user_claims: list[dict], policy_clauses: list[str], policy_context: list[str]) -> bool:
    if policy_clauses or policy_context:
        return True
    for claim in user_claims:
        coverage = claim.get("policy_coverage") or {}
        if coverage.get("policy_clauses"):
            return True
    return False


def planner_node(state: AgentState) -> dict:
    intent = state.get("intent", "")
    user_claims: list[dict] = list(state.get("user_claims", []))
    policy_context: list[str] = list(state.get("policy_context", []))
    policy_clauses: list[str] = list(state.get("policy_clauses", []))
    has_image = bool(state.get("image_local_path"))
    claim_state_updated_this_turn = bool(state.get("claim_state_updated_this_turn"))
    awaiting_post_resolution_followup = bool(state.get("awaiting_post_resolution_followup"))
    last_user = next(
        (m.get("content", "") for m in reversed(state.get("messages", [])) if m.get("role") == "user"),
        "",
    )
    has_claim_memory = bool(state.get("user_claims") or state.get("claim_items"))
    image_only_followup = has_image and not str(last_user).strip() and has_claim_memory

    plan: list[str] = []
    reason = ""
    next_node = "agent_respond"
    state_updates: dict = {}

    if state.get("awaiting_feedback"):
        plan = ["Collect user feedback and close conversation"]
        next_node = "feedback_node"
        reason = "Awaiting like/dislike feedback from user before closing."
    elif state.get("pending_switch_confirmation"):
        plan = ["Resolve pending claim-context switch", "Await or process user confirmation"]
        next_node = "confirmation_handler"
        reason = "Pending context switch confirmation takes priority."
    elif (
        not image_only_followup
        and
        not claim_state_updated_this_turn
        and (intent in _CLAIM_STATE_INTENTS or (intent == "frustration" and (last_user or has_image)))
    ):
        plan = ["Extract structured claim details from the current turn", "Continue planning"]
        next_node = "claim_state_updater"
        reason = "This turn may add or update claim context before downstream reasoning."
    elif awaiting_post_resolution_followup and is_negative_reply(last_user):
        plan = ["Close the resolved support conversation", "Request feedback"]
        next_node = "post_resolution_close_node"
        reason = "The user declined further help after a final claim outcome."
    elif awaiting_post_resolution_followup:
        state_updates["awaiting_post_resolution_followup"] = False
        state_updates["claim_state_updated_this_turn"] = False
    elif intent in _SIMPLE_INTENT_ROUTES:
        next_node = _SIMPLE_INTENT_ROUTES[intent]
        plan = [f"Handle {intent.replace('_', ' ')} request directly", "Return response"]
        reason = f"Intent '{intent}' maps to a terminal support flow."
    elif intent == "frustration" and not _has_unassessed_claims(user_claims):
        plan = ["Acknowledge frustration", "Respond using current claim state"]
        next_node = "agent_respond"
        reason = "No pending policy work remains; respond directly."
    elif _has_unassessed_claims(user_claims):
        plan = ["Assess policy coverage for unresolved claims"]
        next_node = "policy_checker"
        reason = "At least one claim has no policy assessment yet."
    else:
        covered_without_checks = [
            claim for claim in user_claims
            if (claim.get("policy_coverage") or {}).get("covered") and not claim.get("visual_checks")
        ]
        pending_visual_validation = [
            claim for claim in user_claims
            if (claim.get("policy_coverage") or {}).get("covered")
            and any(not check.get("finding") for check in claim.get("visual_checks", []))
        ]

        if has_image and covered_without_checks:
            plan = ["Plan evidence checks for covered claims", "Analyse uploaded image", "Validate claim"]
            next_node = "evidence_planner"
            reason = "A covered claim has an image but no planned visual checks."
        elif has_image and pending_visual_validation:
            plan = ["Analyse uploaded image against planned checks", "Validate claim"]
            next_node = "vision_analysis"
            reason = "Visual checks exist and an image is available."
        elif _has_final_verdicts(user_claims) and _has_policy_grounding(user_claims, policy_clauses, policy_context):
            plan = ["Respond to the user", "Issue final decision"]
            next_node = "claim_decision"
            reason = "All claims are resolved and the decision is grounded in retrieved policy."
        else:
            plan = ["Respond to the user"]
            next_node = "agent_respond"
            if _has_final_verdicts(user_claims):
                reason = "Claims are resolved, but there is no retrieved policy grounding for a formal decision."
            else:
                reason = "More information or user interaction is still needed before final resolution."

    return {
        "execution_plan": plan,
        "next_node": next_node,
        "planner_reason": reason,
        "claim_state_updated_this_turn": state_updates.get("claim_state_updated_this_turn", claim_state_updated_this_turn),
        **state_updates,
    }
