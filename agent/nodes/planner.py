"""
Planner node — hybrid router.

Layer 1  Hard rules     : mechanical state flags — always win, no LLM needed
Layer 2  Simple intents : direct intent → node map, no LLM needed
Layer 3  LLM planner    : claim-flow reasoning via fast LLM
"""
from __future__ import annotations

import json
import logging

from agent.prompt_store import get_prompt
from gateway.llm_gateway import fast_llm
from models.state import AgentState

logger = logging.getLogger(__name__)

_SIMPLE_INTENT_ROUTES: dict[str, str] = {
    "greeting": "greeting_node",
    "out_of_scope": "fallback_node",
    "escalation": "escalation_node",
    "cancellation": "cancellation_node",
    "status_query": "status_node",
}

_VALID_LLM_NODES = frozenset({
    "claim_state_updater",
    "policy_checker",
    "evidence_planner",
    "vision_analysis",
    "agent_respond",
    "claim_decision",
})


# ── State inspection helpers (also used by tests) ────────────────────────────

def _has_unassessed_claims(user_claims: list[dict]) -> bool:
    return any(c.get("policy_coverage") is None for c in user_claims)


def _has_final_verdicts(user_claims: list[dict]) -> bool:
    if not user_claims:
        return False
    final = {"approved", "rejected", "escalated"}
    return all(c.get("claim_verdict") in final for c in user_claims)


def _has_policy_grounding(
    user_claims: list[dict],
    policy_clauses: list[str],
    policy_context: list[str],
) -> bool:
    if policy_clauses or policy_context:
        return True
    for claim in user_claims:
        coverage = claim.get("policy_coverage") or {}
        if coverage.get("policy_clauses"):
            return True
    return False


# ── LLM planner (Layer 3) ─────────────────────────────────────────────────────

def _build_state_summary(state: AgentState) -> str:
    user_claims: list[dict] = state.get("user_claims", [])
    claims_lines: list[str] = []
    for c in user_claims:
        comp = c.get("component", "unknown")
        verdict = c.get("claim_verdict") or "none"
        coverage = c.get("policy_coverage") or {}
        covered = coverage.get("covered")
        exclusion = coverage.get("exclusion_reason", "")
        clauses = ", ".join(coverage.get("policy_clauses", []))
        if c.get("policy_coverage") is None:
            assessment = "not assessed"
        else:
            assessment = f"covered={covered}"
            if exclusion:
                assessment += f", exclusion={exclusion}"
            if clauses:
                assessment += f", clauses=[{clauses}]"
        visual_checks = c.get("visual_checks", [])
        claims_lines.append(
            f"  - {comp}: {assessment}, verdict={verdict}, visual_checks={len(visual_checks)}"
        )

    unassessed = sum(1 for c in user_claims if c.get("policy_coverage") is None)
    policy_chunks = len(state.get("policy_context", []))
    has_image = bool(state.get("image_local_path"))
    has_visual_checks = any(c.get("visual_checks") for c in user_claims)
    covered_awaiting_evidence = sum(
        1 for c in user_claims
        if (c.get("policy_coverage") or {}).get("covered") and not c.get("visual_checks")
        and c.get("claim_verdict") is None
    )
    claims_with_pending_checks = sum(
        1 for c in user_claims
        if any(not ch.get("finding") for ch in c.get("visual_checks", []))
    )

    # Explicitly signal whether the formal decision letter is ready to send
    verdict_explanation_given = any(
        m.get("role") == "assistant" and (
            "excluded" in str(m.get("content", "")).lower()
            or "not covered" in str(m.get("content", "")).lower()
            or "covered" in str(m.get("content", "")).lower()
        )
        for m in state.get("messages", [])
    )
    ready_for_decision = (
        bool(user_claims)
        and unassessed == 0
        and _has_final_verdicts(user_claims)
        and policy_chunks > 0
        and verdict_explanation_given
    )

    lines = [
        f"intent: {state.get('intent', 'unknown')}",
        f"user_claims_total: {len(user_claims)}",
        f"user_claims_unassessed: {unassessed}",
        f"covered_claims_awaiting_evidence_plan: {covered_awaiting_evidence}",
        f"ready_for_formal_decision: {ready_for_decision}",
    ]
    if claims_lines:
        lines.append("claims:")
        lines.extend(claims_lines)
    lines += [
        f"policy_chunks_available: {policy_chunks}",
        f"image_available: {has_image}",
        f"claims_with_pending_visual_checks: {claims_with_pending_checks}",
    ]
    return "\n".join(lines)


def _llm_planner(state: AgentState) -> tuple[str, str]:
    """Ask the fast LLM which node to run next for claim-related intents."""
    state_summary = _build_state_summary(state)

    history = [
        m for m in state.get("messages", [])
        if m.get("role") in ("user", "assistant")
    ][-10:]
    conversation = "\n".join(
        f"{m['role'].upper()}: {str(m.get('content', ''))[:400]}"
        for m in history
    )

    messages = [
        {"role": "system", "content": get_prompt("planner")},
        {
            "role": "user",
            "content": f"CURRENT STATE:\n{state_summary}\n\nRECENT CONVERSATION:\n{conversation}",
        },
    ]

    raw = fast_llm(messages, json_mode=True)
    try:
        parsed = json.loads(raw)
        next_node = str(parsed.get("next_node", "")).strip()
        reason = str(parsed.get("reason", "")).strip()
        if next_node in _VALID_LLM_NODES:
            return next_node, reason
        logger.warning("LLM planner returned invalid node %r — falling back to agent_respond", next_node)
    except Exception as exc:
        logger.warning("LLM planner parse error: %s — falling back to agent_respond", exc)

    return "agent_respond", "Fallback: LLM planner output could not be parsed."


# ── Planner node ──────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    intent = state.get("intent", "")

    # ── Layer 1: Hard rules — mechanical state flags, always win ─────────────
    if state.get("awaiting_feedback"):
        return {
            "execution_plan": ["Collect user feedback and close conversation"],
            "next_node": "feedback_node",
            "planner_reason": "Awaiting like/dislike feedback from user.",
        }

    if state.get("pending_switch_confirmation"):
        return {
            "execution_plan": ["Resolve pending claim context switch"],
            "next_node": "confirmation_handler",
            "planner_reason": "Pending context switch confirmation takes priority.",
        }

    if state.get("awaiting_post_resolution_followup"):
        return {
            "execution_plan": ["Close the resolved support conversation"],
            "next_node": "post_resolution_close_node",
            "planner_reason": "Formal decision given; collecting feedback and closing session.",
        }

    if intent == "prompt_injection":
        return {
            "execution_plan": ["Block and close session"],
            "next_node": "prompt_injection_node",
            "planner_reason": "Prompt injection detected.",
        }

    # ── Layer 2: Simple intent routes — direct map, no LLM needed ────────────
    if intent in _SIMPLE_INTENT_ROUTES:
        node = _SIMPLE_INTENT_ROUTES[intent]
        return {
            "execution_plan": [f"Handle {intent.replace('_', ' ')} request directly"],
            "next_node": node,
            "planner_reason": f"Intent '{intent}' maps directly to {node}.",
        }

    # ── Layer 3: LLM planner for claim-related intents ────────────────────────
    next_node, reason = _llm_planner(state)
    return {
        "execution_plan": [reason],
        "next_node": next_node,
        "planner_reason": reason,
    }
