"""
Router node — classifies the current user message using recent conversation
and compact structured memory for context.
Short-circuits to confirmation_handler if a context-switch is pending.
"""
from __future__ import annotations

import json
import logging

from agent.prompt_store import get_prompt
from gateway.llm_gateway import fast_llm
from models.state import AgentState

logger = logging.getLogger(__name__)

_INTENTS = (
    "greeting",       # hello / hi / how are you
    "out_of_scope",   # unrelated to warranty or claims
    "escalation",     # wants human agent, very upset
    "cancellation",   # wants to cancel the claim
    "status_query",   # asking about current claim status
    "frustration",    # frustrated but not escalating
    "claim",          # filing / submitting a new claim
    "policy",         # asking about coverage or policy details
    "issue",          # describing a product problem
    "evidence",       # providing a photo, receipt, or other proof
    "clarification",  # answering a follow-up question
)


def _build_recent_conversation(messages: list[dict], max_turns: int = 4) -> str:
    recent = [m for m in messages if m.get("role") in ("user", "assistant")]
    recent = recent[-(max_turns * 2):]
    lines = [
        f"{str(msg.get('role', '')).upper()}: {str(msg.get('content', '')).strip()}"
        for msg in recent
        if str(msg.get("content", "")).strip()
    ]
    return "\n".join(lines) if lines else "None"


def _build_structured_memory(state: AgentState, max_claims: int = 3) -> str:
    memory_lines: list[str] = []
    claim_items: list[dict] = list(state.get("claim_items", []))
    active_index = state.get("active_claim_index", 0)

    if claim_items and 0 <= active_index < len(claim_items):
        active = claim_items[active_index]
        memory_lines.append("Active claim:")
        memory_lines.append(
            f"- component={active.get('component') or 'unknown'}; "
            f"incident_type={active.get('incident_type') or 'unknown'}; "
            f"status={active.get('claim_status') or 'open'}; "
            f"has_image={bool(active.get('has_image'))}; "
            f"has_receipt={bool(active.get('has_receipt'))}"
        )
    else:
        memory_lines.append("Active claim: none")

    memory_lines.append(
        f"Pending context switch confirmation: {bool(state.get('pending_switch_confirmation'))}"
    )

    user_claims: list[dict] = list(state.get("user_claims", []))
    if user_claims:
        memory_lines.append("Known user claims:")
        for claim in user_claims[:max_claims]:
            coverage = claim.get("policy_coverage") or {}
            covered = (
                coverage.get("covered")
                if claim.get("policy_coverage") is not None
                else "unknown"
            )
            memory_lines.append(
                f"- component={claim.get('component') or 'unknown'}; "
                f"type={claim.get('assertion_type') or 'unknown'}; "
                f"verdict={claim.get('claim_verdict') or 'unresolved'}; "
                f"covered={covered}"
            )
    else:
        memory_lines.append("Known user claims: none")

    return "\n".join(memory_lines)


def router_node(state: AgentState) -> dict:
    # If awaiting feedback, bypass LLM classification
    if state.get("awaiting_feedback"):
        return {
            "intent": "feedback_response",
            "router_confidence": 1.0,
        }

    # If a context switch is already pending, bypass LLM classification
    if state.get("pending_switch_confirmation"):
        return {
            "intent": "pending_switch",
            "router_confidence": 1.0,
        }

    history = state["messages"]
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    recent_conversation = _build_recent_conversation(history, max_turns=4)
    structured_memory = _build_structured_memory(state)
    user_content = (
        f"Recent conversation:\n{recent_conversation}\n\n"
        f"Structured memory:\n{structured_memory}\n\n"
        f"Current user message to classify:\n{last_user}"
    )

    messages = [
        {
            "role": "system",
            "content": get_prompt("router").format(intents=", ".join(_INTENTS)),
        },
        {"role": "user", "content": user_content},
    ]

    raw = fast_llm(messages, json_mode=True)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Router JSON parse failed, defaulting to 'issue'")
        data = {}

    intent = data.get("intent", "issue")
    if intent not in _INTENTS:
        intent = "issue"

    return {
        "intent": intent,
        "router_confidence": float(data.get("confidence", 0.5)),
    }
