"""
Simple agent nodes that require minimal or no LLM calls:
  greeting_node, fallback_node, status_node,
  escalation_node, cancellation_node, confirmation_handler, post_resolution_close_node,
  prompt_injection_node
"""
from __future__ import annotations

import logging

from gateway.llm_gateway import main_llm
from agent.prompt_store import get_prompt
from models.state import AgentState

logger = logging.getLogger(__name__)

_NEGATIVE_REPLY_MARKERS = {
    "no",
    "nope",
    "nah",
    "nothing",
    "nothing else",
    "no thanks",
    "not now",
}


def is_negative_reply(text: str) -> bool:
    normalized = " ".join(str(text).strip().lower().split())
    return normalized in _NEGATIVE_REPLY_MARKERS


def greeting_node(state: AgentState) -> dict:
    last_user = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), ""
    )
    system_prompt = (
        get_prompt("simple_nodes", "initial_greeting_system")
        if state.get("awaiting_first_user_turn")
        else get_prompt("simple_nodes", "standard_greeting_system")
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": last_user},
    ]
    text = main_llm(messages)
    return {"messages": [{"role": "assistant", "content": text}]}


def fallback_node(state: AgentState) -> dict:
    last_user = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), ""
    )
    messages = [
        {"role": "system", "content": get_prompt("simple_nodes", "fallback_system")},
        {"role": "user", "content": last_user},
    ]
    text = main_llm(messages)
    return {"messages": [{"role": "assistant", "content": text}]}


# ── Status ────────────────────────────────────────────────────────────────────

def status_node(state: AgentState) -> dict:
    claim_items: list[dict] = state.get("claim_items", [])
    if not claim_items:
        text = (
            "I don't have any active claims on record for this session yet. "
            "Please describe the issue you'd like to file a claim for."
        )
    else:
        lines = ["Here is your current claim status:\n"]
        for i, ctx in enumerate(claim_items, 1):
            comp = ctx.get("component") or "Unknown component"
            status = ctx.get("claim_status", "open")
            lines.append(f"{i}. **{comp}** — status: *{status}*")
        text = "\n".join(lines)

    return {"messages": [{"role": "assistant", "content": text}]}


# ── Escalation ────────────────────────────────────────────────────────────────

def escalation_node(state: AgentState) -> dict:
    claim_items: list[dict] = list(state.get("claim_items", []))
    active = state.get("active_claim_index", 0)

    if claim_items:
        ctx = dict(claim_items[active])
        ctx["claim_status"] = "escalated"
        claim_items[active] = ctx

    text = (
        "I'm sorry to hear you're having a difficult experience. "
        "I've flagged your case for escalation to our specialist team. "
        "A human agent will review your claim and reach out within 1 business day. "
        "Your case reference has been saved."
    )
    return {
        "claim_items": claim_items,
        "messages": [{"role": "assistant", "content": text}],
    }


# ── Cancellation ──────────────────────────────────────────────────────────────

def cancellation_node(state: AgentState) -> dict:
    claim_items: list[dict] = list(state.get("claim_items", []))
    active = state.get("active_claim_index", 0)

    if claim_items:
        ctx = dict(claim_items[active])
        ctx["claim_status"] = "cancelled"
        ctx.setdefault("audit_notes", []).append("Cancelled by user request.")
        claim_items[active] = ctx

    text = (
        "Your claim has been cancelled as requested. "
        "If you change your mind or have a different issue, feel free to start a new claim."
    )
    return {
        "claim_items": claim_items,
        "messages": [{"role": "assistant", "content": text}],
    }


# ── Confirmation handler (context switch) ─────────────────────────────────────

def confirmation_handler_node(state: AgentState) -> dict:
    """
    Resolves a pending context switch.
    """
    claim_items: list[dict] = list(state.get("claim_items", []))
    pending_draft: dict | None = state.get("pending_claim_draft")
    active = state.get("active_claim_index", 0)
    last_user = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), ""
    ).lower()

    separate_markers = ("separate", "new claim", "another claim", "different claim")
    same_markers = ("same", "same incident", "together", "one claim")

    if any(token in last_user for token in separate_markers):
        if pending_draft:
            claim_items.append(dict(pending_draft))
            new_index = len(claim_items) - 1
        else:
            new_index = active
        text = (
            "Understood. I'll treat that as a separate claim and continue with the new issue."
        )
        return {
            "claim_items": claim_items,
            "active_claim_index": new_index,
            "pending_switch_confirmation": False,
            "pending_claim_draft": None,
            "messages": [{"role": "assistant", "content": text}],
        }

    if any(token in last_user for token in same_markers):
        if pending_draft and claim_items and 0 <= active < len(claim_items):
            ctx = dict(claim_items[active])
            draft_component = pending_draft.get("component")
            if draft_component and draft_component != ctx.get("component"):
                ctx.setdefault("audit_notes", []).append(
                    f"Additional related issue reported: {draft_component}"
                )
            for field in ("incident_type", "incident_date", "purchase_date"):
                if pending_draft.get(field) and not ctx.get(field):
                    ctx[field] = pending_draft[field]
            if pending_draft.get("has_receipt"):
                ctx["has_receipt"] = True
            if pending_draft.get("has_image"):
                ctx["has_image"] = True
            claim_items[active] = ctx
        text = (
            "Understood. I'll keep this under the same claim and continue from the current case."
        )
        return {
            "claim_items": claim_items,
            "pending_switch_confirmation": False,
            "pending_claim_draft": None,
            "messages": [{"role": "assistant", "content": text}],
        }

    text = (
        "Please confirm whether the additional issue should be handled as the same claim "
        "or as a separate claim."
    )
    return {
        "pending_switch_confirmation": True,
        "pending_claim_draft": pending_draft,
        "messages": [{"role": "assistant", "content": text}],
    }


def post_resolution_close_node(state: AgentState) -> dict:
    text = (
        "Thanks for reaching out to VoltEdge warranty support. "
        "Before we close, please rate your experience:\n\n"
        "**1.** 👍 Like\n"
        "**2.** 👎 Dislike"
    )
    return {
        "awaiting_post_resolution_followup": False,
        "awaiting_feedback": True,
        "messages": [{"role": "assistant", "content": text}],
    }


def feedback_node(state: AgentState) -> dict:
    last_user = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), ""
    ).strip()

    normalized = last_user.lower()
    if normalized in {"1", "like", "👍", "thumbs up", "good", "great"}:
        feedback = "like"
        text = "Thank you for the positive feedback! 😊 We're glad we could help. Have a great day!"
    elif normalized in {"2", "dislike", "👎", "thumbs down", "bad", "poor"}:
        feedback = "dislike"
        text = "Thank you for your feedback. We're sorry the experience wasn't ideal — we'll use this to improve. Have a great day!"
    else:
        # Unrecognised response — ask again
        text = (
            "Sorry, I didn't catch that. Please reply with:\n\n"
            "**1.** 👍 Like\n"
            "**2.** 👎 Dislike"
        )
        return {"messages": [{"role": "assistant", "content": text}]}

    return {
        "awaiting_feedback": False,
        "user_feedback": feedback,
        "messages": [{"role": "assistant", "content": text}],
    }


def prompt_injection_node(state: AgentState) -> dict:
    text = (
        "Sorry, we believe this request is attempting to manipulate policy or claim handling. "
        "We are closing this session for now."
    )
    return {
        "terminate_session": True,
        "messages": [{"role": "assistant", "content": text}],
    }
