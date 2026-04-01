"""
Simple agent nodes that require minimal or no LLM calls:
  greeting_node, fallback_node, status_node,
  escalation_node, cancellation_node, confirmation_handler
"""
from __future__ import annotations

import logging

from gateway.llm_gateway import main_llm
from models.state import AgentState

logger = logging.getLogger(__name__)


# ── Greeting ──────────────────────────────────────────────────────────────────

_GREETING_SYSTEM = """\
You are VoltEdge's warranty claims agent. The user is greeting you.
Respond warmly and briefly. Let them know you can help with warranty claims,
coverage questions, and claim status. Keep your reply to 2–3 sentences.
"""


def greeting_node(state: AgentState) -> dict:
    last_user = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), ""
    )
    messages = [
        {"role": "system", "content": _GREETING_SYSTEM},
        {"role": "user", "content": last_user},
    ]
    text = main_llm(messages)
    return {"messages": [{"role": "assistant", "content": text}]}


# ── Fallback (out of scope) ────────────────────────────────────────────────────

_FALLBACK_SYSTEM = """\
You are VoltEdge's warranty claims agent. The user's message is outside your scope.
Politely explain that you can only assist with warranty claims and coverage questions.
Keep your reply to 2 sentences. Do not apologise excessively.
"""


def fallback_node(state: AgentState) -> dict:
    last_user = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), ""
    )
    messages = [
        {"role": "system", "content": _FALLBACK_SYSTEM},
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
    For MVP: auto-approve the switch and start a new claim item.
    """
    claim_items: list[dict] = list(state.get("claim_items", []))
    # Create a new empty claim slot — context_extractor will populate it next turn
    from models.state import ClaimContext
    new_ctx = ClaimContext()
    claim_items.append(new_ctx.model_dump())
    new_index = len(claim_items) - 1

    text = (
        "Got it — I'll start a new claim for the additional issue you mentioned. "
        "Could you describe what's wrong with it?"
    )
    return {
        "claim_items": claim_items,
        "active_claim_index": new_index,
        "pending_switch_confirmation": False,
        "messages": [{"role": "assistant", "content": text}],
    }
