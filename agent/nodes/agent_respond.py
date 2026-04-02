"""
Agent respond node — generates the primary user-facing response.
Prepends empathy_prefix if set. Asks for missing evidence when needed.
"""
from __future__ import annotations

import logging

from gateway.llm_gateway import main_llm
from agent.prompt_store import get_prompt
from models.state import AgentState
from observability.tracing import start_span

logger = logging.getLogger(__name__)


def _all_claims_rejected(user_claims: list[dict]) -> bool:
    return bool(user_claims) and all(c.get("claim_verdict") == "rejected" for c in user_claims)


def _build_rejection_followup(user_claims: list[dict]) -> str:
    components = [str(c.get("component", "")).strip() for c in user_claims]
    components = [c for c in components if c]
    if not components:
        subject = "this claim"
    elif len(components) == 1:
        subject = f"the claim for your {components[0]}"
    else:
        subject = "these claims"

    return (
        f"Sorry, we cannot accept {subject} for claim processing. "
        "Can I do something else for you?"
    )


def agent_respond_node(state: AgentState) -> dict:
    with start_span("node.agent_respond") as span:
        # Build system message with all available context
        context_parts: list[str] = []

        policy_context = state.get("policy_context", [])
        if policy_context:
            context_parts.append("Relevant policy sections:\n" + "\n---\n".join(policy_context))

        user_claims = state.get("user_claims", [])
        if user_claims:
            claims_summary = []
            for c in user_claims:
                comp = c.get("component", "Unknown")
                verdict = c.get("claim_verdict") or "not yet assessed"
                coverage = c.get("policy_coverage") or {}
                covered_str = "covered" if coverage.get("covered") else "not covered"
                clauses = ", ".join(coverage.get("policy_clauses", []))
                claims_summary.append(
                    f"- {comp}: {covered_str} ({clauses}), verdict: {verdict}"
                )
            context_parts.append("Current claim assessment:\n" + "\n".join(claims_summary))

        damage_report = state.get("damage_report", {})
        if damage_report and damage_report.get("image_quality"):
            context_parts.append(
                f"Image analysis: quality={damage_report.get('image_quality')}, "
                f"confidence={damage_report.get('overall_confidence', 0):.0%}"
            )

        span.set_attribute("rag.policy_chunk_count", len(policy_context))
        span.set_attribute("rag.policy_chunks_preview", [c[:200] for c in policy_context])

        system_content = get_prompt("agent_respond")
        if context_parts:
            system_content += "\n\n" + "\n\n".join(context_parts)

        # Prepend empathy prefix to system instructions if set
        empathy_prefix = state.get("empathy_prefix", "").strip()
        if empathy_prefix:
            system_content += f"\n\nBegin your response with: \"{empathy_prefix}, and ...\""

        # Build the message list (last 10 turns)
        history = [m for m in state["messages"] if m["role"] in ("user", "assistant")][-10:]
        messages = [{"role": "system", "content": system_content}] + history

        text = main_llm(messages)
        if not text:
            text = "I'm sorry, I encountered an issue generating a response. Please try again."

        span.set_attribute("response.length", len(text))
        return {
            "messages": [{"role": "assistant", "content": text}],
            "empathy_prefix": "",   # Clear after use
        }
