"""
Agent respond node — generates the primary user-facing response.
Prepends empathy_prefix if set. Asks for missing evidence when needed.
"""
from __future__ import annotations

import logging

from gateway.llm_gateway import main_llm
from models.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are VoltEdge's warranty claims agent. Be concise, warm, and professional.

Guidelines:
- Write your response in plain, conversational language. Do NOT mention policy sections or clause
  numbers inline in your response text.
- If a claim has been assessed as covered, acknowledge it clearly and warmly.
- If a claim is excluded, explain the reason in plain language without citing clauses inline.
- If you need more information (purchase date, photo, receipt), ask for ONE piece at a time.
- If an image was analysed, summarise what was found and how it affects the claim.
- Never invent policy details that were not provided to you.
- Keep your response under 200 words unless a detailed explanation is needed.
- If policy sections were consulted and are relevant to the outcome, append them at the very end
  under a separate heading exactly as shown:

CITATIONS:
- §X Title

  Only include this section when policy context was actually retrieved. Omit it entirely otherwise.
"""


def agent_respond_node(state: AgentState) -> dict:
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

    system_content = _SYSTEM
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

    return {
        "messages": [{"role": "assistant", "content": text}],
        "empathy_prefix": "",   # Clear after use
    }
