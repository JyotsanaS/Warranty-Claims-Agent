"""
Claim decision node — emits the final verdict with policy clause citations.
Runs only when all user claims have a non-pending verdict.
"""
from __future__ import annotations

import logging

from gateway.llm_gateway import main_llm
from models.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are VoltEdge's warranty claims agent issuing a final decision.

You will be given the full claim assessment. Write a clear, empathetic final decision letter.

Structure:
1. Brief summary of each claim and its verdict (approved / rejected / escalated).
2. For each approved claim: next steps (replacement / repair process).
3. For each rejected claim: the specific policy clause that excludes it and why.
4. Closing: reference number placeholder and next-steps contact.

Tone: professional, empathetic, clear. Maximum 300 words.
Always cite policy sections (e.g. §2, §3.1) when explaining decisions.
"""


def claim_decision_node(state: AgentState) -> dict:
    user_claims = state.get("user_claims", [])
    policy_clauses = state.get("policy_clauses", [])
    policy_context = state.get("policy_context", [])
    claim_items = state.get("claim_items", [])

    # Build decision brief for the LLM
    brief_lines: list[str] = ["Claim assessment summary:"]
    for c in user_claims:
        comp = c.get("component", "Unknown")
        verdict = c.get("claim_verdict", "pending")
        coverage = c.get("policy_coverage") or {}
        clauses = ", ".join(coverage.get("policy_clauses", []))
        exclusion = coverage.get("exclusion_reason", "")
        brief_lines.append(
            f"- {comp}: verdict={verdict}, clauses={clauses or 'N/A'}"
            + (f", exclusion: {exclusion}" if exclusion else "")
        )

    if policy_context:
        brief_lines.append("\nRelevant policy sections:\n" + "\n---\n".join(policy_context[:3]))

    system_content = _SYSTEM + "\n\n" + "\n".join(brief_lines)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "Please issue the final claim decision."},
    ]

    text = main_llm(messages, temperature=0.2)
    if not text:
        text = "Your claim has been reviewed. Please contact support for the final decision."

    # Update claim_items status based on verdicts
    updated_claim_items = list(claim_items)
    verdict_map: dict[str, str] = {}
    for c in user_claims:
        comp = c.get("component", "").lower()
        v = c.get("claim_verdict", "pending")
        if v in ("approved", "rejected", "escalated"):
            verdict_map[comp] = v

    for i, ctx in enumerate(updated_claim_items):
        comp = ctx.get("component", "").lower()
        if comp in verdict_map:
            ctx = dict(ctx)
            ctx["claim_status"] = verdict_map[comp]
            updated_claim_items[i] = ctx

    return {
        "messages": [{"role": "assistant", "content": text}],
        "claim_items": updated_claim_items,
    }
