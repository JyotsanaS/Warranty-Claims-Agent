"""
Claim decision node — emits the final verdict with policy clause citations.
Runs only when all user claims have a non-pending verdict.
"""
from __future__ import annotations

import logging

from gateway.llm_gateway import main_llm
from agent.prompt_store import get_prompt
from models.state import AgentState

logger = logging.getLogger(__name__)


def _build_templated_decision(user_claims: list[dict], policy_clauses: list[str]) -> str:
    """
    Deterministic fallback prose when LLM output fails the hallucination check twice.
    Verdict is never changed — only the explanation is replaced with pre-verified data.
    """
    lines = ["Based on our policy review, here is your claim outcome:"]
    for claim in user_claims:
        component = claim.get("component", "claim")
        verdict = (claim.get("claim_verdict") or "pending").upper()
        coverage = claim.get("policy_coverage") or {}
        clauses = coverage.get("policy_clauses", [])
        line = f"- {component}: {verdict}"
        if clauses:
            line += f" (policy reference: {', '.join(clauses)})"
        elif verdict == "REJECTED" and coverage.get("exclusion_reason"):
            line += f" — {coverage['exclusion_reason']}"
        lines.append(line)
    if policy_clauses:
        lines.append(f"\nApplicable policy clauses: {', '.join(policy_clauses)}")
    return "\n".join(lines)


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

    system_content = get_prompt("claim_decision") + "\n\n" + "\n".join(brief_lines)

    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                "Please issue the final claim decision. "
                f"Use case reference: VE-{state.get('session_id', 'UNKNOWN')[-8:].upper()}."
            ),
        },
    ]

    text = main_llm(messages, temperature=0.2)
    if not text:
        text = "Your claim has been reviewed."

    # Hallucination check — guard the customer-facing prose against ungrounded clause citations
    reference = "\n---\n".join(policy_context[:3]) if policy_context else ""
    if reference and text:
        from agent.guardrails.hallucination import is_grounded
        grounded = is_grounded(
            value=text,
            reference=reference,
            query="Final warranty claim decision",
        )
        if not grounded:
            logger.warning("Claim decision hallucination detected — retrying with stricter prompt")
            stricter_system = (
                system_content
                + f"\n\nIMPORTANT: Cite only the following pre-verified policy clauses: "
                f"{policy_clauses}. Do not reference any other clause names."
            )
            retry_messages = [
                {"role": "system", "content": stricter_system},
                messages[1],
            ]
            retry_text = main_llm(retry_messages, temperature=0.2)
            if retry_text:
                grounded_retry = is_grounded(
                    value=retry_text,
                    reference=reference,
                    query="Final warranty claim decision",
                )
                if grounded_retry:
                    text = retry_text
                else:
                    logger.warning(
                        "Claim decision hallucination persists after retry — using templated fallback"
                    )
                    text = _build_templated_decision(user_claims, policy_clauses)
            else:
                text = _build_templated_decision(user_claims, policy_clauses)

    recorded_lines: list[str] = []
    case_reference = f"VE-{state.get('session_id', 'UNKNOWN')[-8:].upper()}"
    if len(user_claims) == 1:
        claim = user_claims[0]
        component = claim.get("component", "claim")
        verdict = (claim.get("claim_verdict") or "pending").upper()
        recorded_lines.append(
            f"Final status: We are marking your {component} claim as {verdict} in our system."
        )
        recorded_lines.append("This status has been recorded for your case.")
    elif user_claims:
        recorded_lines.append("Final status: We have recorded the following claim outcomes in our system.")
        for claim in user_claims:
            component = claim.get("component", "claim")
            verdict = (claim.get("claim_verdict") or "pending").upper()
            recorded_lines.append(f"- {component}: {verdict}")

    verdicts = [c.get("claim_verdict") for c in user_claims]
    if all(v == "approved" for v in verdicts if v):
        overall_status = "APPROVED"
    elif any(v == "rejected" for v in verdicts):
        overall_status = "REJECTED"
    elif any(v == "escalated" for v in verdicts):
        overall_status = "ESCALATED"
    else:
        overall_status = "PENDING"

    footer_lines = [f"Case reference: {case_reference}"]
    if overall_status == "APPROVED":
        footer_lines.append("Next step: Our agent will contact you in the next 3 hours.")
    elif overall_status == "REJECTED":
        footer_lines.append("Next step: This claim has been closed as rejected in our system.")
    elif overall_status == "ESCALATED":
        footer_lines.append("Next step: This claim has been escalated for further review.")

    footer_lines.extend(
        [
            "Please rate this resolution from 1 to 5:",
            "1 - very poor resolution",
            "2 - poor resolution",
            "3 - fair resolution",
            "4 - good resolution",
            "5 - best resolution",
        ]
    )

    if recorded_lines:
        text = "\n".join(recorded_lines) + "\n\n" + text
    text = text.rstrip() + "\n\n" + "\n".join(footer_lines)

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
        "user_claims": user_claims,
        "awaiting_post_resolution_followup": True,
    }
