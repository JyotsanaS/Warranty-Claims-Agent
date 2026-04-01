"""
Evidence planner node — for each policy-valid claim, generates a list of
VisualCheck objects describing what to look for in the uploaded image.
"""
from __future__ import annotations

from agent.services.evidence_service import plan_visual_checks
from models.state import AgentState


def evidence_planner_node(state: AgentState) -> dict:
    user_claims: list[dict] = list(state.get("user_claims", []))
    updated = False

    for i, claim in enumerate(user_claims):
        # Only plan for covered claims that have no checks yet
        coverage = claim.get("policy_coverage") or {}
        if not coverage.get("covered"):
            continue
        if claim.get("visual_checks"):
            continue

        component = claim.get("component", "unknown component")
        statement = claim.get("verbatim_statement", "")
        incident_type = ""
        claim_items = state.get("claim_items", [])
        if claim_items:
            active = state.get("active_claim_index", 0)
            if active < len(claim_items):
                incident_type = claim_items[active].get("incident_type", "")

        checks = plan_visual_checks(
            component=component,
            issue_summary=statement or incident_type or "unspecified damage",
        )

        if checks:
            user_claims[i] = dict(claim)
            user_claims[i]["visual_checks"] = checks
            updated = True

    if updated:
        return {"user_claims": user_claims}
    return {}
