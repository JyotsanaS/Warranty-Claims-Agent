"""
Evidence planner node — for each policy-valid claim, generates a list of
VisualCheck objects describing what to look for in the uploaded image.
"""
from __future__ import annotations

import json
import logging

from gateway.llm_gateway import fast_llm
from models.state import AgentState, VisualCheck

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an evidence-planning agent for a warranty claims department.
Given a component and the nature of the damage, generate a list of visual checks
that an image reviewer should perform.

Return JSON only:
{
  "checks": [
    {
      "check_description": "<what to look for>",
      "expected_finding": "<what a genuine defect looks like>"
    }
  ]
}

Generate 2–4 checks. Be specific and objective. Focus on:
- Physical damage indicators (cracks, burns, corrosion, missing parts)
- Manufacturing defect signs
- Signs of misuse or external damage that would void the warranty
"""


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

        prompt = (
            f"Component: {component}\n"
            f"Customer issue: {statement or incident_type or 'unspecified damage'}"
        )
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ]

        raw = fast_llm(messages, json_mode=True)
        checks: list[dict] = []
        try:
            data = json.loads(raw)
            raw_checks = data.get("checks", [])
            for rc in raw_checks:
                vc = VisualCheck(
                    check_description=rc.get("check_description", ""),
                    expected_finding=rc.get("expected_finding", ""),
                )
                checks.append(vc.model_dump())
        except Exception:
            logger.warning("Evidence planner parse failed for %s", component)
            # Fallback: one generic check
            checks = [
                VisualCheck(
                    check_description="Overall damage assessment",
                    expected_finding="Visible physical damage consistent with reported issue",
                ).model_dump()
            ]

        if checks:
            user_claims[i] = dict(claim)
            user_claims[i]["visual_checks"] = checks
            updated = True

    if updated:
        return {"user_claims": user_claims}
    return {}
