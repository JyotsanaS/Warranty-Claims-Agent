"""
Vision analysis node — sends the uploaded image with all VisualChecks to the
vision LLM in a single call and fills in the findings.
"""
from __future__ import annotations

import logging

from agent.services.vision_service import analyze_image
from models.state import AgentState

logger = logging.getLogger(__name__)


def vision_analysis_node(state: AgentState) -> dict:
    image_path = state.get("image_local_path")
    if not image_path:
        return {}

    user_claims: list[dict] = list(state.get("user_claims", []))

    # Collect all visual checks across claims
    all_checks: list[dict] = []
    for claim in user_claims:
        all_checks.extend(claim.get("visual_checks", []))

    service_result = analyze_image(image_path=image_path, checks=all_checks)

    # Distribute check results back to user_claims
    result_checks: list[dict] = service_result.get("checks", [])

    if not result_checks:
        logger.warning(
            "Vision service returned no check results for session %s — "
            "skipping claim update to avoid falsely rejecting claims",
            state.get("session_id"),
        )
        return {"damage_report": service_result.get("damage_report", {})}

    check_by_desc: dict[str, dict] = {c["check_description"]: c for c in result_checks}

    for i, claim in enumerate(user_claims):
        updated_checks = []
        for vc in claim.get("visual_checks", []):
            desc = vc.get("check_description", "")
            check_result = check_by_desc.get(desc, {})
            merged = dict(vc)
            merged.update({
                "finding": check_result.get("finding", ""),
                "observation": check_result.get("observation", ""),
                "claim_supported": bool(check_result.get("claim_supported", False)),
                "confidence": float(check_result.get("confidence", 0.0)),
            })
            updated_checks.append(merged)
        user_claims[i] = dict(claim)
        user_claims[i]["visual_checks"] = updated_checks

    return {
        "user_claims": user_claims,
        "damage_report": service_result.get("damage_report", {}),
    }
