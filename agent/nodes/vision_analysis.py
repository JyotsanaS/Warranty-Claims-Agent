"""
Vision analysis node — sends the uploaded image with all VisualChecks to the
vision LLM in a single call and fills in the findings.
"""
from __future__ import annotations

import base64
import json
import logging

from gateway.llm_gateway import vision_llm
from models.state import AgentState, VisualCheck
from storage.image_store import read_image

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an image analysis agent for a warranty claims department.
You will receive an image and a list of visual checks to perform.
Analyse the image carefully and fill in the results for each check.

Return JSON only:
{
  "image_quality": "<good|poor|unreadable>",
  "image_type": "<product_photo|receipt|serial_number|other>",
  "overall_confidence": <0.0-1.0>,
  "extracted_purchase_date": "<YYYY-MM-DD or null>",
  "extracted_serial_number": "<string or null>",
  "checks": [
    {
      "check_description": "<same as input>",
      "expected_finding": "<same as input>",
      "finding": "<what you actually see in the image>",
      "observation": "<detailed observation>",
      "claim_supported": <true|false>,
      "confidence": <0.0-1.0>
    }
  ]
}
"""


def vision_analysis_node(state: AgentState) -> dict:
    image_path = state.get("image_local_path")
    if not image_path:
        return {}

    user_claims: list[dict] = list(state.get("user_claims", []))

    # Collect all visual checks across claims
    all_checks: list[dict] = []
    for claim in user_claims:
        all_checks.extend(claim.get("visual_checks", []))

    if not all_checks:
        # No checks planned — do a generic damage assessment
        all_checks = [
            VisualCheck(
                check_description="General product condition",
                expected_finding="Visible damage, defect, or malfunction indicator",
            ).model_dump()
        ]

    # Load and encode image
    try:
        raw_bytes = read_image(image_path)
        b64 = base64.b64encode(raw_bytes).decode()
    except Exception as exc:
        logger.error("Vision: failed to read image %s: %s", image_path, exc)
        return {"damage_report": {"error": str(exc)}}

    checks_json = json.dumps(all_checks, indent=2)
    user_content = [
        {
            "type": "text",
            "text": (
                f"Please perform the following visual checks on this image:\n\n{checks_json}"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
    ]

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_content},
    ]

    raw = vision_llm(messages)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Vision analysis JSON parse failed")
        data = {
            "image_quality": "poor",
            "image_type": "other",
            "overall_confidence": 0.0,
            "checks": [],
        }

    # Distribute check results back to user_claims
    result_checks: list[dict] = data.get("checks", [])
    check_by_desc: dict[str, dict] = {c["check_description"]: c for c in result_checks}

    for i, claim in enumerate(user_claims):
        updated_checks = []
        for vc in claim.get("visual_checks", []):
            desc = vc.get("check_description", "")
            result = check_by_desc.get(desc, {})
            merged = dict(vc)
            merged.update({
                "finding": result.get("finding", ""),
                "observation": result.get("observation", ""),
                "claim_supported": bool(result.get("claim_supported", False)),
                "confidence": float(result.get("confidence", 0.0)),
            })
            updated_checks.append(merged)
        user_claims[i] = dict(claim)
        user_claims[i]["visual_checks"] = updated_checks

    damage_report = {
        "image_quality": data.get("image_quality", ""),
        "image_type": data.get("image_type", ""),
        "overall_confidence": data.get("overall_confidence", 0.0),
        "extracted_purchase_date": data.get("extracted_purchase_date"),
        "extracted_serial_number": data.get("extracted_serial_number"),
    }

    return {
        "user_claims": user_claims,
        "damage_report": damage_report,
    }
