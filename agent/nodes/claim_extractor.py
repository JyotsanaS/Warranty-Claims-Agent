"""
Claim extractor node — extracts / updates UserClaim list from the full conversation.
Additive: never removes existing claims.
"""
from __future__ import annotations

import json
import logging

from gateway.llm_gateway import fast_llm
from models.state import AgentState, UserClaim

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a claim-extraction agent for a warranty claims system.
Analyse the full conversation and extract all distinct product damage, issues or malfunction claims.

Return JSON only — an array of claim objects:
[
  {
    "component": "<product part or product, e.g. LED Display, Charging Cable, Battery Pack>",
    "assertion_type": "<damage|malfunction|cosmetic|missing_part>",
    "verbatim_statement": "<short direct quote from the user describing the issue>",
    "requires_visual_validation": <true if physical damage needs photographic proof>
  }
]

Rules:
- Each distinct component or issue = one object.
- If the same component is mentioned multiple times, include it once.
- Only include issues the USER asserts — not agent responses.
- Return [] if no claims are present yet.
- Return [] when the user only expresses emotions, feelings, or general dissatisfaction without
  describing a specific physical defect or malfunction. Examples that must return []:
  "I am disappointed in your service", "extremely unhappy with your product",
  "terrible experience", "this is unacceptable", "hi".
"""


def claim_extractor_node(state: AgentState) -> dict:
    existing: list[dict] = list(state.get("user_claims", []))

    # Build a compact conversation string (last 12 messages)
    history = state["messages"][-12:]
    conversation = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history
    )

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": conversation},
    ]

    raw = fast_llm(messages, json_mode=True)
    try:
        # The response might be wrapped in an object — try both
        data = json.loads(raw)
        if isinstance(data, dict):
            # e.g. {"claims": [...]}
            extracted = data.get("claims", data.get("items", []))
        else:
            extracted = data  # already a list
    except (json.JSONDecodeError, TypeError):
        logger.warning("Claim extractor JSON parse failed")
        extracted = []

    if not isinstance(extracted, list):
        extracted = []

    # Merge: add newly extracted claims not already present
    existing_components = {c["component"].lower() for c in existing}
    for raw_claim in extracted:
        comp = raw_claim.get("component", "").strip()
        if not comp:
            continue
        if comp.lower() in existing_components:
            continue
        try:
            claim = UserClaim(
                component=comp,
                assertion_type=raw_claim.get("assertion_type", "damage"),
                verbatim_statement=raw_claim.get("verbatim_statement", ""),
                requires_visual_validation=bool(
                    raw_claim.get("requires_visual_validation", False)
                ),
            )
            existing.append(claim.model_dump())
            existing_components.add(comp.lower())
        except Exception as exc:
            logger.warning("Skipping malformed claim: %s — %s", raw_claim, exc)

    return {"user_claims": existing}
