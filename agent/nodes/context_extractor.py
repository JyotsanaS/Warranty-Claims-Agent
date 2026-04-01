"""
Context extractor node — parses structured claim details from the latest user message.
Merges into claim_items and detects context switches.
"""
from __future__ import annotations

import json
import logging

from gateway.llm_gateway import fast_llm
from models.state import AgentState, ClaimContext

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a structured data extraction agent for a warranty claims system.
Analyse the LAST user message and extract warranty claim details.

Return JSON only:
{
  "component": "<component name or empty string>",
  "incident_type": "<brief description of the problem or empty string>",
  "incident_date": "<YYYY-MM-DD or empty string>",
  "purchase_date": "<YYYY-MM-DD or empty string>",
  "has_receipt": <true|false>,
  "has_image": <true|false>,
  "is_new_claim": <true if this describes a NEW product issue, false if adding info to the current one>,
  "context_switch_detected": <true if the user is switching to a completely different product/issue>
}

Only extract values that are explicitly mentioned. Use empty strings for missing fields.
"has_receipt" = true only if the user says they have a receipt or proof of purchase.
"has_image" = true only if the user says they have or are sending a photo/image.
"""


def context_extractor_node(state: AgentState) -> dict:
    history = state["messages"]
    last_user = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"), ""
    )
    if not last_user:
        return {}

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": last_user},
    ]

    raw = fast_llm(messages, json_mode=True)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Context extractor JSON parse failed")
        data = {}

    claim_items: list[dict] = list(state.get("claim_items", []))
    active_index: int = state.get("active_claim_index", 0)
    context_switch = bool(data.get("context_switch_detected", False))
    pending_confirmation = False

    # If a new / switched claim is detected, decide how to handle it
    if data.get("is_new_claim") or context_switch:
        if claim_items and context_switch:
            # Prompt the user to confirm before switching
            pending_confirmation = True
        else:
            # Start a new ClaimContext
            new_ctx = ClaimContext(
                component=data.get("component", ""),
                incident_type=data.get("incident_type", ""),
                incident_date=data.get("incident_date") or None,
                purchase_date=data.get("purchase_date") or None,
                has_receipt=bool(data.get("has_receipt", False)),
                has_image=bool(data.get("has_image", False)),
            )
            claim_items.append(new_ctx.model_dump())
            active_index = len(claim_items) - 1
    else:
        # Update active claim with any newly extracted fields
        if not claim_items:
            new_ctx = ClaimContext(
                component=data.get("component", ""),
                incident_type=data.get("incident_type", ""),
                incident_date=data.get("incident_date") or None,
                purchase_date=data.get("purchase_date") or None,
                has_receipt=bool(data.get("has_receipt", False)),
                has_image=bool(data.get("has_image", False)),
            )
            claim_items.append(new_ctx.model_dump())
            active_index = 0
        else:
            ctx = dict(claim_items[active_index])
            for field in ("component", "incident_type", "incident_date", "purchase_date"):
                val = data.get(field)
                if val:
                    ctx[field] = val
            if data.get("has_receipt"):
                ctx["has_receipt"] = True
            if data.get("has_image") or state.get("image_local_path"):
                ctx["has_image"] = True
            claim_items[active_index] = ctx

    # Image attached this turn → mark active claim
    if state.get("image_local_path") and claim_items:
        ctx = dict(claim_items[active_index])
        ctx["has_image"] = True
        claim_items[active_index] = ctx

    return {
        "claim_items": claim_items,
        "active_claim_index": active_index,
        "context_switch_detected": context_switch,
        "pending_switch_confirmation": pending_confirmation,
    }
