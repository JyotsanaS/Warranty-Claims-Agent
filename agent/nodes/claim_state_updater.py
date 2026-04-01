"""
Claim state updater node — merges structured claim context and normalized user claims
for claim-progressing turns.
"""
from __future__ import annotations

import json
import logging

from gateway.llm_gateway import fast_llm
from agent.prompt_store import get_prompt
from models.state import AgentState, ClaimContext, UserClaim

logger = logging.getLogger(__name__)


def _normalize_assertion_type(incident_type: str) -> str:
    normalized = str(incident_type or "").strip().lower()
    if normalized in {"cosmetic", "scratch", "scratches", "dent", "scuff"}:
        return "cosmetic"
    if normalized in {"missing_part", "missing part"}:
        return "missing_part"
    if normalized in {"malfunction", "electrical", "power", "performance"}:
        return "malfunction"
    return "damage"


def _synthesize_claim_from_context(
    extracted_ctx: dict,
    active_claim: dict,
    last_user: str,
) -> list[dict]:
    component = (
        str(extracted_ctx.get("component", "")).strip()
        or str(active_claim.get("component", "")).strip()
        or "device"
    )
    incident_type = (
        str(extracted_ctx.get("incident_type", "")).strip()
        or str(active_claim.get("incident_type", "")).strip()
    )
    if not incident_type and not last_user.strip():
        return []

    statement = last_user.strip() or incident_type or "Customer reported an issue."
    return [
        {
            "component": component,
            "assertion_type": _normalize_assertion_type(incident_type),
            "verbatim_statement": statement,
            "requires_visual_validation": bool(
                incident_type
                or any(word in statement.lower() for word in ("scratch", "crack", "dent", "damage"))
            ),
        }
    ]


def _merge_claim_context(existing: dict, extracted: dict, image_local_path: str | None) -> dict:
    ctx = dict(existing)
    for field in ("component", "incident_type", "incident_date", "purchase_date"):
        val = extracted.get(field)
        if val:
            ctx[field] = val
    if extracted.get("has_receipt"):
        ctx["has_receipt"] = True
    if extracted.get("has_image") or image_local_path:
        ctx["has_image"] = True
    return ctx


def _merge_user_claims(existing: list[dict], extracted: list[dict]) -> list[dict]:
    merged = [dict(c) for c in existing]
    index: dict[tuple[str, str], int] = {}
    for i, claim in enumerate(merged):
        key = (
            str(claim.get("component", "")).strip().lower(),
            str(claim.get("assertion_type", "damage")).strip().lower(),
        )
        index[key] = i

    for raw_claim in extracted:
        comp = str(raw_claim.get("component", "")).strip()
        assertion_type = str(raw_claim.get("assertion_type", "damage")).strip() or "damage"
        if not comp:
            continue
        key = (comp.lower(), assertion_type.lower())
        try:
            claim = UserClaim(
                component=comp,
                assertion_type=assertion_type,
                verbatim_statement=str(raw_claim.get("verbatim_statement", "")).strip(),
                requires_visual_validation=bool(
                    raw_claim.get("requires_visual_validation", False)
                ),
            ).model_dump()
        except Exception as exc:
            logger.warning("Skipping malformed claim: %s — %s", raw_claim, exc)
            continue

        if key in index:
            existing_claim = dict(merged[index[key]])
            if claim["verbatim_statement"]:
                existing_claim["verbatim_statement"] = claim["verbatim_statement"]
            existing_claim["requires_visual_validation"] = (
                bool(existing_claim.get("requires_visual_validation"))
                or claim["requires_visual_validation"]
            )
            merged[index[key]] = existing_claim
        else:
            index[key] = len(merged)
            merged.append(claim)

    return merged


def claim_state_updater_node(state: AgentState) -> dict:
    history = state["messages"]
    last_user = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"), ""
    )
    claim_items: list[dict] = list(state.get("claim_items", []))
    active_index = state.get("active_claim_index", 0)

    if not last_user:
        if state.get("image_local_path") and claim_items and 0 <= active_index < len(claim_items):
            ctx = dict(claim_items[active_index])
            if not ctx.get("has_image"):
                ctx["has_image"] = True
                claim_items[active_index] = ctx
            return {
                "claim_items": claim_items,
                "active_claim_index": active_index,
                "claim_state_updated_this_turn": True,
            }
        return {
            "claim_state_updated_this_turn": True,
        }

    active_claim = (
        dict(claim_items[active_index])
        if claim_items and 0 <= active_index < len(claim_items)
        else ClaimContext().model_dump()
    )

    relevant_history = [
        m for m in history if m.get("role") in ("user", "assistant")
    ][-6:]
    conversation = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in relevant_history
    )

    prompt = (
        f"Active claim context:\n{json.dumps(active_claim)}\n\n"
        f"Recent conversation:\n{conversation}\n\n"
        f"Latest user message:\n{last_user}"
    )
    messages = [
        {"role": "system", "content": get_prompt("claim_state_updater")},
        {"role": "user", "content": prompt},
    ]

    raw = fast_llm(messages, json_mode=True)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Claim state updater JSON parse failed")
        data = {}

    extracted_ctx = data.get("claim_context") or {}
    extracted_claims = data.get("user_claims") or []
    context_switch = bool(data.get("context_switch_detected", False))
    pending_confirmation = False
    pending_claim_draft = None

    if not isinstance(extracted_claims, list):
        extracted_claims = []
    if not extracted_claims:
        extracted_claims = _synthesize_claim_from_context(extracted_ctx, active_claim, last_user)

    if not claim_items:
        base_ctx = ClaimContext().model_dump()
        merged_ctx = _merge_claim_context(base_ctx, extracted_ctx, state.get("image_local_path"))
        if any(merged_ctx.get(k) for k in ("component", "incident_type", "incident_date", "purchase_date")):
            claim_items.append(merged_ctx)
            active_index = 0
    elif data.get("is_new_claim") or context_switch:
        draft_ctx = ClaimContext(
            component=extracted_ctx.get("component", ""),
            incident_type=extracted_ctx.get("incident_type", ""),
            incident_date=extracted_ctx.get("incident_date") or None,
            purchase_date=extracted_ctx.get("purchase_date") or None,
            has_receipt=bool(extracted_ctx.get("has_receipt", False)),
            has_image=bool(extracted_ctx.get("has_image", False) or state.get("image_local_path")),
        ).model_dump()
        if claim_items and context_switch:
            pending_confirmation = True
            pending_claim_draft = draft_ctx
        else:
            claim_items.append(draft_ctx)
            active_index = len(claim_items) - 1
    else:
        merged_ctx = _merge_claim_context(
            claim_items[active_index], extracted_ctx, state.get("image_local_path")
        )
        claim_items[active_index] = merged_ctx

    if state.get("image_local_path") and claim_items and 0 <= active_index < len(claim_items):
        ctx = dict(claim_items[active_index])
        ctx["has_image"] = True
        claim_items[active_index] = ctx

    user_claims = list(state.get("user_claims", []))
    if not pending_confirmation:
        user_claims = _merge_user_claims(user_claims, extracted_claims)

    return {
        "claim_items": claim_items,
        "active_claim_index": active_index,
        "user_claims": user_claims,
        "context_switch_detected": context_switch,
        "pending_switch_confirmation": pending_confirmation,
        "pending_claim_draft": pending_claim_draft,
        "claim_state_updated_this_turn": True,
    }
