"""
Router node — classifies intent of the latest user message.
Short-circuits to confirmation_handler if a context-switch is pending.
"""
from __future__ import annotations

import json
import logging

from gateway.llm_gateway import fast_llm
from models.state import AgentState

logger = logging.getLogger(__name__)

_INTENTS = (
    "greeting",       # hello / hi / how are you
    "out_of_scope",   # unrelated to warranty or claims
    "escalation",     # wants human agent, very upset
    "cancellation",   # wants to cancel the claim
    "status_query",   # asking about current claim status
    "frustration",    # frustrated but not escalating
    "claim",          # filing / submitting a new claim
    "policy",         # asking about coverage or policy details
    "issue",          # describing a product problem
    "evidence",       # providing a photo, receipt, or other proof
    "clarification",  # answering a follow-up question
)

_SYSTEM = f"""\
You are an intent-classification agent for a warranty claims chatbot.
Classify the LAST user message into exactly one of these intents:
{", ".join(_INTENTS)}

Rules:
- Use "escalation" only when the user explicitly asks for a human agent or is extremely upset.
- Use "frustration" when the user expresses negative emotions, disappointment, dissatisfaction, or
  frustration — even if no specific product problem is described. Sentiment-only messages belong here.
  Examples: "I am disappointed in your service", "this is unacceptable", "very unhappy with you",
  "terrible experience", "I'm fed up".
- Use "issue" ONLY when the user describes a concrete product malfunction or physical problem.
  Examples: "my charger won't turn on", "the display is cracked", "battery drains in an hour".
  Do NOT use "issue" for messages that express feelings without describing a specific product defect.
- Use "claim" when the user is initiating or submitting a warranty claim.
- Use "out_of_scope" for anything unrelated to the product or warranty.

Return JSON only:
{{"intent": "<intent>", "confidence": <0.0-1.0>, "requires_policy_lookup": <true|false>, "requires_vision": <true|false>}}

"requires_policy_lookup" = true when the response needs to reference policy sections.
"requires_vision" = true when the user mentions sending or has already sent a photo.
"""


def router_node(state: AgentState) -> dict:
    # If a context switch is already pending, bypass LLM classification
    if state.get("pending_switch_confirmation"):
        return {
            "intent": "pending_switch",
            "router_confidence": 1.0,
        }

    # Build conversation snippet for classification (last 6 messages max)
    history = state["messages"][-6:]
    last_user = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"), ""
    )

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": last_user},
    ]

    raw = fast_llm(messages, json_mode=True)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Router JSON parse failed, defaulting to 'issue'")
        data = {}

    intent = data.get("intent", "issue")
    if intent not in _INTENTS:
        intent = "issue"

    return {
        "intent": intent,
        "router_confidence": float(data.get("confidence", 0.5)),
    }
