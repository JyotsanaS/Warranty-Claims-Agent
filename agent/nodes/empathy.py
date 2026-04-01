"""
Empathy node — generates a short empathetic prefix for frustration intents.
Does NOT produce the full response; agent_respond will prepend this prefix.
"""
from __future__ import annotations

import logging

from gateway.llm_gateway import main_llm
from models.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an empathy-response generator. The customer is frustrated.
Write ONE short sentence (max 20 words) that acknowledges their frustration warmly.
Do NOT offer a resolution — that comes separately.
Do NOT start with "I" or repeat the word "frustrating".
Examples:
  "That sounds really stressful, and I completely understand."
  "Thank you for your patience — let me help sort this out."
"""


def empathy_node(state: AgentState) -> dict:
    last_user = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), ""
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": last_user},
    ]
    prefix = main_llm(messages, temperature=0.4).strip()
    # Strip trailing punctuation so it flows into the main response
    if prefix and prefix[-1] in ".!?":
        prefix = prefix[:-1]

    return {"empathy_prefix": prefix}
