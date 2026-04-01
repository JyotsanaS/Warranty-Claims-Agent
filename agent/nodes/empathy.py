"""
Empathy node — generates a short empathetic prefix for frustration intents.
Does NOT produce the full response; agent_respond will prepend this prefix.
"""
from __future__ import annotations

import logging

from gateway.llm_gateway import main_llm
from agent.prompt_store import get_prompt
from models.state import AgentState

logger = logging.getLogger(__name__)


def empathy_node(state: AgentState) -> dict:
    last_user = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), ""
    )
    messages = [
        {"role": "system", "content": get_prompt("empathy")},
        {"role": "user", "content": last_user},
    ]
    prefix = main_llm(messages, temperature=0.4).strip()
    # Strip trailing punctuation so it flows into the main response
    if prefix and prefix[-1] in ".!?":
        prefix = prefix[:-1]

    return {"empathy_prefix": prefix}
