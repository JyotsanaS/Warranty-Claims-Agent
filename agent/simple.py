"""
Simple streaming agent for VoltEdge warranty claims.

Responsibilities:
  1. Retrieve relevant policy chunks from Pinecone (RAG).
  2. Optionally analyse an uploaded image via the vision LLM.
  3. Stream the main LLM response token-by-token as SSE events.
  4. Detect a structured CLAIM DECISION marker and emit a claim_decision event.

This is intentionally a single-module, synchronous implementation.
A full LangGraph agent can replace this module without changing the router.
"""
from __future__ import annotations

import base64
import json
import logging
import re

logger = logging.getLogger(__name__)

# ── SSE helpers ──────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    """Format a single SSE line in the custom JSON envelope the UI expects."""
    return f"data: {json.dumps({'event': event, 'data': data})}\n\n"


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
You are VoltEdge's warranty claims agent. Your job is to help customers understand their \
warranty coverage and process claims accurately and empathetically.

Guidelines:
- Be concise, warm, and professional.
- When policy context is provided, cite the specific section (e.g. "§2 Component Coverage").
- Ask for missing evidence (proof of purchase, photo of damage) if it is needed to decide a claim.
- At the end of any response where you can make a definitive determination, output exactly one line:
  CLAIM DECISION: <status> - <short reason> - <policy section>
  where <status> is one of: approved, rejected, pending, escalated
- Only emit CLAIM DECISION when you have enough information to decide. Omit it otherwise.
"""

_SYSTEM_WITH_CONTEXT = _SYSTEM + "\n\nRelevant policy sections:\n{context}"


# ── Decision parser ───────────────────────────────────────────────────────────

_DECISION_RE = re.compile(
    r"CLAIM DECISION:\s*(approved|rejected|pending|escalated)\s*[-–]\s*([^-–\n]+?)(?:\s*[-–]\s*(.+?))?$",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_decision(text: str) -> dict | None:
    m = _DECISION_RE.search(text)
    if not m:
        return None
    status = m.group(1).strip().lower()
    reason = m.group(2).strip()
    clause = m.group(3).strip() if m.group(3) else ""
    return {
        "status": status,
        "reason": reason,
        "cited_clauses": [clause] if clause else [],
    }


# ── Main streaming function ───────────────────────────────────────────────────

def stream_agent_response(
    session_id: str,
    messages: list[dict],
    image_path: str | None = None,
) -> list[str]:
    """
    Generator that yields SSE-formatted strings.

    Emitted events (in order):
      tool_call  – when RAG / vision starts
      tool_result – when a tool finishes
      text_delta – one per LLM token
      claim_decision – if the LLM output contains a CLAIM DECISION line
      done  – stream end
      error – on unrecoverable failure
    """
    from app.config import settings
    from rag.retriever import retrieve
    from storage.image_store import read_image

    try:
        user_text = messages[-1]["content"] if messages else ""

        # ── 1. RAG retrieval ──────────────────────────────────────────────────
        policy_context: list[str] = []
        yield _sse("tool_call", {"tool": "rag", "status": "running"})
        chunks = retrieve(user_text)
        if chunks:
            policy_context = [c["text"] for c in chunks]
            yield _sse("tool_result", {"tool": "rag", "chunks_found": len(chunks)})
            logger.debug("RAG returned %d chunks", len(chunks))
        else:
            yield _sse("tool_result", {"tool": "rag", "chunks_found": 0})

        # ── 2. Vision analysis (if image attached) ────────────────────────────
        image_message_part: dict | None = None
        if image_path:
            yield _sse("tool_call", {"tool": "vision", "status": "running"})
            try:
                raw = read_image(image_path)
                b64 = base64.b64encode(raw).decode()
                image_message_part = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
                yield _sse("tool_result", {"tool": "vision", "result": {"analyzed": True}})
            except Exception as exc:
                logger.warning("Vision image read failed: %s", exc)
                yield _sse("tool_result", {"tool": "vision", "result": {"analyzed": False, "error": str(exc)}})

        # ── 3. Build LLM message list ─────────────────────────────────────────
        if policy_context:
            system_content = _SYSTEM_WITH_CONTEXT.format(
                context="\n---\n".join(policy_context)
            )
        else:
            system_content = _SYSTEM

        llm_messages: list[dict] = [{"role": "system", "content": system_content}]

        # Add prior turns (everything except the last user message)
        for msg in messages[:-1]:
            llm_messages.append({"role": msg["role"], "content": msg["content"]})

        # Last user message — with optional image
        if image_message_part:
            llm_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    image_message_part,
                ],
            })
        else:
            llm_messages.append({"role": "user", "content": user_text})

        # ── 4. LLM streaming call ─────────────────────────────────────────────
        import litellm

        model = settings.llm_vision_model if image_path else settings.llm_main_model
        logger.debug("Calling model %s", model)

        response = litellm.completion(
            model=model,
            messages=llm_messages,
            stream=True,
            api_key=settings.groq_api_key,
        )

        full_text = ""
        for chunk in response:
            token: str = chunk.choices[0].delta.content or ""
            if token:
                full_text += token
                yield _sse("text_delta", {"token": token})

        # ── 5. Claim decision ─────────────────────────────────────────────────
        decision = _parse_decision(full_text)
        if decision:
            yield _sse("claim_decision", decision)

        yield _sse("done", {})

    except Exception as exc:
        logger.exception("Agent error in session %s: %s", session_id, exc)
        yield _sse("error", {"message": str(exc)})
