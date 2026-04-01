"""
Session management endpoints.

In-memory session store (MemorySaver equivalent for MVP).
Sessions are lost on server restart — acceptable for a prototype.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import settings
from storage.image_store import save_image

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory store: session_id → session dict
_sessions: dict[str, dict] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_session(session_id: str) -> dict:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return _sessions[session_id]


# ── POST /sessions ─────────────────────────────────────────────────────────────

@router.post("/sessions", status_code=201)
async def create_session():
    """Create a new conversation session."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "session_id": session_id,
        "messages": [],          # conversation history: [{role, content}]
        "claim_status": None,
        "created_at": datetime.utcnow().isoformat(),
    }
    logger.info("Created session %s", session_id)
    return {"session_id": session_id}


# ── POST /sessions/{session_id}/messages ──────────────────────────────────────

@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    text: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    """
    Send a user message (with optional image) and stream the agent response
    as Server-Sent Events.

    SSE event format:
      data: {"event": "<type>", "data": {...}}\n\n
    """
    session = _require_session(session_id)

    # ── Image validation & storage ────────────────────────────────────────────
    image_path: str | None = None
    if image and image.filename:
        content = await image.read()
        max_bytes = settings.image_max_size_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"Image exceeds {settings.image_max_size_mb} MB limit",
            )
        ext = (image.filename.rsplit(".", 1)[-1].lower() if "." in image.filename else "jpg")
        if ext not in {"jpg", "jpeg", "png", "webp"}:
            raise HTTPException(status_code=400, detail=f"Unsupported image type: {ext}")
        image_path = save_image(session_id, "upload", content, ext)
        logger.debug("Saved image for session %s → %s", session_id, image_path)

    # ── Append user message to history ────────────────────────────────────────
    session["messages"].append({"role": "user", "content": text})

    # ── Build SSE generator ───────────────────────────────────────────────────
    def _event_stream():
        from agent.simple import stream_agent_response

        accumulated_tokens: list[str] = []

        for chunk in stream_agent_response(
            session_id=session_id,
            messages=session["messages"],
            image_path=image_path,
        ):
            # Accumulate text tokens to save back to history
            if chunk.startswith("data: "):
                try:
                    evt = json.loads(chunk[6:].strip())
                    etype = evt.get("event") or evt.get("type")
                    if etype == "text_delta":
                        accumulated_tokens.append(evt["data"].get("token", ""))
                    elif etype == "claim_decision":
                        session["claim_status"] = evt["data"].get("status")
                except (json.JSONDecodeError, KeyError):
                    pass
            yield chunk

        # Save assistant response to conversation history
        full_response = "".join(accumulated_tokens)
        if full_response:
            session["messages"].append({"role": "assistant", "content": full_response})

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── GET /sessions/{session_id} ────────────────────────────────────────────────

@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Return a summary of the session state."""
    session = _require_session(session_id)
    return {
        "session_id": session["session_id"],
        "message_count": len(session["messages"]),
        "claim_status": session.get("claim_status"),
        "created_at": session["created_at"],
    }


# ── DELETE /sessions/{session_id} ─────────────────────────────────────────────

@router.delete("/sessions/{session_id}", status_code=200)
async def delete_session(session_id: str):
    """Close a session and purge its data."""
    _sessions.pop(session_id, None)
    logger.info("Deleted session %s", session_id)
    return {"status": "deleted"}
