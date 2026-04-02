"""
Session management endpoints.

In-memory session store (MemorySaver equivalent for MVP).
Sessions are lost on server restart — acceptable for a prototype.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import settings
from storage.image_store import save_image
from storage.session_store import save_session_json

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory store: session_id → session dict
_sessions: dict[str, dict] = {}
IDLE_TIMEOUT_SECONDS = 180
SESSION_TURN_LIMIT = 15
SESSION_TURN_WINDOW_SECONDS = 5 * 60
INITIAL_ASSISTANT_MESSAGE = (
    "Welcome to VoltEdge warranty support. I can help with warranty claims, "
    "coverage questions, and claim status updates. Tell me what issue you're "
    "facing, and I'll guide you from there."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_session(session_id: str) -> dict:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return _sessions[session_id]


def _enforce_message_rate_limit(session: dict, now: float | None = None) -> None:
    now = time.time() if now is None else now

    if session.get("active_request"):
        raise HTTPException(
            status_code=429,
            detail="A turn is already in progress for this session",
        )

    timestamps = [
        ts for ts in session.get("request_timestamps", [])
        if now - ts < SESSION_TURN_WINDOW_SECONDS
    ]
    if len(timestamps) >= SESSION_TURN_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Session rate limit exceeded: max {SESSION_TURN_LIMIT} turns "
                f"per {SESSION_TURN_WINDOW_SECONDS // 60} minutes"
            ),
        )

    timestamps.append(now)
    session["request_timestamps"] = timestamps
    session["active_request"] = True


def _release_active_request(session: dict) -> None:
    session["active_request"] = False


def _persist_session_artifacts(
    session: dict,
    image_path: str | None,
    trace: list[dict],
    total_ms: int | None,
    openinference_trace_id: str | None,
    openinference_trace: list[dict] | None,
) -> None:
    session_id = session["session_id"]

    conversation_payload = {
        "session_id": session_id,
        "created_at": session.get("created_at"),
        "updated_at": datetime.utcnow().isoformat(),
        "claim_status": session.get("claim_status"),
        "claim_items": session.get("claim_items", []),
        "user_claims": session.get("user_claims", []),
        "pending_claim_draft": session.get("pending_claim_draft"),
        "awaiting_post_resolution_followup": session.get(
            "awaiting_post_resolution_followup", False
        ),
        "active_claim_index": session.get("active_claim_index", 0),
        "messages": session.get("messages", []),
        "latest_image_path": image_path,
        "latest_trace_total_ms": total_ms,
        "latest_trace": trace,
        "openinference_trace_id": openinference_trace_id,
        "openinference_trace": openinference_trace or [],
    }
    save_session_json(session_id, "conversation.json", conversation_payload)



# ── Idle eviction background loop ─────────────────────────────────────────────

async def idle_eviction_loop() -> None:
    """Evict sessions idle for longer than IDLE_TIMEOUT_SECONDS. Runs every 15 s."""
    while True:
        await asyncio.sleep(15)
        now = time.time()
        to_evict = [
            sid for sid, sess in list(_sessions.items())
            if now - sess.get("last_activity_at", now) > IDLE_TIMEOUT_SECONDS
        ]
        for sid in to_evict:
            _sessions.pop(sid, None)
            logger.info("Evicted idle session %s (idle > %ds)", sid, IDLE_TIMEOUT_SECONDS)


# ── POST /sessions ─────────────────────────────────────────────────────────────

@router.post("/sessions", status_code=201)
async def create_session():
    """Create a new conversation session."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "session_id": session_id,
        "messages": [],              # {role, content} dicts
        "claim_status": None,
        "claim_items": [],           # persisted ClaimContext dicts
        "user_claims": [],           # persisted UserClaim dicts
        "pending_claim_draft": None,
        "awaiting_post_resolution_followup": False,
        "active_claim_index": 0,
        "awaiting_first_user_turn": True,
        "created_at": datetime.utcnow().isoformat(),
        "last_activity_at": time.time(),
        "request_timestamps": [],
        "active_request": False,
    }
    logger.info("Created session %s", session_id)
    return {
        "session_id": session_id,
        "initial_assistant_message": INITIAL_ASSISTANT_MESSAGE,
    }


# ── POST /sessions/{session_id}/messages ──────────────────────────────────────

@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    text: str = Form(""),
    image: Optional[UploadFile] = File(None),
):
    """
    Send a user message (with optional image) and stream the agent response
    as Server-Sent Events.

    SSE event format:
      data: {"event": "<type>", "data": {...}}\n\n
    """
    session = _require_session(session_id)
    session["last_activity_at"] = time.time()

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

    text = "" if text is None else str(text).strip()
    if not text and not image_path:
        raise HTTPException(status_code=400, detail="Message must include text or an image")

    _enforce_message_rate_limit(session)

    # ── Append user message to history ────────────────────────────────────────
    session["messages"].append({"role": "user", "content": text})

    # ── Build SSE generator ───────────────────────────────────────────────────
    def _event_stream():
        from agent.runner import pop_final_state, stream_agent_response

        accumulated_tokens: list[str] = []
        last_trace: list[dict] = []
        last_trace_total_ms: int | None = None
        openinference_trace_id: str | None = None
        openinference_trace: list[dict] = []
        close_session_after_response = False

        gen = stream_agent_response(
            session_id=session_id,
            messages=session["messages"],
            image_path=image_path,
            claim_items=session.get("claim_items", []),
            user_claims=session.get("user_claims", []),
            pending_claim_draft=session.get("pending_claim_draft"),
            awaiting_post_resolution_followup=session.get(
                "awaiting_post_resolution_followup", False
            ),
            active_claim_index=session.get("active_claim_index", 0),
            awaiting_first_user_turn=session.get("awaiting_first_user_turn", False),
        )

        try:
            for chunk in gen:
                if chunk.startswith("data: "):
                    try:
                        evt = json.loads(chunk[6:].strip())
                        etype = evt.get("event") or evt.get("type")
                        if etype == "text_delta":
                            accumulated_tokens.append(evt["data"].get("token", ""))
                        elif etype == "claim_decision":
                            session["claim_status"] = evt["data"].get("status")
                        elif etype == "done":
                            last_trace = evt["data"].get("trace", [])
                            last_trace_total_ms = evt["data"].get("total_ms")
                    except (json.JSONDecodeError, KeyError):
                        pass
                yield chunk

            # ── Persist updated agent state back into session ─────────────────
            final = pop_final_state(session_id)
            if final.get("claim_items") is not None:
                session["claim_items"] = final["claim_items"]
            if final.get("user_claims") is not None:
                session["user_claims"] = final["user_claims"]
            if "pending_claim_draft" in final:
                session["pending_claim_draft"] = final.get("pending_claim_draft")
            if "awaiting_post_resolution_followup" in final:
                session["awaiting_post_resolution_followup"] = final.get(
                    "awaiting_post_resolution_followup", False
                )
            if final.get("active_claim_index") is not None:
                session["active_claim_index"] = final["active_claim_index"]
            if final.get("_openinference_trace_id") is not None:
                openinference_trace_id = final["_openinference_trace_id"]
            if final.get("_openinference_trace") is not None:
                openinference_trace = final["_openinference_trace"]
            close_session_after_response = bool(final.get("terminate_session"))
            session["awaiting_first_user_turn"] = False

            # Save assistant reply to conversation history
            full_response = "".join(accumulated_tokens)
            if full_response:
                session["messages"].append({"role": "assistant", "content": full_response})

            _persist_session_artifacts(
                session,
                image_path,
                last_trace,
                last_trace_total_ms,
                openinference_trace_id,
                openinference_trace,
            )

            if close_session_after_response:
                _sessions.pop(session_id, None)
                logger.info("Session %s closed by security guardrail", session_id)
        finally:
            _release_active_request(session)

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
        "claim_items": session.get("claim_items", []),
        "user_claims": session.get("user_claims", []),
        "pending_claim_draft": session.get("pending_claim_draft"),
        "awaiting_post_resolution_followup": session.get(
            "awaiting_post_resolution_followup", False
        ),
        "awaiting_first_user_turn": session.get("awaiting_first_user_turn", False),
        "created_at": session["created_at"],
    }


# ── DELETE /sessions/{session_id} ─────────────────────────────────────────────

@router.delete("/sessions/{session_id}", status_code=200)
async def delete_session(session_id: str):
    """Close a session and purge its data."""
    _sessions.pop(session_id, None)
    logger.info("Deleted session %s", session_id)
    return {"status": "deleted"}
