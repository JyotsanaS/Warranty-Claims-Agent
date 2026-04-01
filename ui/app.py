"""
VoltEdge Warranty & Claims Agent — Streamlit Frontend
"""
import json
import os
import time

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
MOCK_BACKEND = os.getenv("MOCK_BACKEND", "0") == "1"
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

MOCK_SSE_EVENTS = [
    {"event": "tool_call", "data": {"tool": "rag", "status": "running"}},
    {"event": "text_delta", "data": {"token": "Based on "}},
    {"event": "text_delta", "data": {"token": "our warranty policy, "}},
    {"event": "text_delta", "data": {"token": "the LED display is covered for 12 months. "}},
    {"event": "text_delta", "data": {"token": "Since your purchase is within that window, "}},
    {"event": "text_delta", "data": {"token": "your claim is approved."}},
    {
        "event": "claim_decision",
        "data": {
            "status": "approved",
            "reason": "LED display within 12-month warranty period",
            "cited_clauses": ["§2 Specific Component Coverage"],
        },
    },
    {"event": "done", "data": {}},
]

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "session_id": None,
        "messages": [],
        "claim_status": None,
        "claim_decision_data": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# API client helpers
# ---------------------------------------------------------------------------

def create_session() -> str:
    """POST /api/v1/sessions — returns session_id."""
    if MOCK_BACKEND:
        import uuid
        return str(uuid.uuid4())

    try:
        resp = httpx.post(f"{BACKEND_URL}/api/v1/sessions", timeout=10)
        resp.raise_for_status()
        return resp.json()["session_id"]
    except httpx.ConnectError:
        st.error(f"Cannot reach backend at {BACKEND_URL}. Is it running?")
        st.stop()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code} when creating session.")
        st.stop()


def delete_session(session_id: str) -> None:
    """DELETE /api/v1/sessions/{session_id}."""
    if MOCK_BACKEND:
        return
    try:
        httpx.delete(f"{BACKEND_URL}/api/v1/sessions/{session_id}", timeout=10)
    except Exception:
        pass  # best-effort cleanup


def stream_message(session_id: str, text: str, image_bytes: bytes | None = None):
    """
    POST /api/v1/sessions/{session_id}/messages and yield parsed SSE event dicts.

    Uses httpx synchronous streaming with timeout=None because SSE connections
    are long-lived — any finite read timeout would kill the stream prematurely.
    """
    if MOCK_BACKEND:
        for event in MOCK_SSE_EVENTS:
            time.sleep(0.06)
            yield event
        return

    url = f"{BACKEND_URL}/api/v1/sessions/{session_id}/messages"
    files: dict = {"text": (None, text)}
    if image_bytes:
        files["image"] = ("upload.jpg", image_bytes, "image/jpeg")

    try:
        with httpx.Client(timeout=None) as client:
            with client.stream("POST", url, files=files) as response:
                if response.status_code == 404:
                    st.session_state.session_id = None
                    raise RuntimeError("Session expired. Please start a new session.")
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        raw = line[6:]
                        try:
                            yield json.loads(raw)
                        except json.JSONDecodeError:
                            pass
    except httpx.ConnectError:
        raise RuntimeError(f"Cannot reach backend at {BACKEND_URL}.")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"API error {e.response.status_code}")


# ---------------------------------------------------------------------------
# SSE → string generator adapter for st.write_stream()
# ---------------------------------------------------------------------------

def _make_response_gen(event_stream, status_container):
    """
    Adapts the full SSE event stream into a string-only generator for
    st.write_stream(), handling side-effect events inline.

    The stream cannot be replayed, so non-text events (tool calls, claim
    decisions) are processed as side effects during the same pass.
    """
    for event in event_stream:
        etype = event.get("event") or event.get("type")

        if etype == "text_delta":
            yield event["data"]["token"]

        elif etype == "tool_call":
            tool = event["data"].get("tool", "unknown")
            status_container.write(f"Running: **{tool}**...")

        elif etype == "tool_result":
            tool = event["data"].get("tool", "unknown")
            status_container.write(f"Completed: **{tool}**")

        elif etype == "claim_decision":
            st.session_state.claim_status = event["data"]["status"]
            st.session_state.claim_decision_data = event["data"]

        elif etype == "error":
            status_container.update(label="Error", state="error")
            raise RuntimeError(event["data"].get("message", "Unknown error from agent."))

        # "done" → generator exhausts naturally, no action needed


# ---------------------------------------------------------------------------
# Claim badge renderer
# ---------------------------------------------------------------------------

def _render_claim_badge(data: dict, container=None) -> None:
    """Render a coloured claim decision badge."""
    status = data.get("status", "pending")
    reason = data.get("reason", "")
    clauses = ", ".join(data.get("cited_clauses", []))

    label = f"Claim **{status.upper()}**"
    if reason:
        label += f" — {reason}"
    if clauses:
        label += f" *(policy: {clauses})*"

    badge_fn_map = {
        "approved": (container or st).success,
        "rejected": (container or st).error,
        "escalated": (container or st).warning,
        "pending": (container or st).info,
    }
    badge_fn_map.get(status, (container or st).info)(label)


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VoltEdge Warranty Agent",
    page_icon="⚡",
    layout="centered",
)

init_session_state()

# Auto-create a session on first load so the user can chat immediately.
if st.session_state.session_id is None:
    st.session_state.session_id = create_session()

# ---------------------------------------------------------------------------
# ChatGPT-like styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Hide default Streamlit header decoration ──────────────────────────── */
#MainMenu, footer { visibility: hidden; }

/* ── User messages: right-aligned bubble ───────────────────────────────── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
    gap: 0.75rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
    align-items: flex-end;
    background: #2f2f2f;
    border-radius: 18px 18px 4px 18px;
    padding: 0.65rem 1rem;
    max-width: 72%;
    color: #ececec;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] p {
    color: #ececec;
    margin: 0;
}

/* ── Assistant messages: left-aligned, no bubble ───────────────────────── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stChatMessageContent"] {
    background: transparent;
    max-width: 82%;
}

/* ── Chat input bar — matches ChatGPT style ─────────────────────────────── */
[data-testid="stChatInput"] {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ VoltEdge")
    st.caption("Automated Warranty & Claims Agent")
    st.divider()

    if st.button("New Session", type="primary", use_container_width=True):
        if st.session_state.session_id:
            delete_session(st.session_state.session_id)
        new_id = create_session()
        st.session_state.session_id = new_id
        st.session_state.messages = []
        st.session_state.claim_status = None
        st.session_state.claim_decision_data = None
        st.rerun()

    short_id = st.session_state.session_id[:8] + "..."
    st.caption(f"Session: `{short_id}`")

    # Persistent claim status badge in sidebar
    if st.session_state.claim_status:
        st.divider()
        st.markdown("**Claim Status**")
        _render_claim_badge(
            st.session_state.claim_decision_data or {"status": st.session_state.claim_status},
            container=st.sidebar,
        )

    if MOCK_BACKEND:
        st.divider()
        st.info("Mock mode active — no backend needed.")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.header("Warranty Claims Chat")

# Render existing conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image_bytes"):
            st.image(msg["image_bytes"], width=150)
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input — file attachment button lives inside the input bar (ChatGPT style)
# ---------------------------------------------------------------------------
prompt = st.chat_input(
    "Message VoltEdge...",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"],
)

if prompt:
    text = prompt.text or ""
    files = prompt.files or []
    image_bytes: bytes | None = None
    if files:
        raw = files[0].getvalue()
        if len(raw) > MAX_IMAGE_BYTES:
            st.error("Image exceeds the 10 MB limit. Please choose a smaller file.")
            st.stop()
        image_bytes = raw

    # Render user turn immediately
    with st.chat_message("user"):
        if image_bytes:
            st.image(image_bytes, width=150)
        if text:
            st.markdown(text)
    st.session_state.messages.append(
        {"role": "user", "content": text, "image_bytes": image_bytes}
    )

    # Stream assistant response
    with st.chat_message("assistant"):
        status_container = st.status("Thinking...", expanded=True)
        full_text = ""
        try:
            event_gen = stream_message(st.session_state.session_id, text, image_bytes)
            full_text = st.write_stream(_make_response_gen(event_gen, status_container))
            status_container.update(label="Done", state="complete", expanded=False)
        except RuntimeError as e:
            st.error(str(e))
            full_text = ""

    st.session_state.messages.append({"role": "assistant", "content": full_text, "image_bytes": None})

    # Inline claim decision badge (shown once, directly after the turn that produced it)
    if st.session_state.claim_decision_data:
        _render_claim_badge(st.session_state.claim_decision_data)
        # Keep data in state for the sidebar badge but mark as shown
        st.session_state.claim_decision_data = None
