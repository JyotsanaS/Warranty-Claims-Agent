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
INITIAL_ASSISTANT_MESSAGE = (
    "Welcome to VoltEdge warranty support. I can help with warranty claims, "
    "coverage questions, and claim status updates. Tell me what issue you're "
    "facing, and I'll guide you from there."
)

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
        "initial_assistant_message": "",
        "messages": [],
        "claim_status": None,
        "claim_decision_data": None,
        "last_trace": None,
        "last_trace_total_ms": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# API client helpers
# ---------------------------------------------------------------------------

def create_session() -> dict:
    """POST /api/v1/sessions — returns session metadata."""
    if MOCK_BACKEND:
        import uuid
        return {
            "session_id": str(uuid.uuid4()),
            "initial_assistant_message": INITIAL_ASSISTANT_MESSAGE,
        }

    try:
        resp = httpx.post(f"{BACKEND_URL}/api/v1/sessions", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "session_id": data["session_id"],
            "initial_assistant_message": data.get(
                "initial_assistant_message", INITIAL_ASSISTANT_MESSAGE
            ),
        }
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

_NODE_LABELS: dict[str, str] = {
    "router":               "Classify intent",
    "claim_state_updater":  "Update claim state",
    "planner":              "Plan execution",
    "empathy_node":         "Generate empathy",
    "policy_checker":       "RAG retrieval",
    "evidence_planner":     "Plan evidence checks",
    "vision_analysis":      "Analyse image",
    "claim_validator":      "Validate claim",
    "agent_respond":        "Generate response",
    "claim_decision":       "Final decision",
    "greeting_node":        "Greeting",
    "fallback_node":        "Out-of-scope",
    "escalation_node":      "Escalation",
    "cancellation_node":    "Cancellation",
    "status_node":          "Status query",
    "confirmation_handler": "Confirm context switch",
    "feedback_node":        "Collect feedback",
}


def _node_detail(step: dict) -> str:
    """Return a short human-readable detail string for a trace step."""
    node = step.get("node", "")
    if node == "router":
        return f"intent={step.get('intent', '?')}  conf={step.get('confidence', 0):.0%}"
    if node == "claim_state_updater":
        n = step.get("claims_found", 0)
        return f"{n} claim{'s' if n != 1 else ''} found"
    if node == "planner":
        next_node = step.get("next_node", "—") or "—"
        plan = step.get("plan", [])
        plan_summary = " -> ".join(plan[:3]) if plan else "—"
        return f"next={next_node} · {plan_summary}"
    if node == "policy_checker":
        n = step.get("chunks_retrieved", 0)
        clauses = ", ".join(step.get("clauses", [])) or "—"
        return f"{n} chunks · clauses: {clauses}"
    if node == "empathy_node":
        return f'prefix: "{step.get("prefix", "")}"'
    if node == "vision_analysis":
        return f"quality={step.get('image_quality', '?')}  conf={step.get('confidence', 0):.0%}"
    if node == "claim_decision":
        verdicts = step.get("verdicts", {})
        return "  ".join(f"{k}={v}" for k, v in verdicts.items()) or "—"
    return ""


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

        elif etype == "node_trace":
            step = event["data"]
            node = step.get("node", "unknown")
            label = _NODE_LABELS.get(node, node)
            duration = step.get("duration_ms", 0)
            detail = _node_detail(step)
            line = f"**{label}** `{duration} ms`"
            if detail:
                line += f" — {detail}"
            status_container.write(line)

        elif etype == "tool_call":
            pass  # node_trace already covers this visually

        elif etype == "tool_result":
            pass  # node_trace already covers this visually

        elif etype == "claim_decision":
            st.session_state.claim_status = event["data"]["status"]
            st.session_state.claim_decision_data = event["data"]

        elif etype == "done":
            st.session_state.last_trace = event["data"].get("trace", [])
            st.session_state.last_trace_total_ms = event["data"].get("total_ms")

        elif etype == "error":
            status_container.update(label="Error", state="error")
            raise RuntimeError(event["data"].get("message", "Unknown error from agent."))


def _render_trace(trace: list[dict]) -> None:
    """Render a compact agent trace table inside an expander."""
    if not trace:
        return
    total_ms = sum(s.get("duration_ms", 0) for s in trace)
    with st.expander(f"Agent trace — {len(trace)} nodes · {total_ms} ms total"):
        for i, step in enumerate(trace, 1):
            node = step.get("node", "unknown")
            label = _NODE_LABELS.get(node, node)
            duration = step.get("duration_ms", 0)
            elapsed = step.get("elapsed_ms", 0)
            detail = _node_detail(step)
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.markdown(f"**{i}. {label}**" + (f"  \n`{detail}`" if detail else ""))
            col2.caption(f"{duration} ms")
            col3.caption(f"+{elapsed} ms")


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
    session_data = create_session()
    st.session_state.session_id = session_data["session_id"]
    st.session_state.initial_assistant_message = session_data["initial_assistant_message"]
    st.session_state.messages = []

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
    background: transparent;
    border-radius: 0;
    padding: 0;
    max-width: 72%;
    color: inherit;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] p {
    color: inherit;
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
        session_data = create_session()
        st.session_state.session_id = session_data["session_id"]
        st.session_state.initial_assistant_message = session_data["initial_assistant_message"]
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

if st.session_state.initial_assistant_message:
    with st.chat_message("assistant"):
        st.markdown(st.session_state.initial_assistant_message)

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
    text = "" if prompt.text is None else str(prompt.text)
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
            trace = st.session_state.last_trace or []
            total_ms = st.session_state.last_trace_total_ms
            if trace:
                summary = f"Agent trace · {len(trace)} nodes"
                if total_ms is not None:
                    summary += f" · {total_ms} ms"
                status_container.update(label=summary, state="complete", expanded=True)
            else:
                status_container.update(label="Complete", state="complete", expanded=True)
        except RuntimeError as e:
            st.error(str(e))
            full_text = ""

    st.session_state.messages.append({"role": "assistant", "content": full_text, "image_bytes": None})

    # Inline claim decision badge (shown once, directly after the turn that produced it)
    if st.session_state.claim_decision_data:
        _render_claim_badge(st.session_state.claim_decision_data)
        # Keep data in state for the sidebar badge but mark as shown
        st.session_state.claim_decision_data = None

    # Clear per-turn trace state after the status container has rendered it.
    st.session_state.last_trace = None
    st.session_state.last_trace_total_ms = None
