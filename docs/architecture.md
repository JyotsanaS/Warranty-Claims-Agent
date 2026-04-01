# Architecture Description — VoltEdge Automated Warranty & Claims Agent

> This document is written from a Staff-level engineering perspective. It covers not just
> the ML pipeline but the full system: data flow, storage, observability, failure modes,
> cost, and scale. It is a living document — sections are added as each layer is designed.

---

## Table of Contents

1. [External Interface](#1-external-interface)
2. [UI Frontend — Streamlit Chatbot](#2-ui-frontend--streamlit-chatbot)
3. [LLM Gateway — LiteLLM](#3-llm-gateway--litellm)
4. [Session Manager & Agent Orchestration](#4-session-manager--agent-orchestration)
5. [RAG Pipeline](#5-rag-pipeline)
6. [Vision Validation Module](#6-vision-validation-module)
7. [Storage Layer](#7-storage-layer)
8. Non-Functional Requirements (consolidated)

---

## 1. External Interface

### 1.1 Overview

Users interact with the system over **REST HTTP**. Each conversation is a **session** —
a stateful sequence of turns that may include text messages and image uploads. The session
drives the agent through a claim lifecycle: issue description → policy lookup → image
validation → decision.

Agent replies are delivered via **Server-Sent Events (SSE)** streaming. The message
endpoint opens a persistent HTTP connection and pushes typed JSON events as the agent
reasons — token deltas, tool-call status, and the final claim decision — so the client
can render progressively without waiting for the full response.

---

### 1.2 API Surface

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/v1/sessions` | Start a new claim session. Returns `session_id`. |
| `POST` | `/api/v1/sessions/{session_id}/messages` | Send a conversation turn. Accepts text and/or an image (multipart/form-data). **Returns `text/event-stream` (SSE).** |
| `GET`  | `/api/v1/sessions/{session_id}` | Retrieve current session state and claim status. Returns JSON. |
| `DELETE` | `/api/v1/sessions/{session_id}` | Close a session. Triggers PII purge schedule for uploaded images. |

All non-streaming requests and responses are **JSON**. The message endpoint is the sole
SSE endpoint. Every request must include a client-generated or server-assigned `request_id`
for idempotency and tracing.

#### SSE Event Schema

The message endpoint streams a sequence of typed events. Clients must handle all event
types and ignore unknown types for forward compatibility.

```
# 1. Token delta — emitted for every LLM output token
data: {"type": "text_delta", "delta": "Based on", "trace_id": "uuid"}

# 2. Tool call status — emitted when agent invokes RAG or vision
data: {"type": "tool_call", "tool": "rag_retrieval", "status": "running", "trace_id": "uuid"}
data: {"type": "tool_call", "tool": "vision_analysis", "status": "running", "trace_id": "uuid"}

# 3. Tool result — emitted when tool returns (vision result is full structured JSON)
data: {"type": "tool_result", "tool": "vision_analysis", "result": { ... }, "trace_id": "uuid"}

# 4. Claim decision — emitted at most once per turn when agent reaches a verdict
data: {"type": "claim_decision", "status": "approved|rejected|pending|escalated",
       "reason": "...", "policy_clauses": ["§2", "§3.2"], "trace_id": "uuid"}

# 5. Stream end — always the final event
data: {"type": "done", "message_id": "uuid", "trace_id": "uuid"}

# Error event — replaces done on failure
data: {"type": "error", "code": "vision_unreadable|rag_empty|token_limit", "message": "...", "trace_id": "uuid"}
```

---

### 1.3 Request Flow

```
User (browser / mobile / API client)
        │
        │  HTTPS
        ▼
  [ API Gateway ]  ─────────────────────────────────────────────────────
        │          rate limiting (future), TLS termination, trace-id injection
        │
        ▼
  [ FastAPI App Server ]
        │  • Schema validation (Pydantic)
        │  • File validation (format, size, resolution pre-check)
        │  • Attach trace_id to request context
        │
        ├── Text only ──────────────────────► Session Manager
        │                                           │
        └── Text + Image                            ▼
                │                         Agent Orchestrator
                │
                ▼
        [ Object Store ]  (MinIO / S3)
          store raw image, return object_key
                │
                ▼
          Session Manager ──────────────► Agent Orchestrator
```

The message endpoint opens an SSE stream immediately upon receiving the request. The agent
begins reasoning and pushes events as they are produced — token deltas appear within
milliseconds of the first LLM token, tool-call status events keep the client informed
during RAG/vision latency, and the `done` event signals the stream is complete.

FastAPI implements this via `StreamingResponse` with an `async generator` that `yield`s
SSE-formatted strings. The connection stays open for the duration of one agent turn and
closes after the `done` event.

```
User ──POST multipart──► FastAPI
                              │ returns 200 + Content-Type: text/event-stream immediately
                              │
                         async generator begins
                              │
                         ├── yield text_delta  (token by token from LLM)
                         ├── yield tool_call   (RAG / vision invoked)
                         ├── yield tool_result (structured JSON from vision)
                         ├── yield claim_decision (if verdict reached)
                         └── yield done
```

---

### 1.4 Image Handling Rules

| Rule | Detail |
|------|--------|
| Accepted formats | JPEG, PNG, WebP |
| Max file size | 10 MB |
| Min resolution | 200 × 200 px (below this, vision module flags as `too_small`) |
| Storage | Saved to local filesystem under `IMAGE_UPLOAD_DIR` immediately on upload |
| Naming convention | `{session_id}_{type}_{timestamp}.{ext}` — e.g. `sess_abc123_damage_20260331T143022.jpg` |
| Image types | `damage` · `receipt` · `serial` |
| Agent reference | `state.image_local_path` carries the full file path; passed to vision node |
| Retention | Manual purge for prototype; automated purge pipeline is future scope (see §7) |

---

### 1.5 Future Scope — External Interface

| Item | Notes |
|------|-------|
| **Authentication & Authorization** | B2C — JWT-based auth (e.g., Auth0 / Cognito). Session ownership enforced per `user_id`. Not in scope for prototype. |
| **Rate Limiting** | Per-user request cap (suggested: 20 req/min, 100 req/day) enforced at API Gateway layer. Protects against LLM cost abuse. Not in scope for prototype but must be in place before any public launch. |
| **Webhook callbacks** | For async job completion notification as an alternative to SSE for server-to-server integrations. |
| **Object store (S3 / MinIO)** | Replace local filesystem image storage with S3-compatible object store for production (horizontal scaling, encryption, lifecycle policies). Change only `storage/image_store.py`. |
| **SDK / Client libraries** | Typed Python and JavaScript clients generated from OpenAPI spec. |

---

---

## 2. UI Frontend — Streamlit Chatbot

### 2.1 Overview

The frontend is a **Streamlit single-page application**. It is the only channel through
which B2C users interact with the system in the prototype. It communicates exclusively
with the FastAPI backend via the REST API defined in §1.2 — it never touches the agent,
vector store, or object store directly.

Streamlit was chosen because:
- Zero boilerplate for chat UI (`st.chat_message`, `st.chat_input`)
- Built-in file uploader widget — no custom multipart handling on the frontend
- `st.session_state` provides per-browser-tab state without a separate state server
- Fast iteration for a prototype; swap for React/Next.js in production

---

### 2.2 UI Components

| Component | Widget | Behaviour |
|-----------|--------|-----------|
| **New Session** | `st.button("New Session")` | Calls `POST /api/v1/sessions`, stores returned `session_id` in `st.session_state`, clears chat history |
| **Chat History** | `st.chat_message` (loop) | Renders all prior turns from `st.session_state.messages` — user bubbles on right, agent on left |
| **Image Uploader** | `st.file_uploader` (JPEG/PNG/WebP, max 10 MB) | Optional attachment per turn; displayed as thumbnail above the message before sending |
| **Text Input** | `st.chat_input("Describe your issue…")` | Captures free-text; submitting triggers the send flow |
| **Send** | Implicit submit on `st.chat_input` | Opens SSE stream to `POST /api/v1/sessions/{session_id}/messages` with text + optional image |
| **Streaming Agent Reply** | `st.write_stream()` inside `st.chat_message("assistant")` | Renders tokens progressively as they arrive from the SSE stream; no spinner blocking the UI |
| **Tool Status Indicator** | `st.status()` context (collapsible) | Shows intermediate steps — "Searching policy…", "Analysing image…" — driven by `tool_call` SSE events |
| **Claim Status Badge** | `st.success / st.warning / st.error` | Rendered when `claim_decision` SSE event is received; updates `st.session_state.claim_status` |

---

### 2.3 Streamlit Session State Schema

Streamlit reruns the entire script on every interaction. All mutable state is stored in
`st.session_state` to survive reruns.

```python
st.session_state = {
    "session_id": str | None,          # UUID returned by POST /api/v1/sessions
    "messages": [                       # local copy of conversation history
        {
            "role": "user" | "assistant",
            "content": str,
            "image_url": str | None,    # thumbnail URL if image was attached
            "timestamp": str            # ISO-8601
        }
    ],
    "claim_status": "pending" | "approved" | "rejected" | None,
    "uploaded_file": bytes | None       # cleared after each send
}
```

---

### 2.4 User Interaction Flow

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit App                      │
│                                                     │
│  ┌─────────────┐                                    │
│  │ New Session │  ──► POST /api/v1/sessions         │
│  └─────────────┘      stores session_id             │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  Chat History (scrollable)                   │   │
│  │  [User]  "My charger screen is cracked"      │   │
│  │  [Agent] "How long ago did this happen?..."  │   │
│  │  [User]  "Last week" + 📎 damage_photo.jpg   │   │
│  │  [Agent] "Analysing image…  Claim approved." │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌───────────────────────┐  ┌──────────────────┐    │
│  │ 📎 Upload Image       │  │ Type a message…  │    │
│  └───────────────────────┘  └────────┬─────────┘    │
│                                      │ Send         │
└──────────────────────────────────────┼──────────────┘
                                       │
                          POST /api/v1/sessions/{id}/messages
                          multipart: { text, image? }
                                       │
                                 FastAPI Backend
                                       │  Content-Type: text/event-stream
                                       │
                         ◄── data: {"type":"text_delta", "delta":"Based on..."}
                         ◄── data: {"type":"tool_call", "tool":"rag_retrieval", ...}
                         ◄── data: {"type":"tool_call", "tool":"vision_analysis", ...}
                         ◄── data: {"type":"tool_result", "tool":"vision_analysis", ...}
                         ◄── data: {"type":"claim_decision", "status":"approved", ...}
                         ◄── data: {"type":"done", ...}
```

**Streamlit SSE consumption pattern:**
```python
with st.chat_message("assistant"):
    # st.write_stream() accepts a generator; renders tokens as they arrive
    full_response = st.write_stream(consume_sse(response))
    # After stream ends, check session state for claim_decision
```

---

### 2.5 Error Handling in UI

| Scenario | UI Behaviour |
|----------|-------------|
| No active session (user sends without clicking New Session) | Warning banner: "Please start a new session first." |
| API unreachable / 5xx | `st.error("Service unavailable. Please try again.")` — does not clear chat history |
| Image too large or wrong format | Client-side rejection via `file_uploader` type/size config before API call |
| Agent returns low-quality image warning | Agent message displayed as normal turn; claim status remains Pending |
| Session expired (API returns 404) | `st.warning("Session expired.")` + auto-prompt to start new session |

---

### 2.6 Future Scope — UI

| Item | Notes |
|------|-------|
| **Auth / Login screen** | Login page before session creation once JWT auth is added |
| **Claim history view** | Sidebar listing past sessions and their outcomes per authenticated user |
| **Production frontend** | Replace Streamlit with React/Next.js for full UX control, mobile responsiveness, and CI/CD integration. React's `EventSource` API maps cleanly onto the SSE contract already defined. |

---

---

## 3. LLM Gateway — LiteLLM

### 3.1 Overview

The LLM Gateway is the **single module** responsible for all model interactions in the
system — text generation, vision (multimodal), and embeddings. No application code
outside `gateway/llm_gateway.py` ever imports a provider SDK (`groq`, `openai`,
`anthropic`, `sentence_transformers`) directly.

**What lives in the gateway:**
- Text generation clients (`fast_llm`, `main_llm`)
- Vision client (`vision_llm`)
- Embedding function (`get_embeddings`)
- Global token & cost tracker
- Retry and fallback configuration

**What this buys us:** swapping any model — text, vision, or embedding — requires
changing one environment variable. Zero code changes, zero re-deployment of business
logic.

```
Any LangGraph node / RAG pipeline / Vision pipeline
        │  calls gateway functions only
        ▼
  gateway/llm_gateway.py   (LiteLLM inside)
        │
        ├── Text generation ──► groq/llama-3.3-70b-versatile   (today)
        │                  ──► openai/gpt-4o                    (.env swap)
        │                  ──► ollama/llama3                    (.env swap, fully local)
        │
        ├── Vision ──────────► groq/meta-llama/llama-4-scout-17b-16e-instruct  (today)
        │                  ──► openai/gpt-4o                    (.env swap)
        │
        └── Embeddings ──────► local/BAAI/bge-small-en-v1.5    (today, free, in-process)
                           ──► openai/text-embedding-3-small    (.env swap)
                           ──► huggingface/BAAI/bge-small-en-v1.5  (.env swap)
```

---

### 3.2 Gateway Module Design

```python
# gateway/llm_gateway.py

import litellm
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatLiteLLM
from app.config import settings
from gateway.cost_tracker import track_usage

# ── Global LiteLLM callback — fires after EVERY model call ────────────────────
litellm.success_callback = [track_usage]
litellm.set_verbose = False

# ── Text generation clients ───────────────────────────────────────────────────
# Used by: router, claim_state_updater,
#          query_reformulator, evidence_planner
fast_llm = ChatLiteLLM(
    model=settings.LLM_FAST_MODEL,       # "groq/llama-3.1-8b-instant"
    streaming=True,
    temperature=0.0,                      # deterministic — classification nodes
    num_retries=2,
    metadata={"model_role": "fast"},
)

# Used by: agent_respond, empathy_node, claim_decision
main_llm = ChatLiteLLM(
    model=settings.LLM_MAIN_MODEL,       # "groq/llama-3.3-70b-versatile"
    streaming=True,
    temperature=0.3,
    num_retries=2,
    metadata={"model_role": "main"},
)

# ── Vision client ─────────────────────────────────────────────────────────────
# Used by: vision_analysis node only
vision_llm = ChatLiteLLM(
    model=settings.LLM_VISION_MODEL,     # "groq/meta-llama/llama-4-scout-17b-16e-instruct"
    streaming=False,                      # returns structured JSON — no streaming
    temperature=0.0,
    num_retries=2,
    metadata={"model_role": "vision"},
)

# ── Embedding client ──────────────────────────────────────────────────────────
# Used by: RAG indexing pipeline and retrieval queries
# Routes based on LLM_EMBEDDING_MODEL prefix:
#   "local/..."  → sentence-transformers (in-process, free)
#   anything else → litellm.embedding() (API call)

_local_encoder: SentenceTransformer | None = None

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Provider-agnostic embedding function.
    Returns list of float vectors, one per input text.
    """
    model = settings.LLM_EMBEDDING_MODEL   # e.g. "local/BAAI/bge-small-en-v1.5"

    if model.startswith("local/"):
        return _get_local_embeddings(texts, model_name=model.split("/", 1)[1])
    else:
        response = litellm.embedding(model=model, input=texts)
        return [item["embedding"] for item in response["data"]]


def _get_local_embeddings(texts: list[str], model_name: str) -> list[list[float]]:
    global _local_encoder
    if _local_encoder is None:
        _local_encoder = SentenceTransformer(model_name)   # loaded once, cached
    return _local_encoder.encode(texts, convert_to_numpy=True).tolist()
```

---

### 3.3 Model Assignment per Node

| Node | Gateway client | Reason |
|---|---|---|
| `router` | `fast_llm` | First-pass turn classification, cost gate, ~200ms |
| `claim_state_updater` | `fast_llm` | Merges claim memory + normalized claim assertions for claim-relevant turns |
| `query_reformulator` (inside `policy_checker`) | `fast_llm` | Generates 2–3 short query strings |
| `evidence_planner` | `fast_llm` | Generates targeted `VisualCheck` descriptions |
| `empathy_node` | `main_llm` | Tone-aware generation needs stronger reasoning |
| `agent_respond` | `main_llm` | Primary user-facing response, full reasoning |
| `claim_decision` | `main_llm` | Final verdict with policy citation |
| `vision_analysis` | `vision_llm` | Only multimodal node — image + check questions |
| `rag_retrieval` (indexing) | `get_embeddings()` | Encodes policy chunks at index time |
| `rag_retrieval` (query) | `get_embeddings()` | Encodes reformulated queries at runtime |
| `claim_validator` | None (deterministic) | Pure logic — no model call |
| `status_node` | None (deterministic) | Reads state — no model call |

---

### 3.4 Token & Cost Tracking

LiteLLM fires `success_callback` after every completion and embedding call, passing the
full response including `usage` and estimated `cost`. The gateway's `track_usage`
function captures this centrally — no per-node tracking code needed anywhere.

```python
# gateway/cost_tracker.py

import logging
from litellm import completion_cost

logger = logging.getLogger("llm_usage")

# In-process session accumulator (keyed by session_id from call metadata)
_session_totals: dict[str, dict] = {}

def track_usage(kwargs, completion_response, start_time, end_time):
    """
    Called by LiteLLM after every model call (text, vision, embedding).
    Logs structured record and accumulates per-session token count.
    """
    usage  = getattr(completion_response, "usage", None)
    cost   = completion_cost(completion_response=completion_response)
    meta   = kwargs.get("metadata", {})

    record = {
        "model":             kwargs.get("model"),
        "model_role":        meta.get("model_role"),
        "session_id":        meta.get("session_id"),
        "trace_id":          meta.get("trace_id"),
        "node":              meta.get("node"),
        "prompt_tokens":     getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens":      getattr(usage, "total_tokens", 0),
        "cost_usd":          cost,
        "latency_ms":        int((end_time - start_time).total_seconds() * 1000),
    }

    logger.info("llm_usage", extra=record)

    # Accumulate into session total — read by AgentState token budget check
    sid = meta.get("session_id")
    if sid:
        if sid not in _session_totals:
            _session_totals[sid] = {"total_tokens": 0, "total_cost_usd": 0.0}
        _session_totals[sid]["total_tokens"]   += record["total_tokens"]
        _session_totals[sid]["total_cost_usd"] += cost


def get_session_usage(session_id: str) -> dict:
    return _session_totals.get(session_id, {"total_tokens": 0, "total_cost_usd": 0.0})
```

Every LangGraph node passes `session_id`, `trace_id`, and `node` in the `metadata`
field of each `ChatLiteLLM` call. This gives a full per-node, per-session cost breakdown
in structured logs with zero extra instrumentation.

---

### 3.5 What Each Log Record Looks Like

```json
{
  "model":             "groq/llama-3.3-70b-versatile",
  "model_role":        "main",
  "session_id":        "sess_abc123",
  "trace_id":          "trace_xyz789",
  "node":              "agent_respond",
  "prompt_tokens":     1240,
  "completion_tokens": 310,
  "total_tokens":      1550,
  "cost_usd":          0.000217,
  "latency_ms":        1840
}
```

Aggregating these records answers questions like:
- Which node consumes the most tokens per session?
- What is the average cost of a full claim resolution turn?
- How does latency compare across models when we swap providers?

---

### 3.6 Retry & Fallback Configuration

```python
# Prototype: simple per-client retry (already set via num_retries=2 above)

# Production: LiteLLM Router with explicit fallback chain
from litellm import Router

llm_router = Router(
    model_list=[
        # Fast model — primary + fallback
        {"model_name": "fast", "litellm_params": {"model": "groq/llama-3.1-8b-instant"}},
        {"model_name": "fast", "litellm_params": {"model": "openai/gpt-4o-mini"}},

        # Main model — primary + fallback
        {"model_name": "main", "litellm_params": {"model": "groq/llama-3.3-70b-versatile"}},
        {"model_name": "main", "litellm_params": {"model": "openai/gpt-4o"}},

        # Vision model — primary + fallback
        {"model_name": "vision", "litellm_params": {"model": "groq/meta-llama/llama-4-scout-17b-16e-instruct"}},
        {"model_name": "vision", "litellm_params": {"model": "openai/gpt-4o"}},
    ],
    fallbacks=[
        {"fast":   ["fast"]},
        {"main":   ["main"]},
        {"vision": ["vision"]},
    ],
    num_retries=3,
    retry_after=2,
    allowed_fails=2,       # mark a deployment unhealthy after 2 consecutive failures
    cooldown_time=60,      # seconds before retrying an unhealthy deployment
)
```

---

### 3.7 Future Scope — LLM Gateway

| Item | Notes |
|---|---|
| **LiteLLM Proxy Server** | Deploy as a standalone Docker container with virtual API keys. All app services call one internal endpoint. Adds centralised rate limiting, audit logging, and per-team spend controls. |
| **Semantic caching** | LiteLLM supports Redis-backed semantic caching. Near-identical RAG queries return cached completions — significant cost saving for repeated policy questions. |
| **Multi-key rotation** | Rotate between multiple Groq API keys to stay within free-tier RPD limits. Handled transparently by LiteLLM Router load balancing. |
| **Budget hard-limits** | LiteLLM Proxy supports per-key spend caps — enforces `MAX_TOKENS_PER_SESSION` at infrastructure layer, not just in application code. |
| **Hosted embedding API** | Replace `local/BAAI/bge-small-en-v1.5` with `openai/text-embedding-3-small` for horizontal scaling — change one env var. |

---

## 4. Session Manager & Agent Orchestration

### 3.1 Overview

The orchestration layer is built on **LangGraph**. Each user conversation is a LangGraph
**graph execution** — a directed graph of nodes (processing steps) connected by conditional
edges (routing logic). The graph runs once per conversation turn, picks up persisted state
from the previous turn via a checkpointer, and streams events back to the FastAPI SSE
endpoint.

**Why LangGraph over alternatives:**

| Alternative | Why Not |
|---|---|
| LangChain LCEL chains | No stateful graph, no checkpointing, no built-in tool-call loop |
| AutoGen / CrewAI | Multi-agent framework — overkill for a single agent with tools |
| Raw LLM + custom loop | Rebuilds checkpointing, streaming, retry, and tool execution from scratch |
| LangGraph | Explicit graph topology, native `astream_events()` maps 1:1 to our SSE schema, built-in checkpointing, testable nodes |

---

### 3.2 Agent State Schema

LangGraph passes a single typed state object through every node. Nodes read from it and
return partial updates — LangGraph merges them.

```python
class AgentState(TypedDict):
    # Identity
    session_id: str
    trace_id: str
    turn_count: int

    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]  # append-only via reducer
    empathy_prefix: str | None      # set by empathy_node; prepended to next response

    # Evidence (current turn)
    image_object_key: str | None    # S3/MinIO key of uploaded image this turn

    # Tool outputs (accumulated across turns)
    policy_context: list[str]       # chunks returned by RAG retrieval
    damage_report: dict | None      # structured JSON from vision module

    # Multi-claim working memory (see §3.X)
    claim_items: list[ClaimContext]  # all claim items opened this session
    active_claim_index: int          # index into claim_items currently being worked
    context_switch_detected: bool    # claim_state_updater flagged a topic change
    pending_switch_confirmation: bool # agent is waiting for user to confirm switch
    pending_claim_draft: ClaimContext | None  # preserves extracted new-claim details during switch confirmation

    # Session-level decision summary
    policy_clauses: list[str]       # e.g. ["§2 Component Coverage", "§3.2 Env Abuse"]

    # Guardrails
    token_count: int                # cumulative tokens this session
    intent: str | None              # set by router; logged for observability
    router_confidence: float        # if < 0.6 → treat as out_of_scope
```

---

### 3.X ClaimContext — Structured Working Memory

#### Why ClaimContext exists alongside `messages`

The `messages` list stores raw conversation history and is used by the LLM for response
generation. But routing logic, decision gates, and missing-evidence checks need **structured
queryable facts** — not a prose transcript. Without `ClaimContext`, every node must
re-derive the same facts from the full message history on every turn, which is expensive
in tokens and fragile.

`ClaimContext` is the agent's structured working memory for a single claim item. It is
**incrementally extracted and merged** by the `claim_state_updater` node after routing
has already established that the current turn is actually claim-relevant. This avoids
mutating claim state for greetings, policy-only questions, cancellations, and other
administrative turns.

#### ClaimContext Model

```python
class ClaimContext(BaseModel):
    claim_id: str               # UUID — one per claim item in the session

    # What happened
    component: str | None       # "screen", "cable", "logic_board", "housing"
    incident_type: str | None   # "physical_damage", "water_damage",
                                #  "manufacturing_defect", "normal_wear"
    incident_description: str | None   # free-text summary, agent-generated
    incident_date: str | None          # ISO-8601 if user mentioned it

    # Purchase & eligibility
    purchase_date: str | None
    activation_date: str | None

    # User's stated goal
    user_goal: Literal["replacement", "repair", "refund", "policy_inquiry", "unknown"]

    # Evidence checklist — per claim item
    serial_number_provided: bool = False
    proof_of_purchase_provided: bool = False
    damage_photo_provided: bool = False

    # Derived — populated by agent_respond / claim_decision nodes
    applicable_coverage_months: int | None
    is_within_warranty: bool | None
    claim_status: Literal[
        "open",              # collecting info
        "pending_evidence",  # waiting for uploads
        "under_review",      # all evidence in, agent deciding
        "approved", "rejected", "escalated"
    ]

    # Audit trail
    context_version: int        # incremented on every update
    created_at_turn: int
    last_updated_turn: int
```

#### Field Merge Rules

The `claim_state_updater` applies these rules when merging new information into an
existing `ClaimContext`. The updater receives:

- the latest user message
- the current active `ClaimContext`
- a small recent window of claim-relevant turns
- image / attachment metadata for the current turn

It does not need the full transcript on every turn, because claim memory already stores
the structured facts that matter operationally.

| Scenario | Rule |
|---|---|
| User provides a new value for an empty field | WRITE — populate the field |
| User provides a value that contradicts an existing field ("actually 6 months ago") | OVERWRITE with latest value + increment `context_version` + log conflict to trace |
| User adds complementary info ("and also the cable") | APPEND — do not discard prior component; triggers multi-claim check (see below) |
| User goes silent on a previously filled field | PRESERVE — never set a populated field back to `None` |

#### Multi-Component Claims

One session can hold **multiple independent `ClaimContext` items** in `state.claim_items`.
Each has its own evidence checklist and decision outcome. The `active_claim_index` pointer
tracks which one is currently being worked.

When `claim_state_updater` detects a new component or a different incident type in the
user's message, it evaluates:

```python
def should_confirm_switch(claim: ClaimContext) -> bool:
    has_partial_info = any([
        claim.component is not None,
        claim.incident_type is not None,
        claim.incident_date is not None,
    ])
    evidence_not_yet_submitted = not (
        claim.proof_of_purchase_provided and claim.damage_photo_provided
    )
    return has_partial_info and evidence_not_yet_submitted
    # Only ask the user if we have partial info but no evidence uploaded yet.
    # If evidence was already submitted, open the new claim automatically.
```

**Context switch flow:**

```
User: "Also my cable stopped charging"
        │
        ▼
[claim_state_updater]
  detects: new component mentioned, different from active claim
        │
   should_confirm_switch(active_claim)?
        │
        ├── YES  (partial info filled, no evidence submitted yet)
        │     state.context_switch_detected = True
        │     state.pending_switch_confirmation = True
        │     agent asks:
        │       "I see you're also having a cable issue. Would you like to:
        │        (A) Open a separate claim for the cable while keeping your
        │            screen damage claim open, or
        │        (B) Treat both as part of the same incident?"
        │
        │     User: "separate"
        │       → extracted new-claim details stored in state.pending_claim_draft
        │       → if user confirms "separate", pending_claim_draft appended to claim_items
        │       → active_claim_index updated to new item
        │       → pending_switch_confirmation = False
        │
        │     User: "same incident"
        │       → pending_claim_draft merged into active ClaimContext
        │       → pending_switch_confirmation = False
        │
        └── NO  (evidence already submitted OR claim brand new)
              → new ClaimContext created and appended automatically
              → active_claim_index = len(claim_items) - 1
              → no interruption to the conversation
```

**Temporary detour (policy question mid-claim):**
If the router classifies a turn as `policy_inquiry` while a claim is in progress,
`claim_state_updater` is skipped entirely unless the policy question explicitly advances
claim evidence. The active `ClaimContext` is preserved untouched. No confirmation is
required — the user is just asking a question, not abandoning their claim.

#### Data Retention Policy for Claim Items

| Scope | Policy |
|---|---|
| Within a session | All `ClaimContext` items (active + completed) are preserved for the full session lifetime. Items are never deleted — only their `claim_status` changes. |
| Cross-session | Governed by NFR-10: images purged 30 days after session close. `ClaimContext` metadata (no PII) may be retained longer for analytics. |
| User requests cancellation | `claim_status` set to `"escalated"` with reason `"user_cancelled"`. Data retained for audit. |

---

### 3.3 Intent Taxonomy

The `router` node classifies every incoming user turn into one of the following intents
before any expensive operation (RAG, vision, main LLM) is invoked.

| Intent | Example Utterances | Downstream Path |
|---|---|---|
| `greeting` | "Hi", "How does this work?" | `greeting_node` — no RAG, no vision |
| `policy_inquiry` | "Is water damage covered?" | RAG → `agent_respond` |
| `claim_initiation` | "I want to file a claim" | RAG → `agent_respond` |
| `issue_description` | "My screen is cracked" | RAG → optional vision → `agent_respond` |
| `evidence_submission` | User uploads image | RAG + vision → `agent_respond` → `decision_gate` |
| `clarification_response` | "It happened last week" | `agent_respond` only (reuse existing `policy_context`) |
| `status_query` | "What's my claim status?" | `status_node` — reads state, no LLM call |
| `frustration` | "This is ridiculous" | `empathy_node` → re-route by session state |
| `escalation_request` | "I want to talk to a human" | `escalation_node` |
| `cancellation` | "Never mind, cancel it" | `cancellation_node` |
| `out_of_scope` | "What's the weather?" | `fallback_node` — politely redirect |

**Why explicit intent classification matters:**
- `greeting` and `out_of_scope` turns never touch RAG or vision — significant cost saving
  at scale (1M users, even 20% such turns = hundreds of thousands of avoided LLM calls)
- Every turn has a logged `intent` field — enables product analytics ("what % of users
  open with a direct claim vs. a question?")
- Routing logic is a pure, testable function — not buried in a system prompt
- Claim state is mutated only for turns that actually progress a claim, reducing false
  claim creation and stale state accumulation

---

### 3.4 Router Node Design

The router uses a **cheap, fast model** (Claude Haiku / GPT-4o-mini) with structured
output. It is the first node in the graph and completes in ~200ms before any downstream
work begins.

```python
class RouterOutput(BaseModel):
    intent: Literal[
        "greeting", "policy_inquiry", "claim_initiation", "issue_description",
        "evidence_submission", "clarification_response", "status_query",
        "frustration", "escalation_request", "cancellation", "out_of_scope"
    ]
    confidence: float           # 0.0–1.0; below 0.6 overrides intent → out_of_scope
    requires_policy_lookup: bool
    requires_vision: bool       # True only if image is present AND claim-relevant
```

`missing_evidence` is no longer owned by the router — it lives in `ClaimContext.serial_number_provided`,
`proof_of_purchase_provided`, and `damage_photo_provided` (see §3.X). The router reads these
fields from `state.claim_items[state.active_claim_index]` to determine what to ask for next.

**Special case:** If `state.pending_switch_confirmation is True`, the router short-circuits
to `confirmation_handler` regardless of the classified intent — the agent must resolve the
pending context switch before processing any new claim work.

#### Why Router Comes First

The router is intentionally the first interpretation step. From a behavioural standpoint,
this prevents the system from doing claim extraction on turns like:

- "hi"
- "what's my claim status?"
- "cancel this"
- "is water damage covered?"
- "this is unacceptable"

If extraction runs before routing, those turns can accidentally mutate claim memory,
append stale `UserClaim` entries, or force policy questions through a claim-shaped path.
Putting the router first turns it into a cost gate and a state-integrity gate.

---

### 3.4A Claim State Updater Node

`claim_state_updater` replaces the older `context_extractor` + `claim_extractor` split.
It runs only for claim-progressing intents:

- `claim_initiation`
- `issue_description`
- `evidence_submission`
- `clarification_response`

It does **two jobs in one pass**:

1. Updates structured working memory for the active claim (`ClaimContext`)
2. Produces normalized `UserClaim` assertions for downstream policy and vision reasoning

This merge is intentional. In practice, the old split caused overlapping interpretation
logic, extra latency, and state mutation before the system understood what kind of turn it
was handling.

**Inputs**

- latest user message
- active `ClaimContext` (if any)
- small recent window of claim-relevant history
- current-turn attachment metadata

**Outputs**

- updated `claim_items`
- updated `active_claim_index`
- updated `user_claims`
- `context_switch_detected`
- `pending_switch_confirmation`
- `pending_claim_draft` when a new claim has been detected but not yet confirmed

**Design principle:** extract only what changes behaviour downstream. The updater should
not be a generic transcript summariser. It should return stable, mergeable fields that
directly affect routing, RAG, evidence collection, and decisioning.

---

### 3.5 Empathy Wrapper (Frustration Handling)

When the router classifies a turn as `frustration`, the graph routes to `empathy_node`
**before** any substantive processing. The empathy node does not generate the full
response — it generates only a short (1–2 sentence) tone-aware acknowledgment and writes
it to `state.empathy_prefix`. A second conditional edge then re-routes based on session
state.

```
[empathy_node]
  model: main LLM, focused empathy prompt
  output: state.empathy_prefix = "I completely understand how frustrating this must be..."
      │
      ▼
[empathy_router]  ← conditional edge on session state
      │
      ├── claim_status in {approved, rejected} AND user still upset
      │         └──► [escalation_node]
      │               (decision is final; a human agent is the right next step)
      │
      ├── turn_count > 1 AND policy_context already populated
      │         └──► [agent_respond]
      │               (skip RAG — context is fresh; prepend empathy_prefix)
      │
      ├── turn_count > 1 AND policy_context empty
      │         └──► [rag_retrieval] → [agent_respond]  (prepend empathy_prefix)
      │
      └── turn_count ≤ 1  (very early in session)
                └──► [rag_retrieval] → [agent_respond]
                      (empathise + explain the claims process from scratch)
```

**Streaming behaviour:** `empathy_prefix` tokens are emitted as `text_delta` SSE events
first. The downstream node's response continues the same open stream. The user sees one
seamless, empathetic reply — never two separate messages.

---

### 3.6 Full Graph Topology

```
START
  │
  ▼
[router]  ← llama-3.1-8b-instant, structured output, ~200ms
  first-pass turn classification
  │
  ├── pending_switch_confirmation is True (any intent)
  │         └──► [confirmation_handler]
  │               resolves claim_items / active_claim_index / pending_claim_draft
  │               clears pending_switch_confirmation ──────────────────────► END
  │
  ├── greeting ────────────────────────────────────► [greeting_node] ────────► END
  ├── out_of_scope ──────────────────────────────────► [fallback_node] ───────► END
  ├── escalation_request ────────────────────────────► [escalation_node] ─────► END
  ├── cancellation ──────────────────────────────────► [cancellation_node] ───► END
  ├── status_query ── reads state, no LLM call ────────► [status_node] ────────► END
  │
  ├── frustration ───────────────────────────────────► [empathy_node]
  │                                                          │
  │                                                  [empathy_router] (§3.5)
  │
  ├── policy_inquiry ───────────────────────────────► [policy_checker]
  │                                                    inquiry mode:
  │                                                    • query built from user question
  │                                                    • active ClaimContext used only as optional disambiguation
  │                                                    • does not mutate claim_items
  │
  └── claim_initiation / issue_description /
      evidence_submission / clarification_response
              │
              ▼
      [claim_state_updater]  ← lightweight LLM (llama-3.1-8b-instant), claim turns only
        • merges latest structured claim facts into active ClaimContext
        • updates / normalizes UserClaim assertions for downstream reasoning
        • detects context switch without discarding extracted new-claim details
        • writes pending_claim_draft when waiting for user confirmation
              │
              ▼
      [policy_checker]   ← RAG per UserClaim → PolicyCoverage per claim
        • retrieves policy context for each claim (query reformulation per claim)
        • sets claim.policy_coverage (covered / excluded / needs_evidence)
        • claims excluded by policy → claim_verdict = "excluded_by_policy" immediately
        • no vision call needed for fully excluded claims
              │
        has_image AND has policy-valid claims needing visual validation?
              │
              ├── YES
              │     ▼
              │  [evidence_planner]
              │    for each policy-valid claim:
              │    generate specific VisualChecks (check_description + expected_finding)
              │    e.g. "Is a physical crack visible on the display?" expected: True
              │         "Are there signs of liquid contact?"          expected: False
              │     │
              │     ▼
              │  [vision_analysis]   (meta-llama/llama-4-scout-17b-16e-instruct)
              │    one model call: image + all VisualChecks
              │    fills in: finding, observation, claim_supported, confidence
              │     │
              │     ▼
              │  [claim_validator]
              │    per claim: policy_coverage + visual checks → final verdict
              │    "valid" | "invalid" | "inconclusive" | "excluded_by_policy"
              │
              └── NO (no image yet, or all claims already policy-excluded)
                        │
                        ▼
                  [agent_respond]   (llama-3.3-70b-versatile)
                    • summarises claim verdicts found so far
                    • asks for missing evidence if claims are "awaiting_evidence"
                    • streams response via SSE
                        │
                        ▼
                  [decision_gate]
                    all_claims_resolved?
                      ├── YES → [claim_decision] → END
                      └── NO  → END (await next turn)
```

---

### 3.7 Checkpointing — Session Persistence

LangGraph's checkpointer persists the full `AgentState` after every node execution.
Each `session_id` maps directly to a LangGraph `thread_id`. On the next turn, the graph
resumes with the full prior state — conversation history, policy context, damage report,
claim status — without any re-fetching.

| Environment | Checkpointer | Notes |
|---|---|---|
| Prototype (Docker Compose) | `MemorySaver` | In-process; state lost on restart. Sufficient for demo. |
| Production | `AsyncPostgresSaver` | LangGraph's official Postgres backend. Durable, horizontally scalable. |

The FastAPI Session Manager is thin — it stores only the `session_id → thread_id`
mapping and session metadata (created_at, claim_status summary for API responses).
All conversational state lives in the LangGraph checkpointer.

---

### 3.8 Streaming — LangGraph to SSE

LangGraph's `graph.astream_events()` yields structured events as each node executes.
The FastAPI SSE generator consumes these and translates them into the SSE event schema
defined in §1.2.

| `astream_events` event | Node | Translated SSE event |
|---|---|---|
| `on_chat_model_stream` | any LLM node | `text_delta` |
| `on_tool_start` | `rag_retrieval` | `tool_call {tool: "rag_retrieval", status: "running"}` |
| `on_tool_start` | `vision_analysis` | `tool_call {tool: "vision_analysis", status: "running"}` |
| `on_tool_end` | `rag_retrieval` | `tool_result {tool: "rag_retrieval", result: [...]}` |
| `on_tool_end` | `vision_analysis` | `tool_result {tool: "vision_analysis", result: {...}}` |
| `on_chain_end` | `claim_decision` | `claim_decision {status, reason, policy_clauses}` |
| `on_chain_end` | graph END | `done {message_id, trace_id}` |
| any error | any node | `error {code, message, trace_id}` |

---

### 3.9 Token Budget Enforcement

Every session has a configurable `max_tokens` cap (default: 8,000 tokens across the
full session). The `input_processor` checks `state.token_count` before invoking any LLM
node. If the budget is exceeded:

1. Stream an `error` SSE event with `code: "token_limit"`
2. Emit a user-visible message: "This session has reached its limit. Please start a new session."
3. Set `claim_status = "escalated"` if a claim was in progress

This prevents runaway costs from adversarial or unusually long sessions.

---

### 3.10 Future Scope — Agent Orchestration

| Item | Notes |
|---|---|
| **Postgres checkpointer** | Replace `MemorySaver` with `AsyncPostgresSaver` for durable session state in production |
| **Parallel RAG + Vision** | Fan-out both tool nodes simultaneously using LangGraph's `Send` API to reduce latency |
| **Human-in-the-loop** | LangGraph's `interrupt_before` mechanism to pause graph execution pending human review for escalated claims |
| **Multi-turn claim drafts** | Allow users to resume a saved claim across browser sessions once auth is in place |

---

---

## 5. RAG Pipeline

### 4.1 Overview

The RAG pipeline retrieves relevant warranty policy clauses to ground every agent
response. It runs inside the `rag_retrieval` node of the LangGraph graph (§3.6) for
intents that require policy reasoning. It never runs for `greeting`, `out_of_scope`,
`status_query`, or `cancellation` — cost-free turns.

**Stack:**

| Component | Choice | Reason |
|---|---|---|
| Vector store | Pinecone (serverless, free tier) | Managed, low-ops, supports metadata filtering |
| Embedding model | `local/BAAI/bge-small-en-v1.5` via LLM Gateway (`get_embeddings()`) | Routed through gateway — swap to any provider via `LLM_EMBEDDING_MODEL` env var |
| LLM for query reformulation | `fast_llm` via LLM Gateway | `llama-3.1-8b-instant` today; swappable via `LLM_FAST_MODEL` |
| Chunking | Structure-aware with context prefix injection | Policy doc has tables and numbered lists — naive token splits destroy meaning |

---

### 4.2 LLM Model Assignments (All via Groq)

All LLM calls in this system use the Groq API (free tier). Models are configured via
environment variables so they can be swapped without code changes.

| Node | Model | Env var | Reason |
|---|---|---|---|
| Router, claim_state_updater, query_reformulator | `llama-3.1-8b-instant` | `GROQ_FAST_MODEL` | 560 t/s, 14,400 RPD quota, tool calling, ~200ms latency |
| agent_respond, empathy_node, claim_decision | `llama-3.3-70b-versatile` | `GROQ_MAIN_MODEL` | 70B parameters, strong reasoning, tool calling, 131K context |
| vision_analysis | `meta-llama/llama-4-scout-17b-16e-instruct` | `GROQ_VISION_MODEL` | Only vision-capable model on Groq free tier, 750 t/s, up to 5 images/req |

**Rate limit awareness:** The free tier allows 1,000 RPD for the main and vision models,
14,400 RPD for the fast model. The token budget cap (NFR-11, `MAX_TOKENS_PER_SESSION=8000`)
provides a secondary guard against quota exhaustion.

---

### 4.3 Chunking Strategy — Structure-Aware with Context Prefix Injection

The policy document (`sample-policy.md`) contains three distinct content types, each
requiring a different split strategy. Naive fixed-size token chunking is explicitly
rejected — it breaks table rows away from their column headers and list items away from
their parent section, destroying retrieval quality.

| Content type | Split rule | Example |
|---|---|---|
| Prose paragraphs | One paragraph = one chunk | §1 coverage period, §1 Normal Use definition |
| Table rows | One row = one chunk, prefixed with column headers | Each component row in §2 |
| Numbered / bullet list items | One item = one chunk, prefixed with parent section | Each exclusion in §3, each requirement in §4 |

**Context prefix injection** makes every chunk self-contained before embedding. A chunk
that reads `"12 Months — dead pixels (>5) or total backlight failure"` in isolation is
ambiguous. With a prefix it becomes unambiguous:

```
[Section: Specific Component Coverage]
Component: LED Display / Touchscreen | Coverage: 12 Months |
Notes: Only covers dead pixels (>5) or total backlight failure.
```

This technique (similar to Anthropic's Contextual Retrieval) significantly improves
retrieval precision for short, dense policy clauses.

**Expected chunk count for `sample-policy.md`: ~13 chunks**

| Section | Chunks |
|---|---|
| §1 General Warranty (prose) | 2 |
| §2 Component Coverage (table, 4 rows) | 4 |
| §3 Standard Exclusions (5 list items) | 5 |
| §4 Claim Validation Requirements (3 items) | 3 |

---

### 4.4 Chunk Metadata Schema

Each chunk is upserted to Pinecone with the following metadata. Metadata enables
**pre-filtered retrieval** — narrowing the search space before semantic scoring.

```python
{
    "chunk_id": "voltedge_v1_sec2_row2",         # stable ID for deduplication
    "document_id": "voltedge_warranty_v1",
    "section_number": "2",
    "section_title": "Specific Component Coverage",
    "chunk_type": "table_row",                    # paragraph | table_row | list_item
    "component": "LED Display / Touchscreen",     # null for non-component chunks
    "coverage_months": 12,                        # numeric — enables range filter
    "is_exclusion": False,                        # True for §3 chunks
    "policy_version": "v1",                      # namespace-based versioning
    "text": "...",                                # original text (for display / citation)
    "token_count": 52
}
```

**How metadata filtering is used at query time:**
- `ClaimContext.component = "LED Display"` → pre-filter `component = "LED Display / Touchscreen"`
  before semantic search → higher precision, fewer irrelevant chunks scored
- Intent `escalation` or explicit exclusion question → pre-filter `is_exclusion = True`
- Policy version pinning → always query the active `policy_version` namespace

---

### 4.5 Pinecone Index Configuration

| Parameter | Value |
|---|---|
| Index name | `voltedge-policy` |
| Embedding dimensions | 384 (BAAI/bge-small-en-v1.5) |
| Distance metric | cosine |
| Active namespace | `warranty_policy_v1` |
| Prototype tier | Serverless free tier |

**Namespace-based policy versioning:** When the policy document is updated, a new
namespace (`warranty_policy_v2`) is populated and tested before the application config
is switched. The old namespace remains queryable during cutover. Zero downtime.

---

### 4.6 Query Reformulation

Raw user messages are poor vector queries. `"It fell in the pool"` has low cosine
similarity to `"submersion in liquid"`. The `query_reformulator` step runs inside the
`rag_retrieval` node and generates targeted search strings from the user's message
combined with the structured `ClaimContext`.

**Model:** `llama-3.1-8b-instant` (fast model — this is a simple generation task)

**Input to the reformulator:**
```
User message: "It fell in the pool last summer"
ClaimContext: { component: "LED Display", incident_type: null, purchase_date: "2024-01" }
```

**Output — 2–3 reformulated queries:**
```python
[
    "water damage coverage LED display warranty",
    "environmental abuse exclusion submersion liquid",
    "LED display touchscreen claim water incident"
]
```

**Why multiple queries?** A single reformulation can miss synonyms or related clauses.
Three queries with deduplication gives much better recall across a small corpus without
significant cost (3 × Pinecone query = milliseconds).

---

### 4.7 Full Retrieval Flow

```
ClaimContext + user message
        │
        ▼
[query_reformulator]  (llama-3.1-8b-instant)
  generates 2–3 targeted search queries
        │
        ▼ (per query, in parallel)
[Pinecone query]
  embedding: BAAI/bge-small-en-v1.5 (local)
  top_k = 5
  metadata filter: { component, policy_version }
  namespace: warranty_policy_v1
        │
        ▼
[deduplication + merge]
  union by chunk_id
  sort by similarity score descending
  return top 5 unique chunks
        │
        ▼
[threshold check]
  if max(score) < RAG_SIMILARITY_THRESHOLD (0.75):
      return RAGResult(chunks=[], fallback=True)
  else:
      return RAGResult(chunks=[...], fallback=False)
        │
        ▼
  written to state.policy_context
  consumed by agent_respond node
```

---

### 4.8 Fallback Handling (NFR-14)

When retrieval returns no chunks above the similarity threshold, the agent must **not
hallucinate a policy ruling**. The `agent_respond` node checks `state.policy_context`
for the `fallback=True` flag and responds accordingly:

```
Agent: "I wasn't able to find a specific policy clause that directly covers your
        situation. To make sure you get the right answer, I'd like to connect you
        with a VoltEdge specialist. Would that work for you?"
```

`claim_status` is set to `"escalated"` with `decision_reason = "no_policy_context"`.
This is logged and surfaced in observability as a signal to improve the policy corpus.

---

### 4.9 Indexing Pipeline (One-Time / On Policy Update)

The indexing pipeline runs at system startup if the Pinecone namespace is empty, and
manually on policy document updates. It is **not invoked per request**.

```
sample-policy.md
        │
        ▼
[markdown parser]
  identifies: H2 headers, tables, numbered lists, bullet lists
        │
        ▼
[structure-aware chunker]
  table  → split by row, prefix with column headers
  list   → split by item, prefix with parent section header
  prose  → split by paragraph
        │
        ▼
[context prefix injector]
  prepend "[Section: {title}]" breadcrumb to each chunk text
        │
        ▼
[metadata extractor]
  populate: section, chunk_type, component, coverage_months, is_exclusion
        │
        ▼
[BAAI/bge-small-en-v1.5 encoder]  (local, sentence-transformers)
  encode prefixed chunk text → 384-dim vector
        │
        ▼
[Pinecone upsert]
  namespace: warranty_policy_v1
  batch size: 100 (well within our ~13 chunk corpus)
```

---

### 4.10 Future Scope — RAG Pipeline

| Item | Notes |
|---|---|
| **Hybrid search (BM25 + dense)** | Pinecone supports sparse+dense natively. Improves recall for exact legal terms (`"Acts of God"`, `"3000 Joules"`). Intentionally deferred from prototype. |
| **Cross-encoder re-ranking** | Add a `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranker after retrieval for larger policy corpora. Not needed at ~13 chunks. |
| **Hosted embedding API** | Change `LLM_EMBEDDING_MODEL=openai/text-embedding-3-small` in `.env` — the gateway handles the rest. No code change required. |
| **Multi-document support** | Index multiple policy versions or product-specific policies under separate namespaces; route queries by product SKU from `ClaimContext`. |
| **Retrieval evaluation** | Track faithfulness and relevancy scores per query (see EVALS.md) to monitor chunk quality drift as the policy evolves. |

---

---

## 6. Vision Validation Module

### 5.1 Overview

The vision module is a **three-node reasoning pipeline** within the LangGraph graph (§3.6):
`evidence_planner` → `vision_analysis` → `claim_validator`. It is preceded upstream by
`claim_state_updater` and `policy_checker`, which together determine *what to validate*
before the vision model is ever called.

**Core principle:** Vision is claim-driven, not image-driven. The system never asks
"what's in this image?" It asks "does this image support or contradict what the user
claimed?" The visual checks are dynamically generated per conversation — not a fixed
set of hardcoded fields.

**Model:** `meta-llama/llama-4-scout-17b-16e-instruct` via Groq (`GROQ_VISION_MODEL`).

---

### 5.2 Data Structures

#### UserClaim — extracted from conversation

```python
class UserClaim(BaseModel):
    claim_id: str                  # UUID, one per distinct claim statement
    component: str | None          # "screen", "cable", "housing", "logic_board"
    assertion: str                 # "physical_damage", "normal_use",
                                   # "manufacturing_defect", "within_warranty_period",
                                   # "no_liquid_contact", "no_tampering"
    user_statement: str            # verbatim quote from conversation
    requires_visual_validation: bool

    # Set by policy_checker
    policy_coverage: PolicyCoverage | None

    # Set by evidence_planner (list of checks to run) and filled by vision_analysis
    visual_checks: list[VisualCheck]

    # Set by claim_validator
    claim_verdict: Literal[
        "valid",              # policy covers it + visual evidence supports it
        "invalid",            # visual evidence contradicts the claim
        "inconclusive",       # image insufficient to decide
        "excluded_by_policy", # policy voids before vision even runs
        "awaiting_evidence"   # no image submitted yet
    ]
    verdict_reason: str | None
```

#### PolicyCoverage — result of RAG check per claim

```python
class PolicyCoverage(BaseModel):
    is_covered: bool
    coverage_clause: str | None    # e.g. "§2 LED Display — 12 months"
    exclusion_clause: str | None   # e.g. "§3.2 Environmental Abuse"
    visual_evidence_required: list[str]  # what the policy says needs to be visually confirmed
```

#### VisualCheck — dynamic, generated per claim by evidence_planner

```python
class VisualCheck(BaseModel):
    check_description: str         # natural language question for the vision model
                                   # e.g. "Is a physical crack visible on the display?"
                                   #      "Does the image show any signs of liquid contact?"
                                   #      "Are device seals or screws visibly tampered with?"
    expected_finding: bool         # True  = presence of this supports the claim
                                   # False = absence of this supports the claim
    # Filled by vision_analysis
    finding: bool | None           # what vision actually observed
    observation: str | None        # factual one-sentence description of what was seen
    claim_supported: bool | None   # finding matches expected_finding?
    confidence: float | None       # 0.0–1.0
```

#### VisionReport — output of vision_analysis node

```python
class VisionReport(BaseModel):
    image_quality: Literal[
        "acceptable", "too_blurry", "too_dark", "too_small", "unreadable"
    ]
    image_type: Literal[
        "damage_photo", "receipt", "serial_number", "unrelated", "unclear"
    ]
    completed_checks: list[VisualCheck]   # all checks with findings filled in
    overall_confidence: float
    cannot_assess_reason: str | None      # set if image_quality != "acceptable"

    # Receipt / serial extraction (populated when image_type matches)
    purchase_date: str | None             # ISO-8601 if readable from receipt
    product_name: str | None
    serial_number: str | None
```

---

### 5.3 Full Pipeline Flow

```
state.user_claims  (set by claim_state_updater — §5.4)
        │
        ▼
[policy_checker]   (llama-3.1-8b-instant + RAG per claim)
  For each UserClaim:
    • query reformulator generates targeted RAG query from claim assertion + component
    • retrieve policy chunks from Pinecone
    • determine: is_covered? coverage_clause? exclusion_clause?
    • claims with exclusion_clause → claim_verdict = "excluded_by_policy" immediately
    • no vision call for excluded claims — saves latency and cost
        │
   has policy-valid claims AND image submitted this turn?
        │
        ├── YES
        │     ▼
        │  [evidence_planner]   (llama-3.1-8b-instant)
        │    For each policy-valid claim:
        │      generate VisualChecks (check_description + expected_finding)
        │      driven by: claim.assertion + policy_coverage.visual_evidence_required
        │
        │    Example outputs:
        │      Claim "physical_damage to screen":
        │        → VisualCheck("Is a visible crack or fracture on the display?", True)
        │
        │      Claim "no_liquid_contact":
        │        → VisualCheck("Are there visible signs of liquid contact?", False)
        │
        │      Claim "no_tampering":
        │        → VisualCheck("Do device seals or screws appear tampered with?", False)
        │
        │      Claim "purchase within 12 months" (receipt image):
        │        → VisualCheck("Is a purchase date readable on the receipt?", True)
        │     │
        │     ▼
        │  [vision_analysis]   (meta-llama/llama-4-scout-17b-16e-instruct)
        │
        │    Pre-processing:
        │      • fetch image from MinIO via pre-signed URL
        │      • resize to max 1024px longest edge (cost control)
        │      • encode to base64
        │
        │    Model call — one call, all checks in a single prompt:
        │      System: "You are a warranty claim image analyst. Answer each check
        │               question factually based only on what is visually present.
        │               Do not infer or assume. Return JSON matching VisionReport."
        │      User:   [check_description list] + base64 image
        │
        │    Output: VisionReport with completed_checks filled in
        │
        │    On Pydantic parse failure:
        │      VisionReport(image_quality="unreadable",
        │                   cannot_assess_reason="parse_failure",
        │                   completed_checks=[])
        │     │
        │     ▼
        │  [claim_validator]   (deterministic — no LLM call)
        │    For each claim:
        │      all checks claim_supported == True → claim_verdict = "valid"
        │      any check claim_supported == False → claim_verdict = "invalid"
        │      any check confidence < 0.6         → claim_verdict = "inconclusive"
        │      image_quality != "acceptable"      → claim_verdict = "inconclusive"
        │                                            (ask for better image)
        │    Updates state.user_claims[i].claim_verdict and verdict_reason
        │    Sets state.all_claims_resolved
        │
        └── NO (no image, or all claims excluded)
                  ▼
            [agent_respond]

```

---

### 5.4 Claim State Updater Node

`claim_state_updater` runs after the router and only on claim-progressing turns. It
replaces the older `context_extractor` + `claim_extractor` split with one merged pass
that updates `ClaimContext` and normalized `UserClaim` assertions together.

**Model:** `llama-3.1-8b-instant`

**Why this merged design is better:**

- avoids mutating claim state for greetings, cancellations, and policy-only questions
- removes duplicate extraction logic across two adjacent nodes
- preserves a cleaner behavioural boundary: route first, then update claim memory
- lets multi-claim switch handling store a `pending_claim_draft` instead of asking the
  user to restate information already extracted

**Example merged update:**

```
Conversation so far:
  Turn 1: "My screen cracked during normal use"
  Turn 2: "I never dropped it in water"
  Turn 3: "I bought it about 8 months ago"

Updated ClaimContext:
  {
    component: "screen",
    incident_type: "physical_damage",
    incident_description: "screen cracked during normal use",
    purchase_date: "2024-08-xx",
    damage_photo_provided: False
  }

Normalized UserClaims:
  [
    { assertion: "physical_damage", component: "screen",
      user_statement: "screen cracked during normal use",
      requires_visual_validation: True },

    { assertion: "no_liquid_contact", component: "screen",
      user_statement: "I never dropped it in water",
      requires_visual_validation: True }
  ]
```

Claims still accumulate across turns, but they are merged against structured claim memory
rather than appended blindly from the full transcript on every turn.

---

### 5.5 Reasoning Example — End to End

**User journey:**
```
Turn 1: "My screen cracked during normal use"
Turn 2: "I never dropped it in water, it just cracked on its own"
Turn 3: uploads damage_photo.jpg
```

**claim_state_updater output:**
```
Claim A: physical_damage to screen, requires_visual_validation: True
Claim B: no_liquid_contact,          requires_visual_validation: True
```

**policy_checker output:**
```
Claim A: is_covered=True (§2, 12 months), visual_evidence_required=["crack visible"]
Claim B: is_covered=True, exclusion_clause=§3.2 if liquid found,
         visual_evidence_required=["absence of liquid signs"]
```

**evidence_planner output:**
```
Claim A checks:
  VisualCheck("Is a visible crack or fracture present on the display?",  expected=True)

Claim B checks:
  VisualCheck("Are there any signs of liquid contact on the device?",    expected=False)
  VisualCheck("Are device seals or internal components visibly corroded?", expected=False)
```

**vision_analysis output:**
```
Claim A — check 1: finding=True,  observation="Clear diagonal crack across display",
                   claim_supported=True, confidence=0.95
Claim B — check 1: finding=True,  observation="Visible moisture residue around port area",
                   claim_supported=False, confidence=0.88
Claim B — check 2: finding=True,  observation="Green oxidation visible on connector pins",
                   claim_supported=False, confidence=0.91
```

**claim_validator output:**
```
Claim A: verdict="valid"   (crack confirmed)
Claim B: verdict="invalid" (liquid signs found — contradicts user assertion)
         verdict_reason: "Visual evidence of liquid contact contradicts user's
                          statement. §3.2 Environmental Abuse exclusion applies."
```

**claim_decision:**
```
Claim A valid BUT Claim B invalid + §3.2 exclusion triggered
→ Session claim_status = REJECTED
→ decision_reason: "Water damage indicators detected visually despite user denial.
                    Excluded under §3.2 Environmental Abuse."
```

---

### 5.6 Future Scope — Vision Module

| Item | Notes |
|---|---|
| **Image quality edge cases** | Handle too_blurry / unrelated / unreadable with user-facing guidance — currently returns `inconclusive`, deferred to future sprint |
| **Multi-image per turn** | Groq vision supports 5 images/request. Allow multiple angles per turn; aggregate across checks |
| **OCR fallback for receipts** | If vision cannot extract `purchase_date`, fall back to `pytesseract` before asking the user to type it |
| **Image drift monitoring** | Track `image_quality`, `overall_confidence`, and `claim_supported` distributions over time — signals model degradation or client-side upload issues (see EVALS.md) |
| **Fine-tuned classifier** | Replace general vision LLM with a VoltEdge-specific fine-tuned model for high-volume production — dramatically reduces per-call cost |
| **Production vision model** | Replace `llama-4-scout` (Preview) with GPT-4o vision or Claude 3.5 Sonnet when cost budget allows |

---

---

## 7. Storage Layer

### 7.1 Overview

The prototype uses three storage systems, each with a clearly bounded responsibility.
No storage system is used for purposes outside its defined scope.

| Store | Technology | Responsibility |
|---|---|---|
| **Conversation & agent state** | LangGraph `MemorySaver` (in-process) | Full `AgentState` per session — messages, `ClaimContext`, `UserClaims`, verdicts, token counts |
| **Images** | Local filesystem (`IMAGE_UPLOAD_DIR`) | Raw image bytes, structured by naming convention |
| **Policy vectors** | Pinecone (serverless) | Policy chunk embeddings + metadata for RAG retrieval |

---

### 7.2 Conversation State — LangGraph MemorySaver

LangGraph's `MemorySaver` stores a checkpoint of the full `AgentState` after every node
execution. Each session maps to a LangGraph `thread_id` (equal to `session_id`). On the
next conversation turn, the graph resumes from the last checkpoint — full history,
`ClaimContext`, `UserClaims`, token count, and all intermediate state are available with
no re-fetching.

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()   # in-process, shared across all sessions in one run

graph = workflow.compile(checkpointer=checkpointer)

# Each turn invocation
config = {"configurable": {"thread_id": session_id}}
async for event in graph.astream_events(input, config=config, version="v2"):
    ...
```

**Limitation:** state is lost if the process restarts. Acceptable for a prototype / demo.
For production, replace `MemorySaver` with `AsyncPostgresSaver` — one line change,
zero business logic change (see §7.5 Future Scope).

---

### 7.3 Image Storage — Local Filesystem

Images are saved to the local directory specified by `IMAGE_UPLOAD_DIR` (default:
`./data/images`). The directory is created on startup if it does not exist.

#### Naming Convention

```
{session_id}_{type}_{timestamp}.{ext}
```

| Segment | Values | Example |
|---|---|---|
| `session_id` | UUID | `sess_abc123` |
| `type` | `damage` · `receipt` · `serial` | `damage` |
| `timestamp` | ISO-8601 compact, UTC | `20260331T143022` |
| `ext` | `jpg` · `png` · `webp` | `jpg` |

**Full example:** `sess_abc123_damage_20260331T143022.jpg`

Multiple images per session are distinguished by timestamp. If a user uploads two damage
photos in the same session the filenames are guaranteed unique.

#### Write / Read Flow

```python
# storage/image_store.py

import os
from datetime import datetime, timezone
from pathlib import Path
from app.config import settings

def save_image(session_id: str, image_type: str,
               data: bytes, ext: str) -> str:
    """
    Saves image to local filesystem.
    Returns the full file path stored in state.image_local_path.
    """
    upload_dir = Path(settings.IMAGE_UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"{session_id}_{image_type}_{ts}.{ext}"
    path = upload_dir / filename

    path.write_bytes(data)
    return str(path)


def read_image(image_local_path: str) -> bytes:
    """Read image bytes for vision analysis."""
    return Path(image_local_path).read_bytes()
```

`state.image_local_path` carries the full path and is passed directly to the
`vision_analysis` node. The vision node calls `read_image()` — no object store,
no pre-signed URLs, no network round-trip.

#### AgentState field update

`image_object_key` (previously referencing MinIO) is renamed to `image_local_path`
throughout `AgentState` and all graph nodes.

---

### 7.4 Policy Vectors — Pinecone

Pinecone is the only external storage service in the prototype. It holds the 384-dim
embeddings of all policy chunks plus their metadata. The indexing pipeline (§5.9) runs
once at startup if the namespace is empty. Runtime RAG queries hit Pinecone on every
relevant turn.

| Parameter | Value |
|---|---|
| Index name | `voltedge-policy` |
| Namespace | `warranty_policy_v1` |
| Dimensions | 384 (BAAI/bge-small-en-v1.5) |
| Record count | ~13 chunks |
| Tier | Serverless free |

Nothing in the application writes to Pinecone at runtime — it is read-only after indexing.

---

### 7.5 Future Scope — Storage

| Item | Notes |
|---|---|
| **LangGraph `AsyncPostgresSaver`** | Replace `MemorySaver` with Postgres-backed checkpointer for durable session state. One-line change in graph compilation. Enables session resume across restarts and horizontal scaling. |
| **PostgreSQL — application tables** | Add `sessions`, `claim_items`, and `images` tables for queryable claim history, analytics, and audit. Decouples application state from the LangGraph checkpoint blob. |
| **Object store (S3 / MinIO)** | Replace local filesystem image storage with S3-compatible object store. Change `save_image()` and `read_image()` in `storage/image_store.py` — no graph node changes needed. |
| **Image encryption at rest** | Apply AES-256 encryption to stored images (S3 SSE or application-layer). Required before handling real PII in production. |
| **Automated image purge** | Scheduled job that deletes images older than `SESSION_IMAGE_RETENTION_DAYS` after session close. Required for GDPR/privacy compliance. |
| **Redis** | Add for LiteLLM semantic caching and rate limiting once those features are activated. |

---

## Non-Functional Requirements

The table below is the consolidated NFR register for the entire system. It will be
referenced in each architectural section. The **Scope** column indicates whether the
requirement is addressed in the prototype or deferred.

| ID | Category | Requirement | Target | Scope |
|----|----------|-------------|--------|-------|
| NFR-01 | Latency | Time-to-first-token (TTFT) for text-only turns (P95) | < 800ms | In Scope |
| NFR-02 | Latency | Time-to-first-token (TTFT) for vision + RAG turns (P95) | < 2s | In Scope |
| NFR-02b | Latency | Full response completion for vision + RAG turns (P95) | < 12s | In Scope |
| NFR-16 | Streaming | All agent text responses delivered via SSE; no blocking full-response waits | Required | In Scope |
| NFR-03 | Scalability | Concurrent active sessions | 10,000+ | Out of Scope (prototype is single-instance) |
| NFR-04 | Availability | API uptime SLA | 99.9% | Out of Scope |
| NFR-05 | Security | All endpoints require user authentication | JWT / API key | Out of Scope (future) |
| NFR-06 | Security | Per-user rate limiting | 20 req/min | Out of Scope (future — must precede public launch) |
| NFR-07 | Reliability | Idempotent message sends via `request_id` dedup | No duplicate agent turns on retry | In Scope |
| NFR-08 | File Handling | Max image upload size | 10 MB | In Scope |
| NFR-09 | File Handling | Accepted image formats | JPEG, PNG, WebP | In Scope |
| NFR-10 | Privacy / Compliance | Receipt and damage images contain PII — stored locally for prototype; encryption and automated purge are production requirements | Local filesystem for prototype; AES-256 + automated purge in production | Out of Scope (prototype uses local storage; full compliance pipeline is future scope — see §7.5) |
| NFR-11 | Cost | LLM token budget cap per session (prevents runaway cost) | Configurable max tokens | In Scope |
| NFR-12 | Observability | Every request carries a `trace_id` propagated end-to-end through all services | Required | In Scope |
| NFR-13 | Observability | Structured JSON logs for all agent decisions | Required | In Scope |
| NFR-14 | Resilience | Graceful degradation when RAG returns no context | Agent must respond with fallback, not hallucinate | In Scope |
| NFR-15 | Resilience | Graceful degradation when image is unreadable/blurry | Structured error in vision JSON output | In Scope |
