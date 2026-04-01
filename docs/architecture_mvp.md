# Architecture MVP — VoltEdge Automated Warranty & Claims Agent

> Scope: Only blocks required to run a working end-to-end demo.
> Everything marked "Future Scope" in `architecture.md` is excluded.
> Production concerns (auth, rate limiting, Postgres, S3, caching) are deferred.

---

## Building Blocks

### 1. Config & Environment

**File:** `app/config.py`

Central Pydantic `Settings` class loaded from `.env`. All other modules import from here — no raw `os.getenv` calls outside this file.

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Auth for all Groq model calls |
| `PINECONE_API_KEY` | Auth for Pinecone vector store |
| `PINECONE_INDEX_NAME` | e.g. `voltedge-policy` |
| `PINECONE_NAMESPACE` | e.g. `warranty_policy_v1` |
| `LLM_FAST_MODEL` | e.g. `groq/llama-3.1-8b-instant` |
| `LLM_MAIN_MODEL` | e.g. `groq/llama-3.3-70b-versatile` |
| `LLM_VISION_MODEL` | e.g. `groq/meta-llama/llama-4-scout-17b-16e-instruct` |
| `LLM_EMBEDDING_MODEL` | e.g. `local/BAAI/bge-small-en-v1.5` |
| `IMAGE_UPLOAD_DIR` | Local path, e.g. `./data/images` |
| `MAX_TOKENS_PER_SESSION` | Token budget cap, e.g. `8000` |
| `RAG_SIMILARITY_THRESHOLD` | Min cosine score, e.g. `0.75` |

---

### 2. LLM Gateway

**Files:** `gateway/llm_gateway.py`, `gateway/cost_tracker.py`

Single module that owns all model interactions. No other module imports a provider SDK directly.

**Clients to instantiate:**
- `fast_llm` — `ChatLiteLLM`, streaming, temperature 0.0 (router, extractors, query reformulator)
- `main_llm` — `ChatLiteLLM`, streaming, temperature 0.3 (agent_respond, empathy, claim_decision)
- `vision_llm` — `ChatLiteLLM`, non-streaming, temperature 0.0 (vision_analysis only)
- `get_embeddings(texts)` — routes to local `SentenceTransformer` (prefix `local/`) or `litellm.embedding()` for API providers

**Cost tracker:** LiteLLM `success_callback` → `track_usage()` logs structured JSON per call and accumulates per-session token totals. Exposes `get_session_usage(session_id)`.

---

### 3. Data Models

**File:** `models/state.py`

All typed data structures passed through the graph.

#### `AgentState` (LangGraph `TypedDict`)
```
session_id, trace_id, turn_count
messages          — append-only (add_messages reducer)
empathy_prefix    — set by empathy_node
image_local_path  — full path to uploaded image this turn
policy_context    — list[str] chunks from RAG
damage_report     — dict from vision module
claim_items       — list[ClaimContext]
active_claim_index
context_switch_detected
pending_switch_confirmation
policy_clauses    — list[str] for decision citation
token_count
intent
router_confidence
user_claims       — list[UserClaim]
```

#### `ClaimContext` (Pydantic)
Structured working memory per claim item: component, incident_type, incident_date, purchase_date, evidence checklist booleans, claim_status, warranty eligibility fields, audit metadata.

#### `UserClaim` (Pydantic)
Extracted claim assertion per conversation turn: component, assertion type, verbatim statement, visual validation flag, `PolicyCoverage`, `list[VisualCheck]`, claim_verdict.

#### `VisualCheck` (Pydantic)
One check question: `check_description`, `expected_finding`, `finding`, `observation`, `claim_supported`, `confidence`.

#### `VisionReport` (Pydantic)
Output of `vision_analysis`: image_quality, image_type, completed_checks, overall_confidence, extracted purchase_date / serial_number.

#### `RouterOutput` (Pydantic)
`intent`, `confidence`, `requires_policy_lookup`, `requires_vision`.

---

### 4. Storage — Local Image Store

**File:** `storage/image_store.py`

Two functions only:
- `save_image(session_id, image_type, data, ext) -> str` — writes bytes to `IMAGE_UPLOAD_DIR`, returns full path
- `read_image(image_local_path) -> bytes` — reads bytes for vision node

Naming convention: `{session_id}_{type}_{timestamp}.{ext}`

No object store, no pre-signed URLs — local filesystem for prototype.

---

### 5. RAG Pipeline

**Files:** `rag/indexer.py`, `rag/retriever.py`

#### 5a. Indexing Pipeline (`rag/indexer.py`)
Runs once at startup if Pinecone namespace is empty, and manually on policy updates.

Steps: parse `sample-policy.md` → structure-aware chunker (table row / list item / prose paragraph) → context prefix injector (`[Section: …]` breadcrumb) → metadata extractor → `get_embeddings()` → Pinecone upsert (namespace `warranty_policy_v1`).

Chunk metadata fields: `chunk_id`, `document_id`, `section_number`, `section_title`, `chunk_type`, `component`, `coverage_months`, `is_exclusion`, `policy_version`, `text`, `token_count`.

#### 5b. Retrieval (`rag/retriever.py`)
Called by the `policy_checker` node at runtime.

Steps: `query_reformulator` (fast_llm, generates 2–3 queries from user message + ClaimContext) → parallel Pinecone queries (top_k=5, metadata filter by component + policy_version) → deduplicate by chunk_id → sort by score → threshold check → return `RAGResult(chunks, fallback)`.

If `fallback=True` (no chunk above `RAG_SIMILARITY_THRESHOLD`): agent must not hallucinate; escalate with `"no_policy_context"`.

---

### 6. LangGraph Agent Graph

**File:** `agent/graph.py`

Compiled with `MemorySaver` checkpointer. Each `session_id` = LangGraph `thread_id`. State is persisted across turns in-process (lost on restart — acceptable for prototype).

#### Graph Nodes

All nodes are in `agent/nodes/`.

| Node | Model | Role |
|---|---|---|
| `context_extractor` | fast_llm | Extracts structured updates from latest user message; merges into `ClaimContext`; detects context switch |
| `claim_extractor` | fast_llm | Extracts / updates `UserClaim` list from full conversation; additive, never removes existing claims |
| `router` | fast_llm | Classifies intent + sets `requires_policy_lookup` / `requires_vision`; short-circuits to `confirmation_handler` if `pending_switch_confirmation` is True |
| `confirmation_handler` | — | Resolves pending context switch; updates `claim_items` and `active_claim_index` |
| `greeting_node` | main_llm | Handles `greeting` intent; no RAG, no vision |
| `fallback_node` | main_llm | Handles `out_of_scope`; politely redirects |
| `status_node` | none | Reads `claim_items` from state; no LLM call |
| `empathy_node` | main_llm | Generates short empathy prefix for `frustration` intent; writes `state.empathy_prefix` |
| `escalation_node` | — | Sets `claim_status = "escalated"`; returns human handoff message |
| `cancellation_node` | — | Sets `claim_status = "escalated"` with reason `user_cancelled` |
| `policy_checker` | fast_llm + RAG | For each `UserClaim`: runs query reformulation + retrieval → fills `PolicyCoverage`; marks excluded claims immediately |
| `evidence_planner` | fast_llm | For each policy-valid claim: generates `VisualCheck` list (check_description + expected_finding) |
| `vision_analysis` | vision_llm | One model call with image + all checks; returns `VisionReport`; reads image via `read_image()` |
| `claim_validator` | none | Deterministic; maps check results → `claim_verdict` per claim |
| `agent_respond` | main_llm | Streams primary user-facing response; prepends `empathy_prefix` if set; asks for missing evidence |
| `decision_gate` | none | Checks `all_claims_resolved`; routes to `claim_decision` or END |
| `claim_decision` | main_llm | Emits final verdict with policy clause citations; sets session `claim_status` |

#### Graph Topology (simplified)

```
START
  │
  ▼
[context_extractor] → [claim_extractor] → [router]
  │
  ├── pending_switch_confirmation → [confirmation_handler] → END
  ├── greeting       → [greeting_node]    → END
  ├── out_of_scope   → [fallback_node]    → END
  ├── escalation     → [escalation_node]  → END
  ├── cancellation   → [cancellation_node]→ END
  ├── status_query   → [status_node]      → END
  ├── frustration    → [empathy_node] → [empathy_router]
  │                      → [rag_retrieval / agent_respond path]
  │
  └── policy/claim/issue/evidence/clarification
          ▼
      [policy_checker]
          │
          ├── has image + policy-valid claims
          │     → [evidence_planner] → [vision_analysis] → [claim_validator]
          │                                                        │
          └── no image / all excluded                              ▼
                    └──────────────────────────────────► [agent_respond]
                                                                   │
                                                           [decision_gate]
                                                             │         │
                                                       resolved      pending
                                                             │         │
                                                    [claim_decision] END
                                                             │
                                                            END
```

---

### 7. FastAPI Backend

**File:** `app/main.py`, `app/routers/sessions.py`

#### Endpoints

| Method | Endpoint | Notes |
|---|---|---|
| `POST` | `/api/v1/sessions` | Creates session; returns `session_id` |
| `POST` | `/api/v1/sessions/{session_id}/messages` | Accepts multipart (text + optional image); returns `text/event-stream` (SSE) |
| `GET` | `/api/v1/sessions/{session_id}` | Returns session state summary + claim_status |
| `DELETE` | `/api/v1/sessions/{session_id}` | Closes session |

#### SSE Event Translation (`app/sse.py`)

Consumes `graph.astream_events()` and yields typed SSE strings:

| LangGraph event | SSE event type |
|---|---|
| `on_chat_model_stream` | `text_delta` |
| `on_tool_start` (rag / vision) | `tool_call {status: "running"}` |
| `on_tool_end` | `tool_result` |
| `on_chain_end` at `claim_decision` | `claim_decision` |
| `on_chain_end` at graph END | `done` |
| any exception | `error` |

Image upload: validated (JPEG/PNG/WebP, ≤ 10 MB, ≥ 200×200 px) then saved via `save_image()`. Path written to `AgentState.image_local_path` before graph invocation.

Each request attaches a `trace_id` (UUID) propagated through all nodes via LangGraph state and LiteLLM call metadata.

---

### 8. Streamlit Frontend

**File:** `ui/app.py`

Single-page chat app. Communicates only with the FastAPI backend.

| Component | Widget |
|---|---|
| New session | `st.button` → `POST /api/v1/sessions` |
| Chat history | `st.chat_message` loop over `st.session_state.messages` |
| Image uploader | `st.file_uploader` (JPEG/PNG/WebP, 10 MB max) |
| Text input | `st.chat_input` |
| Streaming reply | `st.write_stream()` consuming SSE generator |
| Tool status | `st.status()` driven by `tool_call` events |
| Claim badge | `st.success / st.warning / st.error` on `claim_decision` event |

`st.session_state` holds: `session_id`, `messages`, `claim_status`, `uploaded_file`.

---

### 9. Policy Document

**File:** `data/sample-policy.md`

The warranty policy that is chunked and indexed into Pinecone at startup. Contains:
- §1 General Warranty (prose)
- §2 Component Coverage (table)
- §3 Standard Exclusions (list)
- §4 Claim Validation Requirements (list)

~13 chunks total after structure-aware splitting.

---

## What Is Explicitly Out of Scope for MVP

- Authentication / JWT
- Per-user rate limiting
- LangGraph `AsyncPostgresSaver` (use `MemorySaver`)
- S3 / MinIO object store (use local filesystem)
- LiteLLM Proxy server / semantic caching / multi-key rotation
- Automated image purge / encryption at rest
- Hybrid BM25+dense retrieval / cross-encoder re-ranking
- Multi-document policy namespaces
- Human-in-the-loop (`interrupt_before`)
- OCR fallback for receipts
- React/Next.js frontend
- Observability dashboards / eval harness
