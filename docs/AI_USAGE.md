# AI Usage

## Tools
- **Claude Code** — architecture docs, scaffolding, iteration
- **Codex** — inline completion
- **Excalidraw** — initial HLD before any AI involvement

---

## Where AI helped

- Expanded `ARCHITECTURE.md` from my Excalidraw diagram — component descriptions, flow, scaling direction
- Scaffolded FastAPI boilerplate: sessions, SSE, image upload, Dockerfile, docker-compose
- Wired LangGraph graph once node list was settled
- Generated terminal nodes (greeting, escalation, status) after the node pattern was established
- Produced first drafts of system prompts, which then needed manual refinement through testing

---

## Where I overrode AI

**Planner layering.** AI suggested a single LLM call for all routing. I built three layers —
hard state-flag rules, a direct intent map, then LLM only for claim-reasoning — because
mechanical transitions shouldn't burn tokens and need to be unit-testable.

**Deterministic claim validator.** AI suggested LLM-based verdict assignment. Claim decisions
are auditable legal outputs — they must be reproducible. I made `claim_validator_node`
deterministic: confidence threshold, explicit rules, no LLM.

**Two-model claim state.** AI used one object for all claim data. I split into `ClaimContext`
(conversational working memory) and `UserClaim` (policy-assessable assertion) so the policy
engine receives clean, minimal input and each turn is independently auditable.

**Conversation memory management.** AI had no input on how to manage conversation history.
The key decision was not the window size but the architecture behind it: externalizing durable
claim data into structured state so nodes only need a short sliding window (last 10 turns) to
stay coherent. Without that, every node would need much longer windows to avoid losing context —
increasing cost and degrading focus. Window size, per-node truncation (planner caps content at
400 chars), and the decision to keep full history in state while letting each node take only
what it needs were all worked through independently.

**Custom chunking.** AI suggested a standard recursive splitter. Policy documents use heading
structure and tables to convey meaning — a generic splitter corrupts both. The section-aware,
table-preserving chunker I wrote brought context recall to 1.00.

**Post-resolution flow.** AI had no contribution to end-of-session state management —
`awaiting_post_resolution_followup`, `pending_switch_confirmation`, feedback collection.
These required understanding the full product flow, not just the claim-processing path.

**Retrieval design.** Chunking strategy, BGE-M3 selection, and top-k configuration were
worked through independently. We applied similarity thresholding after retrieval rather than
reranking — noisy low-relevance chunks passed upstream confuse generation more than they help,
and thresholding is a simpler, cheaper control with the same effect at this corpus size.
No meaningful AI input on this path.
