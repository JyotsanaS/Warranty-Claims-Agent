# VoltEdge — High-Level Block Diagram (MVP)

```mermaid
flowchart TD
    User(["👤 User"])

    subgraph FE["Frontend — ui/app.py"]
        ST["Streamlit Chat UI\n(chat_input · file_uploader\nwrite_stream · claim badge)"]
    end

    subgraph BE["Backend — app/main.py"]
        API["FastAPI\nPOST /sessions\nPOST /sessions/{id}/messages\nGET /sessions/{id}\nDELETE /sessions/{id}"]
        SSE["SSE Translator\napp/sse.py\n(text_delta · tool_call\nclaim_decision · done · error)"]
        IMG_VAL["Image Validator\n(JPEG/PNG/WebP ≤10MB ≥200px)"]
    end

    subgraph STORE["Storage — storage/image_store.py"]
        FS["Local Filesystem\ndata/images/\n{session}_{type}_{ts}.ext"]
    end

    subgraph GW["LLM Gateway — gateway/"]
        LLM_GW["llm_gateway.py\nfast_llm · main_llm · vision_llm\nget_embeddings()"]
        COST["cost_tracker.py\nper-call JSON log\nper-session token totals"]
    end

    subgraph RAG["RAG Pipeline — rag/"]
        IDX["indexer.py\nparse sample-policy.md\nchunk → embed → upsert"]
        RTV["retriever.py\nquery_reformulator\nparallel Pinecone queries\ndedupe · threshold check"]
    end

    subgraph GRAPH["LangGraph Agent Graph — agent/graph.py  (MemorySaver)"]
        direction TB

        subgraph ENTRY["Entry Pipeline"]
            CTX["context_extractor\n(fast_llm)"]
            CLE["claim_extractor\n(fast_llm)"]
            ROU["router\n(fast_llm)"]
        end

        subgraph SHORT["Short-Circuit Routes"]
            CONF["confirmation_handler"]
            GREET["greeting_node\n(main_llm)"]
            FB["fallback_node\n(main_llm)"]
            ESC["escalation_node"]
            CAN["cancellation_node"]
            STAT["status_node"]
            EMP["empathy_node\n(main_llm)"]
        end

        subgraph CLAIM["Claim Resolution Pipeline"]
            POL["policy_checker\n(fast_llm + RAG)"]
            EP["evidence_planner\n(fast_llm)"]
            VIS["vision_analysis\n(vision_llm)"]
            VAL["claim_validator\n(deterministic)"]
            AR["agent_respond\n(main_llm · streaming)"]
            DG["decision_gate\n(deterministic)"]
            CD["claim_decision\n(main_llm)"]
        end
    end

    subgraph EXT["External Services"]
        GROQ["Groq API\nllama-3.1-8b-instant  fast\nllama-3.3-70b-versatile  main\nllama-4-scout  vision"]
        EMB["Local SentenceTransformer\nBAAI/bge-small-en-v1.5"]
        PC["Pinecone\nIndex: voltedge-policy\nNS: warranty_policy_v1"]
    end

    subgraph DATA["Policy Data"]
        POL_DOC["data/sample-policy.md\n§1 General Warranty\n§2 Component Coverage\n§3 Exclusions\n§4 Claim Requirements"]
    end

    %% User ↔ UI ↔ API
    User -- "text + optional image" --> ST
    ST -- "multipart POST / SSE consume" --> API
    API -- "SSE stream" --> ST
    ST -- "streamed reply + claim badge" --> User

    %% API internals
    API --> SSE
    API --> IMG_VAL
    IMG_VAL -- "save_image()" --> FS
    API -- "invoke graph\n(session_id, message, image_path)" --> GRAPH

    %% SSE ← graph events
    GRAPH -- "astream_events()" --> SSE

    %% Entry pipeline flow
    CTX --> CLE --> ROU

    %% Router short-circuits
    ROU -- "pending_switch" --> CONF
    ROU -- "greeting" --> GREET
    ROU -- "out_of_scope" --> FB
    ROU -- "escalation" --> ESC
    ROU -- "cancellation" --> CAN
    ROU -- "status_query" --> STAT
    ROU -- "frustration" --> EMP --> POL

    %% Main claim path
    ROU -- "policy/claim/issue\nevidence/clarification" --> POL
    POL -- "policy-valid + image" --> EP --> VIS --> VAL --> AR
    POL -- "no image / all excluded" --> AR
    AR --> DG
    DG -- "all resolved" --> CD
    DG -- "pending" --> END_NODE(["END"])
    CD --> END_NODE

    %% Gateway calls
    LLM_GW --> COST
    GRAPH -- "all LLM calls routed through" --> LLM_GW

    %% Gateway → external
    LLM_GW -- "chat completions" --> GROQ
    LLM_GW -- "embeddings (local/)" --> EMB

    %% RAG
    RTV -- "Pinecone query" --> PC
    EMB -- "embed queries" --> RTV
    IDX -- "embed chunks\nupsert at startup" --> PC
    POL_DOC -- "parse + chunk" --> IDX
    POL -- "retriever.retrieve()" --> RTV

    %% Vision reads image
    VIS -- "read_image()" --> FS

    %% Embedding for indexing
    EMB -- "embed chunks" --> IDX

    classDef ext fill:#f0e6ff,stroke:#9b59b6
    classDef store fill:#e8f8e8,stroke:#27ae60
    classDef gateway fill:#fff3e0,stroke:#f39c12
    classDef graph fill:#e3f2fd,stroke:#2196f3
    classDef fe fill:#fce4ec,stroke:#e91e63
    classDef be fill:#e0f7fa,stroke:#00bcd4

    class GROQ,PC,EMB ext
    class FS,POL_DOC store
    class LLM_GW,COST gateway
    class CTX,CLE,ROU,CONF,GREET,FB,ESC,CAN,STAT,EMP,POL,EP,VIS,VAL,AR,DG,CD graph
    class ST fe
    class API,SSE,IMG_VAL be
```

---

## Layer Summary

| Layer | Files | Responsibility |
|---|---|---|
| **Frontend** | `ui/app.py` | Streamlit chat UI, SSE consumption, claim badge |
| **Backend** | `app/main.py`, `app/routers/sessions.py`, `app/sse.py` | REST + SSE API, image validation & storage, graph invocation |
| **Agent Graph** | `agent/graph.py`, `agent/nodes/*.py` | Full multi-turn claim resolution logic via LangGraph |
| **LLM Gateway** | `gateway/llm_gateway.py`, `gateway/cost_tracker.py` | Single provider abstraction, token tracking |
| **RAG Pipeline** | `rag/indexer.py`, `rag/retriever.py` | Policy indexing at startup, query-time retrieval |
| **Storage** | `storage/image_store.py` | Local filesystem image save/read |
| **Config** | `app/config.py`, `models/state.py` | Pydantic settings, all typed state structures |
| **External** | Groq API, Pinecone, local SentenceTransformer | LLM inference, vector store, embeddings |
