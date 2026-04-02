# VoltEdge Warranty & Claims Agent

Multimodal warranty-support prototype built for the ChargePoint take-home exercise. The system combines:

- a FastAPI backend for session management and streaming agent responses
- a Streamlit frontend for chat, image upload, and claim-status display
- a LangGraph-based agent for conversation state and orchestration
- Pinecone-backed retrieval for policy grounding
- vision analysis for uploaded damage evidence

## What It Does

The agent helps a user:

- describe a charger or hardware issue in natural language
- retrieve relevant warranty policy context
- upload supporting images
- receive a policy-grounded claim outcome such as `approved`, `rejected`, or `pending`

## Repo Structure

- [`app/`](/home/antpc/Desktop/chargepoint/app) FastAPI app and API routes
- [`ui/`](/home/antpc/Desktop/chargepoint/ui) Streamlit frontend
- [`agent/`](/home/antpc/Desktop/chargepoint/agent) LangGraph agent, nodes, prompts, guardrails
- [`rag/`](/home/antpc/Desktop/chargepoint/rag) retrieval and indexing logic
- [`storage/`](/home/antpc/Desktop/chargepoint/storage) local persistence for session artifacts and images
- [`data/`](/home/antpc/Desktop/chargepoint/data) sample policy and scenarios
- [`tests/`](/home/antpc/Desktop/chargepoint/tests) unit and integration tests
- [`docs/architecture.md`](/home/antpc/Desktop/chargepoint/docs/architecture.md) architecture document
- [`AI_USAGE.md`](/home/antpc/Desktop/chargepoint/AI_USAGE.md) AI collaboration notes

## Prerequisites

- Python 3.12
- `uv`
- Docker and Docker Compose if you want containerized setup
- API credentials for the providers you intend to use

## Environment Setup

This repo keeps the template as [`env.example`](/home/antpc/Desktop/chargepoint/env.example) so it is visible in GitHub.

Create your local env file:

```bash
cp env.example .env
```

At minimum, configure:

- `GROQ_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_NAMESPACE`

The app stores generated session artifacts under `RESULTS_DIR`, which defaults to `./results`.

## Local Development With uv

Install dependencies:

```bash
uv sync
```

Run the backend:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Run the frontend in a second terminal:

```bash
uv run streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
```

Open:

- frontend: `http://localhost:8501`
- backend health: `http://localhost:8000/health`

## Docker Setup

Build and start both services:

```bash
docker compose up --build
```

This starts:

- backend on `http://localhost:8000`
- frontend on `http://localhost:8501`

The frontend is configured to call the backend service inside Compose using `BACKEND_URL=http://backend:8000`.

## Running Tests

Run the test suite:

```bash
uv run pytest -q
```

Notes:

- most tests are unit tests
- integration tests are marked with `@pytest.mark.integration`
- Pinecone-backed integration tests require a valid `.env`

Run only non-integration tests:

```bash
uv run pytest -q -m "not integration"
```

## API Summary

Main endpoints:

- `POST /api/v1/sessions`
- `POST /api/v1/sessions/{session_id}/messages`
- `GET /health`

Agent responses are streamed over SSE from the message endpoint.

## Key Implementation Notes

- Dependency management uses [`pyproject.toml`](/home/antpc/Desktop/chargepoint/pyproject.toml) and [`uv.lock`](/home/antpc/Desktop/chargepoint/uv.lock), not `requirements.txt`
- Environment template is intentionally kept as [`env.example`](/home/antpc/Desktop/chargepoint/env.example)
- Uploaded images and saved conversation artifacts are written under `RESULTS_DIR`
- The architecture write-up lives in [`docs/architecture.md`](/home/antpc/Desktop/chargepoint/docs/architecture.md)

## Submission Artifacts

- Architecture: [`docs/architecture.md`](/home/antpc/Desktop/chargepoint/docs/architecture.md)
- AI usage notes: [`AI_USAGE.md`](/home/antpc/Desktop/chargepoint/AI_USAGE.md)
- Requirement brief used for the exercise: [`REQUIREMENTS.md`](/home/antpc/Desktop/chargepoint/REQUIREMENTS.md)
