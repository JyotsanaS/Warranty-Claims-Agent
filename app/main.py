"""
VoltEdge Warranty Agent — FastAPI application entry point.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.config import settings
from app.routers import sessions
from observability.tracing import setup_tracing

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"... <truncated {len(value) - limit} chars>"


def _format_request_body(body: bytes, content_type: str | None) -> str:
    if not body:
        return "<empty>"

    limit = settings.log_http_body_max_chars
    content_type = content_type or ""
    try:
        decoded = body.decode("utf-8")
    except UnicodeDecodeError:
        decoded = body.decode("utf-8", errors="replace")

    if "multipart/form-data" in content_type:
        return _truncate(decoded, limit)
    if "application/json" in content_type or "text/" in content_type:
        return _truncate(decoded, limit)
    return _truncate(decoded, limit)


async def _safe_read_body(request: Request) -> bytes:
    cached = getattr(request, "_cached_body", None)
    if isinstance(cached, (bytes, bytearray)):
        return bytes(cached)

    cached = request.scope.get("_cached_body")
    if isinstance(cached, (bytes, bytearray)):
        return bytes(cached)

    try:
        body = await request.body()
    except RuntimeError:
        return b"<stream consumed>"

    setattr(request, "_cached_body", body)
    request.scope["_cached_body"] = body
    return body


async def _replay_request(request: Request) -> Request:
    body = await _safe_read_body(request)

    async def receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}

    replay = Request(request.scope, receive)
    setattr(replay, "_cached_body", body)
    replay.scope["_cached_body"] = body
    return replay


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_tracing()
    logger.info("VoltEdge API starting up (env=%s)", settings.app_env)
    eviction_task = asyncio.create_task(sessions.idle_eviction_loop())
    yield
    eviction_task.cancel()
    logger.info("VoltEdge API shutting down")


app = FastAPI(
    title="VoltEdge Warranty Agent API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_http_request_response(request: Request, call_next: Callable):
    if not settings.log_http_bodies:
        return await call_next(request)

    request = await _replay_request(request)
    body = await _safe_read_body(request)
    content_type = request.headers.get("content-type")
    logger.info(
        "HTTP request %s %s content-type=%s body=%s",
        request.method,
        request.url.path,
        content_type or "-",
        _format_request_body(body, content_type),
    )

    response = await call_next(request)

    if isinstance(response, StreamingResponse) or response.media_type == "text/event-stream":
        logger.info(
            "HTTP response %s %s status=%s content-type=%s body=<streaming omitted>",
            request.method,
            request.url.path,
            response.status_code,
            response.media_type or response.headers.get("content-type", "-"),
        )
        return response

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    response_text = _format_request_body(
        response_body,
        response.media_type or response.headers.get("content-type"),
    )
    logger.info(
        "HTTP response %s %s status=%s content-type=%s body=%s",
        request.method,
        request.url.path,
        response.status_code,
        response.media_type or response.headers.get("content-type", "-"),
        response_text,
    )

    headers = dict(response.headers)
    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=headers,
        media_type=response.media_type,
        background=response.background,
    )

app.include_router(sessions.router, prefix="/api/v1")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await _safe_read_body(request)
    content_type = request.headers.get("content-type")
    logger.error(
        "Validation error on %s %s errors=%s body=%s",
        request.method,
        request.url.path,
        exc.errors(),
        _format_request_body(body, content_type),
    )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
