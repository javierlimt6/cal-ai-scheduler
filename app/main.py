"""FastAPI app: serves the chat UI and the /api/chat endpoints."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.agent import Agent, AgentReply, PendingActionError, build_dispatch, build_system_prompt
from app.calcom import CalComClient, CalComError
from app.config import get_settings
from app.llm import LLMProviderError, get_provider
from app.ratelimit import RateLimiter

STATIC_DIR = Path(__file__).parent / "static"
LOG_DIR = Path(__file__).resolve().parent.parent / "tmp"  # gitignored

# Per-session ceiling on chat traffic; generous for a human, cheap insurance
# once a paid LLM key is wired in.
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60.0

logger = logging.getLogger(__name__)


class _FileLogHandler(RotatingFileHandler):
    """Marker subclass: repeated lifespans (tests) must not stack handlers."""


def _setup_file_logging() -> None:
    """Local observability: everything the app logs also lands in tmp/app.log
    (rotating, gitignored), so a server run can be inspected after the fact."""
    LOG_DIR.mkdir(exist_ok=True)
    root = logging.getLogger()
    if any(isinstance(handler, _FileLogHandler) for handler in root.handlers):
        return
    handler = _FileLogHandler(LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s — %(message)s"))
    root.addHandler(handler)
    if root.level in (logging.NOTSET, logging.WARNING):  # let INFO through (uvicorn leaves WARNING)
        root.setLevel(logging.INFO)


def _validate_timezone(name: str) -> str:
    """Return ``name`` if it's a usable IANA timezone, else "" (with a warning)."""
    if not name:
        return ""
    try:
        ZoneInfo(name)
    except (ValueError, KeyError):  # ZoneInfoNotFoundError is a KeyError
        logger.warning("Ignoring invalid timezone %r from the cal.com profile", name)
        return ""
    return name


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_file_logging()
    settings = get_settings()
    if not settings.cal_api_key:
        logger.warning(
            "CAL_API_KEY is not set — cal.com calls will fail. Copy .env.example to .env."
        )
    calcom = CalComClient(api_key=settings.cal_api_key, base_url=settings.cal_api_base_url)
    try:  # everything after client construction must release it, even on startup failure
        # Identity bootstrap: anything not configured explicitly comes from the
        # authenticated cal.com profile, so a bare CAL_API_KEY is enough to run.
        # Profile data is best-effort — a bad value degrades, never crashes boot
        # (an explicit TIMEZONE env var, by contrast, still fails fast below).
        username, timezone = settings.cal_username, settings.timezone
        if settings.cal_api_key and not (username and timezone):
            try:
                me = await calcom.get_me()
                username = username or me.get("username") or ""
                if not timezone:
                    timezone = _validate_timezone(str(me.get("timeZone") or ""))
                logger.info(
                    "Resolved from cal.com /me: username=%r, timezone=%r", username, timezone
                )
            except CalComError as exc:
                logger.warning("Could not resolve profile from cal.com /me: %s", exc)
        timezone = timezone or "UTC"

        build_system_prompt(timezone)  # fail fast on an invalid TIMEZONE
        app.state.agent = Agent(
            provider=get_provider(settings.llm_provider, settings),
            dispatch=build_dispatch(calcom, username),
            system_prompt=lambda: build_system_prompt(timezone),
            booking_lookup=calcom.get_booking,  # rich confirmation cards
            timezone=timezone,
        )
        app.state.rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)
        yield
    finally:
        await calcom.aclose()


# Local demo posture, deliberate: no auth and no CORS/TrustedHost middleware —
# the API is meant to be driven by its own UI on localhost (ARCHITECTURE.md §7).
app = FastAPI(title="cal.com scheduling assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:  # whitespace satisfies min_length but is an empty turn
            raise ValueError("message must not be blank")
        return value


class ConfirmRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    action_id: str = Field(min_length=1, max_length=64)
    approved: bool


class ToolActivityOut(BaseModel):
    name: str
    ok: bool


class PendingActionOut(BaseModel):
    # Deliberately just id + summary: raw tool arguments stay server-side.
    id: str
    summary: str


class ChatResponse(BaseModel):
    reply: str
    tool_activity: list[ToolActivityOut]
    pending_action: PendingActionOut | None = None


def _to_chat_response(reply: AgentReply) -> ChatResponse:
    return ChatResponse(
        reply=reply.text,
        tool_activity=[ToolActivityOut(name=a.name, ok=a.ok) for a in reply.tool_activity],
        pending_action=(
            PendingActionOut(id=reply.pending_action.id, summary=reply.pending_action.summary)
            if reply.pending_action
            else None
        ),
    )


def _check_rate_limit(session_id: str) -> None:
    if not app.state.rate_limiter.allow(session_id):
        raise HTTPException(
            status_code=429, detail="You're sending messages too quickly — give it a moment."
        )


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# Strong refs to in-flight turn tasks (the event loop only keeps weak ones);
# a task outlives its SSE stream if the client disconnects mid-turn, so the
# session history is still completed consistently.
_turn_tasks: set[asyncio.Task] = set()


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Run one chat turn, streaming progress as Server-Sent Events.

    Event order: any number of `thinking` / `text` deltas and `tool`
    completions, then exactly one terminal `done` (the full ChatResponse
    payload) or `error` ({detail, status}).
    """
    _check_rate_limit(request.session_id)
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def emit(event: dict[str, Any]) -> None:
        await queue.put(_sse(event.pop("type"), event))

    async def run_turn() -> None:
        try:
            reply = await app.state.agent.chat(request.session_id, request.message, emit)
            await queue.put(_sse("done", _to_chat_response(reply).model_dump()))
        except LLMProviderError as exc:
            # Adapter messages are already user-presentable; pass them through.
            logger.error("LLM provider error in /api/chat: %s", exc)
            await queue.put(_sse("error", {"detail": str(exc), "status": 502}))
        except Exception:
            logger.exception("Unhandled error in /api/chat")
            await queue.put(
                _sse(
                    "error",
                    {"detail": "Something went wrong handling that message.", "status": 500},
                )
            )
        finally:
            await queue.put(None)  # end of stream

    task = asyncio.create_task(run_turn())
    _turn_tasks.add(task)
    task.add_done_callback(_turn_tasks.discard)

    async def event_source():
        while (frame := await queue.get()) is not None:
            yield frame

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/confirm")
async def confirm(request: ConfirmRequest) -> ChatResponse:
    """Resolve a held destructive action. Only this endpoint executes one, and
    only with the arguments frozen when it was proposed."""
    _check_rate_limit(request.session_id)
    try:
        reply = await app.state.agent.resolve_pending(
            request.session_id, request.action_id, request.approved
        )
    except PendingActionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled error in /api/chat/confirm")
        raise HTTPException(
            status_code=500, detail="Something went wrong resolving that action."
        ) from exc
    return _to_chat_response(reply)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
