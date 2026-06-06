"""FastAPI app: serves the chat UI and the /api/chat endpoint."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent import Agent, build_dispatch, build_system_prompt
from app.calcom import CalComClient, CalComError
from app.config import get_settings
from app.llm import LLMProviderError, get_provider

STATIC_DIR = Path(__file__).parent / "static"


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.cal_api_key:
        logger.warning(
            "CAL_API_KEY is not set — cal.com calls will fail. Copy .env.example to .env."
        )
    calcom = CalComClient(api_key=settings.cal_api_key, base_url=settings.cal_api_base_url)

    # Identity bootstrap: anything not configured explicitly comes from the
    # authenticated cal.com profile, so a bare CAL_API_KEY is enough to run.
    username, timezone = settings.cal_username, settings.timezone
    if settings.cal_api_key and not (username and timezone):
        try:
            me = await calcom.get_me()
            username = username or me.get("username") or ""
            timezone = timezone or me.get("timeZone") or ""
            logger.info("Resolved from cal.com /me: username=%r, timezone=%r", username, timezone)
        except CalComError as exc:
            logger.warning("Could not resolve profile from cal.com /me: %s", exc)
    timezone = timezone or "UTC"

    build_system_prompt(timezone)  # fail fast on an invalid TIMEZONE
    app.state.agent = Agent(
        provider=get_provider(settings.llm_provider, settings),
        dispatch=build_dispatch(calcom, username),
        system_prompt=lambda: build_system_prompt(timezone),
    )
    try:
        yield
    finally:
        await calcom.aclose()


app = FastAPI(title="cal.com scheduling assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)


class ToolActivityOut(BaseModel):
    name: str
    ok: bool


class ChatResponse(BaseModel):
    reply: str
    tool_activity: list[ToolActivityOut]


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        reply = await app.state.agent.chat(request.session_id, request.message)
    except LLMProviderError as exc:
        # Adapter messages are already user-presentable; pass them through.
        logger.error("LLM provider error in /api/chat: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # last-resort guard so the UI always gets JSON
        # HTTPException is handled (not propagated), so log the root cause here
        # or it vanishes entirely.
        logger.exception("Unhandled error in /api/chat")
        raise HTTPException(
            status_code=500, detail="Something went wrong handling that message."
        ) from exc
    return ChatResponse(
        reply=reply.text,
        tool_activity=[ToolActivityOut(name=a.name, ok=a.ok) for a in reply.tool_activity],
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
