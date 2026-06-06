# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A conversational scheduling assistant for cal.com, built for the coding challenge in
`CHALLENGE.md`: a Python chatbot through which a user books, views, cancels, and reschedules
cal.com events in plain language — Claude-powered, served behind a FastAPI web chat UI.

`docs/ARCHITECTURE.md` is the end-to-end reference (request lifecycle, layer contracts,
extension points) — keep it in sync when changing structure or contracts.

## Commands

Uses [uv](https://docs.astral.sh/uv/) for everything:

```bash
uv sync                                  # install/refresh dependencies
uv run pytest                            # run all tests
uv run pytest tests/test_agent_loop.py   # run one test file
uv run pytest -k test_book_event         # run one test
uv run ruff check . && uv run ruff format --check .   # lint + format (CI-enforced)
uv run mypy                              # type-check app/ (CI-enforced)
uv run uvicorn app.main:app --reload     # dev server at http://localhost:8000
```

Runtime config comes from `.env` (see `.env.example`): `CAL_API_KEY`, `ANTHROPIC_API_KEY`,
`LLM_PROVIDER` (`anthropic` or `mock`; the Settings default is `mock`), `LLM_MODEL` (optional),
`CAL_USERNAME`/`TIMEZONE` (optional — resolved from cal.com `GET /me` at startup). Tests need no
env vars or network — cal.com is mocked with respx, the LLM with `MockProvider`, and
`tests/conftest.py` pins the env so a developer's local `.env` can't leak into the suite.

## Architecture

Request flow: browser UI (`app/static/index.html`) → `POST /api/chat` (SSE stream: `thinking` /
`text` / `tool` events, terminal `done`/`error`) or `/api/chat/confirm` (plain JSON)
(`app/main.py`, wiring in lifespan, per-session rate limit) → `Agent.chat()` /
`Agent.resolve_pending()` (`app/agent/loop.py`) → LLM provider + tool dispatch → `CalComClient`
(`app/calcom/client.py`) → cal.com v2 API.

Key invariants to preserve:

- **The agent loop speaks only the neutral LLM types** in `app/llm/base.py` (`Message`,
  `ToolDef`, `ToolCall`, `ToolResult`, `LLMResponse`). Never import a vendor SDK outside an
  `app/llm/<provider>.py` adapter (`app/llm/anthropic.py` is the worked example); register
  providers in `get_provider()` (`app/llm/__init__.py`). The `raw` field on
  `Message`/`LLMResponse` is an opaque provider payload (e.g. Claude thinking blocks, which must
  be replayed mid tool-turn) — the loop copies it and never inspects it.
- **Destructive actions execute only on an explicit user click.** The loop freezes
  `cancel_booking`/`reschedule_booking` calls as a `PendingAction`; only
  `Agent.resolve_pending` (via `POST /api/chat/confirm`) executes one, with the arguments frozen
  at proposal time. A new mutating tool belongs in `DESTRUCTIVE_TOOLS`. The pending slot is
  reserved synchronously (no await between check and store) before the card-enriching
  `booking_lookup` fetch — keep it that way or concurrent calls race.
- **Never hardcode a thinking config** in the Anthropic adapter — support is resolved per model
  via the Models API (`_thinking_config`, cached); Haiku 4.5 has no adaptive thinking and 400s
  if sent one.
- **The system prompt is rebuilt every turn** (it embeds the current datetime so relative dates
  resolve correctly in a long-lived server). `Agent` takes a zero-arg callable, not a string.
- **cal.com pins API versions per endpoint** via the `cal-api-version` header — the constants at
  the top of `app/calcom/client.py` are per-endpoint and intentionally not shared. `_request`
  strips `None` values from params and JSON bodies in one place; tool/client methods just pass
  optionals through. `list_bookings` follows `pagination.nextCursor` (bounded by `max_pages`).
- **Tool schemas and dispatch live together** in `app/agent/tools.py`; adding a tool means one
  `ToolDef` + one dispatch entry against the `SchedulingClient` protocol (tests build dispatch
  via `build_dispatch` against a fake client — don't hand-copy the mapping).
- Growth is bounded everywhere: tool iterations (`MAX_TOOL_ITERATIONS`), per-session history
  (`MAX_HISTORY_MESSAGES`), session count (`MAX_SESSIONS`), chat rate
  (`app/ratelimit.py`), bookings pages (`max_pages`).
- **Model/user text never reaches `innerHTML`** in `index.html` — the mini-markdown renderer
  builds DOM nodes only.

The `MockProvider` (`app/llm/mock.py`) is deterministic keyword→tool-call scripting so the whole
stack runs and is tested without any LLM key; its phrasing rules are listed in the README. It
also understands the gate's `CONFIRMATION_REQUIRED` tool result.
