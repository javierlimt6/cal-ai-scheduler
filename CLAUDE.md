# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A conversational scheduling assistant for cal.com, built for the coding challenge in
`CHALLENGE.md`: a Python chatbot through which a user books, views, cancels, and reschedules
cal.com events in plain language, served behind a FastAPI web chat UI.

## Commands

Uses [uv](https://docs.astral.sh/uv/) for everything:

```bash
uv sync                                  # install/refresh dependencies
uv run pytest                            # run all tests
uv run pytest tests/test_agent_loop.py   # run one test file
uv run pytest -k test_book_event         # run one test
uv run uvicorn app.main:app --reload     # dev server at http://localhost:8000
```

Runtime config comes from `.env` (see `.env.example`): `CAL_API_KEY`, `CAL_USERNAME`,
`LLM_PROVIDER` (default `mock`), `TIMEZONE`. Tests need no env vars or network — cal.com is
mocked with respx and the LLM with `MockProvider`.

## Architecture

Request flow: browser UI (`app/static/index.html`) → `POST /api/chat` (`app/main.py`, wiring in
lifespan) → `Agent.chat()` (`app/agent/loop.py`) → LLM provider + tool dispatch → `CalComClient`
(`app/calcom/client.py`) → cal.com v2 API.

Key invariants to preserve:

- **The agent loop speaks only the neutral LLM types** in `app/llm/base.py` (`Message`,
  `ToolDef`, `ToolCall`, `ToolResult`, `LLMResponse`). Never import a vendor SDK outside an
  `app/llm/<provider>.py` adapter; register new providers in `get_provider()`
  (`app/llm/__init__.py`).
- **The system prompt is rebuilt every turn** (it embeds the current datetime so relative dates
  resolve correctly in a long-lived server). `Agent` takes a zero-arg callable, not a string.
- **cal.com pins API versions per endpoint** via the `cal-api-version` header — the constants at
  the top of `app/calcom/client.py` are per-endpoint and intentionally not shared. `_request`
  strips `None` values from params and JSON bodies in one place; tool/client methods just pass
  optionals through.
- **Tool schemas and dispatch live together** in `app/agent/tools.py`; adding a tool means one
  `ToolDef` + one dispatch entry (tests build dispatch via `build_dispatch` against a fake
  client — don't hand-copy the mapping).
- Growth is bounded everywhere: tool iterations (`MAX_TOOL_ITERATIONS`), per-session history
  (`MAX_HISTORY_MESSAGES`), session count (`MAX_SESSIONS`).

The `MockProvider` (`app/llm/mock.py`) is deterministic keyword→tool-call scripting so the whole
stack runs and is tested without any LLM key; its phrasing rules are listed in the README.
