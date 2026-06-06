# End-to-end architecture

This document walks the entire system — every layer, file, contract, and design decision — from a
keystroke in the browser to a booking on cal.com and back.

For a quick start, see the [README](../README.md). The original challenge brief is in
[CHALLENGE.md](../CHALLENGE.md).

## 1. System overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Browser — app/static/index.html                                          │
│   single-page chat: message list, input, typing indicator,               │
│   per-tab session id (crypto.randomUUID), tool-activity chips            │
└───────────────┬──────────────────────────────────────────────────────────┘
                │ POST /api/chat  {session_id, message}
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ HTTP layer — app/main.py (FastAPI)                                        │
│   /            chat UI          /api/health   liveness                    │
│   /api/chat    validated DTOs in/out; last-resort error guard             │
│   lifespan: builds CalComClient + Agent once, fails fast on bad config    │
└───────────────┬──────────────────────────────────────────────────────────┘
                │ Agent.chat(session_id, message)
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Agent — app/agent/                                                        │
│   loop.py     per-session history + lock, tool-use loop, caps             │
│   tools.py    6 tool JSON schemas + dispatch table                        │
│   prompts.py  system prompt, rebuilt every turn with live "now"           │
└───────┬───────────────────────────────────────────┬──────────────────────┘
        │ LLMProvider.complete(system, history,     │ execute_tool(name, args)
        │                      tools)               ▼
        ▼                                  ┌────────────────────────────────┐
┌──────────────────────────────┐           │ cal.com client — app/calcom/    │
│ LLM layer — app/llm/          │           │   client.py: async httpx client │
│   base.py  neutral types +    │           │   per-endpoint cal-api-version  │
│            LLMProvider proto  │           │   None-stripping, CalComError   │
│   mock.py  deterministic      │           └────────────────┬───────────────┘
│            keyword router     │                            │ HTTPS
│   __init__ get_provider()     │                            ▼
└──────────────────────────────┘                   cal.com v2 REST API
```

Two invariants shape everything:

1. **The agent loop never sees a vendor SDK.** It speaks only the neutral types in
   `app/llm/base.py`. Swapping LLM vendors means writing one adapter file.
2. **All cal.com knowledge lives in `app/calcom/client.py`.** Endpoint paths, API-version
   headers, payload shapes, and error translation never leak upward.

## 2. Repository layout

```
livex/
├── app/
│   ├── main.py                 # FastAPI app, DTOs, lifespan wiring
│   ├── config.py               # Settings (pydantic-settings, .env)
│   ├── agent/
│   │   ├── loop.py             # Agent: sessions, locks, tool-use loop, caps
│   │   ├── tools.py            # ToolDef schemas + dispatch + execute_tool
│   │   └── prompts.py          # system prompt template + build_system_prompt
│   ├── llm/
│   │   ├── base.py             # Message/ToolDef/ToolCall/ToolResult/LLMResponse + protocol
│   │   ├── mock.py             # MockProvider (keyword → tool call, result → prose)
│   │   └── __init__.py         # get_provider() registry
│   ├── calcom/
│   │   └── client.py           # CalComClient + CalComError
│   └── static/
│       └── index.html          # the whole frontend (no build step)
├── tests/
│   ├── test_calcom_client.py   # respx-mocked HTTP contract tests
│   ├── test_agent_loop.py      # loop behavior with MockProvider + FakeCalCom
│   └── test_api.py             # ASGI-level endpoint tests (manual lifespan)
├── docs/ARCHITECTURE.md        # this file
├── CHALLENGE.md                # original brief
├── pyproject.toml              # uv project; pytest config
└── .env.example                # CAL_API_KEY, CAL_USERNAME, LLM_PROVIDER, TIMEZONE
```

## 3. Request lifecycle — one message, end to end

What happens when the user types *"What's on my calendar?"*:

1. **Browser** (`index.html`): the submit handler rejects empty/in-flight sends, renders the user
   bubble, shows a typing indicator, and `fetch`es `POST /api/chat` with the tab's session id.
2. **Validation** (`main.py`): `ChatRequest` enforces `session_id` (1–128 chars) and `message`
   (1–4000 chars); FastAPI returns 422 on violation before any work happens.
3. **Session entry** (`loop.py`): `Agent.chat` acquires the session's `asyncio.Lock` — concurrent
   requests for the *same* session queue up; different sessions proceed in parallel. Inside the
   lock: LRU bookkeeping, possible eviction (`MAX_SESSIONS`), history trim
   (`MAX_HISTORY_MESSAGES`), then the user message is appended.
4. **System prompt** (`prompts.py`): rebuilt **for this turn** with the current datetime in the
   configured timezone, so "tomorrow"/"Thursday afternoon" always resolve against real "now" —
   never against process start time.
5. **Tool-use loop** (`loop.py`, ≤ `MAX_TOOL_ITERATIONS`):
   - `provider.complete(system, history, TOOLS)` → an `LLMResponse`.
   - **Text response** → append to history, trim, return.
   - **Tool calls** → append the assistant message (with calls), execute all calls
     **concurrently** (`asyncio.gather` over `_run_tool`), append one `role="tool"` message
     carrying every `ToolResult`, loop again so the provider can see results.
   - Tool failures never crash the turn: `CalComError` → its message with `is_error=True`
     (the LLM explains and recovers conversationally); unexpected exceptions → logged with
     traceback, generic error result.
   - Iteration cap exhausted → polite "couldn't finish" reply.
6. **Tool execution** (`tools.py` → `calcom/client.py`): `execute_tool` looks up the dispatch
   table and calls the client; results are JSON-serialized (`default=str`) for the provider.
7. **HTTP to cal.com** (`client.py`): one `_request` choke point adds the per-endpoint
   `cal-api-version` header, strips `None` params/body fields (httpx would serialize them as
   empty), unwraps the `{"status", "data"}` envelope, and translates failures:
   - HTTP error responses → `CalComError(status, message)` (tolerates both `{"error": {...}}`
     and `{"error": "string"}` bodies),
   - transport failures (DNS, refused, protocol) → `CalComError(503, "Could not reach cal.com…")`.
8. **Response out** (`main.py`): `AgentReply` is projected onto wire DTOs — `reply` plus
   `tool_activity` as `[{name, ok}]` (the `arguments` field is deliberately **not** exposed to
   the browser). A last-resort `except` guarantees the UI always receives JSON.
9. **Browser** renders tool-activity chips (✓/✗ per tool) and the assistant bubble.

## 4. Layer guide

### 4.1 Frontend — `app/static/index.html`

One file, zero build step: HTML + CSS + ~80 lines of vanilla JS. Session identity is a
`crypto.randomUUID()` per tab ("session = browser tab"). All assistant/user text is inserted via
`textContent` (no HTML injection); the only `innerHTML` use renders server-controlled tool names.
An `inFlight` flag guards against Enter-key double submits (a disabled button does not block form
submission). Failure modes are first-class: non-OK responses render the server's `detail`,
network failures render a "server unreachable" bubble.

### 4.2 HTTP layer — `app/main.py`

- **Lifespan owns the object graph**: settings → `CalComClient` → `Agent` (provider, dispatch,
  system-prompt thunk), with `aclose()` teardown. Nothing is constructed at import time, which is
  what lets tests drive the same wiring with mocked transport.
- **Fail fast**: an invalid `TIMEZONE` raises at startup (the prompt is built once to validate);
  a missing `CAL_API_KEY` logs a prominent warning.
- **DTO boundary**: `ChatRequest` / `ChatResponse` / `ToolActivityOut` are the wire contract —
  intentionally separate from the agent's dataclasses so internal fields (e.g. raw tool
  arguments) don't leak, and so the OpenAPI schema is accurate.

### 4.3 Agent — `app/agent/`

**`loop.py`** is the orchestrator. State per session is just `list[Message]`. Bounds everywhere:

| Cap | Constant | Why |
|---|---|---|
| 8 tool iterations/turn | `MAX_TOOL_ITERATIONS` | runaway-loop protection (provider that always calls tools) |
| 60 messages/session | `MAX_HISTORY_MESSAGES` | bounds memory and LLM token cost; trimmed at turn start *and* end, window re-aligned to start on a `user` message |
| 500 sessions | `MAX_SESSIONS` | bounds total memory; LRU eviction (recency refreshed each turn), locks evicted with their session |

Per-session `asyncio.Lock` serializes turns within a session (no interleaved histories); separate
sessions are fully concurrent, as are tool calls within one turn.

**`tools.py`** declares the six tools and their dispatch:

| Tool | Maps to | Notes |
|---|---|---|
| `list_bookings` | `client.list_bookings` | status/date filters; also how the LLM finds uids |
| `list_event_types` | `client.list_event_types(username)` | username bound from config; tolerates stray kwargs |
| `get_available_slots` | `client.get_slots` | requires event_type_id + window; optional `duration` matches `length_in_minutes` |
| `create_booking` | `client.create_booking` | schema description instructs: verify slots first |
| `cancel_booking` | `client.cancel_booking` | description instructs: confirm before destructive action |
| `reschedule_booking` | `client.reschedule_booking` | new start must be checked for availability |

Tool descriptions are prescriptive about *when* to call ("Call this first when booking…") — that
wording is the main steering mechanism for a real LLM. Adding a tool = one `ToolDef` + one
dispatch entry; tests build dispatch through `build_dispatch` so the mapping can't drift.

**`prompts.py`** holds the persona and behavioral rules (resolve relative dates; check slots
before booking; gather only missing details; disambiguate before cancel/reschedule; confirm
afterwards; explain failures plainly). `build_system_prompt(timezone)` is called per turn via the
thunk injected in `main.py`.

### 4.4 LLM layer — `app/llm/`

**`base.py`** defines the entire vocabulary the agent speaks:

```
ToolDef(name, description, parameters)        # JSON Schema in
ToolCall(id, name, arguments)                 # what the model wants run
ToolResult(tool_call_id, content, is_error)   # what happened
Message(role: user|assistant|tool, content, tool_calls, tool_results)
LLMResponse(text, tool_calls)                 # one completion
LLMProvider protocol: async complete(system, messages, tools) -> LLMResponse
```

These are plain dataclasses on purpose — no pydantic, no vendor types — so a provider adapter is
pure translation.

**`mock.py`** (`MockProvider`) makes the whole stack run deterministically with **no LLM key**:

- *Routing* (user message → tool call): keyword intents, checked in precedence order
  (slots → cancel → reschedule → book → event types → calendar → help). Booking matches on a
  word boundary (`\bbook\b`) so "show my bookings" lists rather than books. Before extracting
  numeric event-type ids or booking uids, ISO datetimes and emails are stripped from the text so
  `2026‑06‑12T10:00:00Z` can't become event type `2026` and `ada123@x.com` can't become a uid.
- *Summarizing* (tool result → prose): after the loop feeds results back, the mock formats them
  per tool (calendar listing with uids, slots by day, booked/cancelled/rescheduled
  confirmations); errors become "That didn't work: <api message>".

**`__init__.py`** exposes `get_provider(name)` — the registry where a real adapter
(`app/llm/anthropic.py`, `app/llm/openai.py`, …) gets one `if` branch.

### 4.5 cal.com client — `app/calcom/client.py`

A deliberately thin async client. The non-obvious parts, all verified against cal.com's docs:

- **Per-endpoint API versions.** cal.com v2 versions endpoints *individually* via the
  `cal-api-version` header; the constants at the top are per endpoint and intentionally not
  shared (`/bookings` list `2026-05-01`, booking writes `2026-02-25`, `/slots` `2024-09-04`,
  `/event-types` and `/me` `2024-06-14`).
- **Response shapes** (don't "fix" these to something more defensive — they're documented):
  slots → `{"YYYY-MM-DD": [{"start": ...}, ...]}`; event types → flat array with
  `id`/`title`/`lengthInMinutes`; everything arrives wrapped in `{"status", "data"}` which
  `_request` unwraps.
- **`None`-stripping in one place.** httpx serializes `None` params as `a=` rather than omitting
  them, so `_request` strips `None` from params *and* JSON bodies; method signatures stay flat.
- **Error translation.** Every failure becomes `CalComError(status_code, message)` — the single
  exception type the agent layer handles.

| Method | Endpoint |
|---|---|
| `list_bookings(status, after_start, before_end, limit)` | `GET /bookings` |
| `create_booking(event_type_id, start, attendee_*, time_zone, …)` | `POST /bookings` |
| `cancel_booking(uid, reason)` | `POST /bookings/{uid}/cancel` |
| `reschedule_booking(uid, new_start, reason)` | `POST /bookings/{uid}/reschedule` |
| `get_slots(start, end, event_type_id, time_zone, duration)` | `GET /slots` |
| `list_event_types(username)` | `GET /event-types` |
| `get_me()` | `GET /me` |

### 4.6 Configuration — `app/config.py` + `.env`

`Settings` (pydantic-settings) reads `.env`: `CAL_API_KEY`, `CAL_USERNAME`,
`LLM_PROVIDER` (default `mock`), `TIMEZONE` (default `UTC`), `CAL_API_BASE_URL` (override for
self-hosted cal.com). `get_settings()` is `lru_cache`d — read once per process.

## 5. Testing strategy

29 tests, **zero network and zero secrets required** — the seam for each layer is mocked at the
layer below it:

| Suite | Mocks | Proves |
|---|---|---|
| `test_calcom_client.py` | HTTP via respx | auth + per-endpoint version headers, payload shapes, `None`-omission, envelope unwrap, error translation (dict/string/non-JSON bodies, network failures), header omitted when key is empty |
| `test_agent_loop.py` | `FakeCalCom` behind `build_dispatch` + real `MockProvider` | all four user journeys (view/book/cancel/reschedule), conversational error relay, session isolation, the regression cases (year≠event-id, email≠uid, "bookings"≠book), concurrency (no interleaving under `asyncio.gather`), trim alignment, LRU eviction, runaway-loop cap |
| `test_api.py` | respx + manual lifespan over ASGITransport | endpoint wiring end to end: health, UI serving, request validation, full chat round trip through the real Agent/Mock/CalComClient stack |

Run: `uv run pytest` (or `-k name` / a file path for a subset).

## 6. Extension points

- **Real LLM provider** (the designed-for next step): create `app/llm/<vendor>.py` translating
  `Message/ToolDef/ToolCall ↔ vendor SDK`, register it in `get_provider()`, set
  `LLM_PROVIDER=<vendor>` + the vendor key in `.env`. Nothing else changes.
- **New tool**: add a `ToolDef` + dispatch entry in `app/agent/tools.py` (and a client method if
  it's a new endpoint). The loop, providers, and UI pick it up automatically.
- **Streaming replies**: becomes worthwhile with a real provider — swap `/api/chat` to SSE and
  append deltas in the UI; the agent loop's seam (`complete()`) is where a `stream()` variant
  would slot in.
- **Persistence / multi-process**: `Agent._sessions` is a dict by design (demo scope). The seam
  is narrow — replace the dict with a store keyed the same way and keep the lock per worker.

## 7. Known trade-offs (deliberate, demo-scoped)

- **In-memory sessions, single process** — a restart clears conversations; multiple workers
  wouldn't share state. Acceptable per challenge scope; see extension point above.
- **Session = browser tab** — no auth/cookies; the session id is client-minted. Fine for a demo;
  a production deployment would mint server-side, signed.
- **Eviction edge**: a session evicted at the `MAX_SESSIONS` boundary while a turn is in flight
  finishes against its orphaned history (the turn completes; subsequent turns start fresh).
- **`ToolActivity.arguments`** is recorded server-side but intentionally not sent to the browser
  (avoid leaking attendee details into the DOM); it exists for logging/audit.
- **Mock provider's NLU is keyword-grade** by design — it exists to exercise the full stack
  deterministically, not to replace the LLM. The exact phrasings are in the README.
