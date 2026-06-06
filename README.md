# Conversational scheduling assistant

A chatbot that lets a busy founder manage their [cal.com](https://cal.com) calendar through plain
conversation — book events, see what's coming up, cancel, and reschedule — with a web chat UI.

Built for the coding challenge described in [CHALLENGE.md](CHALLENGE.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv`).

```bash
cp .env.example .env       # add your cal.com API key + username
uv run uvicorn app.main:app
```

Open http://localhost:8000 and chat. Run the tests with:

```bash
uv run pytest
```

### Configuration (`.env`)

| Variable | Description |
|---|---|
| `CAL_API_KEY` | cal.com API key ([create one here](https://app.cal.com/settings/developer/api-keys)) |
| `CAL_USERNAME` | Your cal.com username (used to look up your event types) |
| `LLM_PROVIDER` | `mock` (default — runs with no LLM key; see below) |
| `TIMEZONE` | IANA timezone used when talking about times, e.g. `America/New_York` |

## Architecture

```
Browser chat UI (app/static/index.html)
        │  POST /api/chat {session_id, message}
        ▼
FastAPI (app/main.py)
        ▼
Agent loop (app/agent/loop.py) ──── system prompt w/ live "now" (app/agent/prompts.py)
        │   provider-agnostic tool-use loop, per-session history
        ├──► LLM provider (app/llm/) — swappable behind the LLMProvider protocol
        └──► Tool dispatch (app/agent/tools.py)
                     ▼
             CalComClient (app/calcom/client.py) ──► cal.com v2 REST API
```

- **Agent loop** — sends the conversation + tool schemas to the LLM; executes any tool calls it
  returns (concurrently) against cal.com; feeds results back; repeats until the LLM answers in
  prose. Iterations, history length, and session count are all capped.
- **Tools** — `list_bookings`, `list_event_types`, `get_available_slots`, `create_booking`,
  `cancel_booking`, `reschedule_booking`.
- **LLM abstraction** — the loop speaks only the neutral types in `app/llm/base.py`
  (`Message`, `ToolDef`, `ToolCall`, `LLMResponse`), so the vendor is swappable. The included
  `MockProvider` maps simple phrasings to tool calls deterministically, which keeps the whole
  stack runnable and testable with **no LLM API key**. A real provider (Anthropic/OpenAI) is a
  single adapter class behind the same protocol.
- **cal.com client** — thin async client over the v2 API. Note cal.com versions endpoints
  individually via the `cal-api-version` header; each method pins its documented version.

### Mock-mode phrasings

With `LLM_PROVIDER=mock`, the assistant understands simple deterministic phrasings:

- `What's on my calendar?`
- `What event types do I have?`
- `Show slots for event type 123`
- `Book event type 123 at 2026-06-12T10:00:00Z for ada@example.com`
- `Cancel booking <uid>`
- `Reschedule booking <uid> to 2026-06-12T10:00:00Z`

With a real LLM provider these become free-form ("book a 30-min intro with a candidate Thursday
afternoon"), with the provider doing the date resolution and slot negotiation the system prompt
describes.

## Known trade-offs

- Sessions are in-memory (one per browser tab) and evicted oldest-first past a cap — fine for a
  demo, not multi-process safe.
- Responses are plain JSON (no streaming); streaming becomes worthwhile once a real LLM provider
  is wired in.
