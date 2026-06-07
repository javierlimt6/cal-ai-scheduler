"""Anthropic adapter: neutral-type translation and error mapping (no network)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import anthropic
import httpx
import pytest

from app.config import Settings
from app.llm import (
    LLMProviderError,
    Message,
    MockProvider,
    ToolCall,
    ToolDef,
    ToolResult,
    get_provider,
)
from app.llm.anthropic import (
    DEFAULT_MODEL,
    AnthropicProvider,
    from_anthropic_response,
    to_anthropic_messages,
    to_anthropic_tool,
)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _status_error(cls, status_code, message="boom"):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    return cls(message, response=response, body=None)


def _provider(adaptive_supported=True, capability_lookup_fails=False):
    """AnthropicProvider with the Models API capability lookup stubbed out."""
    provider = AnthropicProvider(api_key="test-key")
    if capability_lookup_fails:
        provider._client.models.retrieve = AsyncMock(side_effect=RuntimeError("offline"))
    else:
        caps = {"thinking": {"types": {"adaptive": {"supported": adaptive_supported}}}}
        provider._client.models.retrieve = AsyncMock(
            return_value=SimpleNamespace(capabilities=SimpleNamespace(to_dict=lambda: caps))
        )
    return provider


# --- request mapping -------------------------------------------------------


def test_tooldef_maps_to_input_schema():
    tool = ToolDef(name="t", description="d", parameters={"type": "object", "properties": {}})

    assert to_anthropic_tool(tool) == {
        "name": "t",
        "description": "d",
        "input_schema": {"type": "object", "properties": {}},
    }


def test_history_maps_roles_and_tool_results():
    messages = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="checking",
            tool_calls=[ToolCall(id="c1", name="list_bookings", arguments={"status": "upcoming"})],
        ),
        Message(
            role="tool",
            tool_results=[
                ToolResult(tool_call_id="c1", content="[]"),
                ToolResult(tool_call_id="c2", content="nope", is_error=True),
            ],
        ),
    ]

    out = to_anthropic_messages(messages)

    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "checking"},
            {
                "type": "tool_use",
                "id": "c1",
                "name": "list_bookings",
                "input": {"status": "upcoming"},
            },
        ],
    }
    # Tool results ride in a user turn; is_error appears only when set
    assert out[2]["role"] == "user"
    assert out[2]["content"][0] == {"type": "tool_result", "tool_use_id": "c1", "content": "[]"}
    assert out[2]["content"][1]["is_error"] is True


def test_assistant_with_empty_text_emits_no_empty_text_block():
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="list_bookings", arguments={})],
        )
    ]

    (assistant,) = to_anthropic_messages(messages)

    assert all(block["type"] == "tool_use" for block in assistant["content"])


def test_assistant_raw_blocks_are_replayed_verbatim():
    raw = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "x"}]
    messages = [Message(role="assistant", content="x", tool_calls=[], raw=raw)]

    (assistant,) = to_anthropic_messages(messages)

    assert assistant["content"] is raw


# --- response mapping ------------------------------------------------------


def test_response_collects_text_and_tool_calls():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="On it."),
            SimpleNamespace(
                type="tool_use", id="c9", name="cancel_booking", input={"booking_uid": "abc"}
            ),
        ],
        stop_reason="tool_use",
    )

    result = from_anthropic_response(response)

    assert result.text == "On it."
    assert result.tool_calls == [
        ToolCall(id="c9", name="cancel_booking", arguments={"booking_uid": "abc"})
    ]
    assert result.raw == list(response.content)


def test_empty_response_gets_fallback_text():
    response = SimpleNamespace(content=[], stop_reason="end_turn")

    assert "rephrase" in from_anthropic_response(response).text


def test_refusal_gets_refusal_text():
    response = SimpleNamespace(content=[], stop_reason="refusal")

    assert "can't help" in from_anthropic_response(response).text


def test_truncated_response_tells_the_user():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="First half of a long answer")],
        stop_reason="max_tokens",
    )

    result = from_anthropic_response(response)

    assert result.text.startswith("First half")
    assert "ran out of room" in result.text


# --- complete() ------------------------------------------------------------


async def test_complete_sends_model_thinking_and_mapped_payload():
    provider = _provider()
    create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hello")], stop_reason="end_turn"
        )
    )
    provider._client.messages.create = create

    result = await provider.complete(
        "SYSTEM",
        [Message(role="user", content="hi")],
        [ToolDef(name="t", description="d", parameters={"type": "object"})],
    )

    assert result.text == "hello"
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["system"] == "SYSTEM"
    assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kwargs["tools"][0]["input_schema"] == {"type": "object"}
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


async def test_thinking_omitted_for_models_without_adaptive_support():
    provider = _provider(adaptive_supported=False)  # e.g. claude-haiku-4-5
    create = AsyncMock(
        return_value=SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])
    )
    provider._client.messages.create = create

    await provider.complete("s", [Message(role="user", content="hi")], [])
    await provider.complete("s", [Message(role="user", content="again")], [])

    assert "thinking" not in create.call_args.kwargs
    assert provider._client.models.retrieve.await_count == 1  # capability cached


async def test_capability_lookup_failure_assumes_adaptive():
    provider = _provider(capability_lookup_fails=True)
    create = AsyncMock(
        return_value=SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])
    )
    provider._client.messages.create = create

    await provider.complete("s", [Message(role="user", content="hi")], [])

    assert create.call_args.kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}


class _FakeStreamManager:
    def __init__(self, events, final):
        self._events = events
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def generate():
            for event in self._events:
                yield event

        return generate()

    async def get_final_message(self):
        return self._final


async def test_streaming_emits_deltas_and_returns_final_message():
    provider = _provider()
    deltas = [
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="thinking_delta", thinking="pondering "),
        ),
        SimpleNamespace(type="message_delta"),  # non-content events are skipped
        SimpleNamespace(
            type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="Hi!")
        ),
    ]
    final = SimpleNamespace(content=[SimpleNamespace(type="text", text="Hi!")])
    captured: dict = {}

    def fake_stream(**kwargs):
        captured.update(kwargs)
        return _FakeStreamManager(deltas, final)

    provider._client.messages.stream = fake_stream
    seen: list[tuple[str, str]] = []

    async def on_event(event):
        seen.append((event.kind, event.delta))

    result = await provider.complete("SYS", [Message(role="user", content="hi")], [], on_event)

    assert seen == [("thinking", "pondering "), ("text", "Hi!")]
    assert result.text == "Hi!"
    assert captured["thinking"] == {"type": "adaptive", "display": "summarized"}


@pytest.mark.parametrize(
    ("error", "fragment"),
    [
        (_status_error(anthropic.AuthenticationError, 401), "ANTHROPIC_API_KEY"),
        (
            _status_error(
                anthropic.BadRequestError, 400, message="Your credit balance is too low ..."
            ),
            "out of credits",
        ),
        (_status_error(anthropic.RateLimitError, 429), "rate-limited"),
        (_status_error(anthropic.InternalServerError, 500), "temporarily unavailable"),
        (
            anthropic.APIConnectionError(
                request=httpx.Request("POST", "https://api.anthropic.com")
            ),
            "Could not reach",
        ),
    ],
)
async def test_errors_become_friendly_provider_errors(error, fragment):
    provider = _provider()
    provider._client.messages.create = AsyncMock(side_effect=error)

    with pytest.raises(LLMProviderError, match=fragment):
        await provider.complete("s", [Message(role="user", content="hi")], [])


# --- factory ---------------------------------------------------------------


def test_get_provider_mock():
    assert isinstance(get_provider("mock", _settings()), MockProvider)


def test_get_provider_anthropic_requires_key():
    with pytest.raises(LLMProviderError, match="ANTHROPIC_API_KEY"):
        get_provider("anthropic", _settings(anthropic_api_key=""))


def test_get_provider_anthropic_uses_default_and_custom_model():
    default = get_provider("anthropic", _settings(anthropic_api_key="k"))
    custom = get_provider("anthropic", _settings(anthropic_api_key="k", llm_model="claude-x"))

    assert isinstance(default, AnthropicProvider)
    assert default._model == DEFAULT_MODEL
    assert custom._model == "claude-x"


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_provider("nope", _settings())
