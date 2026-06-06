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


def _status_error(cls, status_code):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    return cls("boom", response=response, body=None)


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


# --- complete() ------------------------------------------------------------


async def test_complete_sends_model_thinking_and_mapped_payload():
    provider = AnthropicProvider(api_key="test-key")
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
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["tools"][0]["input_schema"] == {"type": "object"}
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    ("error", "fragment"),
    [
        (_status_error(anthropic.AuthenticationError, 401), "ANTHROPIC_API_KEY"),
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
    provider = AnthropicProvider(api_key="test-key")
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
