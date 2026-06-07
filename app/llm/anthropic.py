"""Anthropic (Claude) provider adapter.

The only module allowed to import the ``anthropic`` SDK. It translates the
neutral types in :mod:`app.llm.base` to and from the Claude Messages API:

- neutral ``Message`` history -> Anthropic ``messages`` (tool results become
  ``tool_result`` blocks in a user turn; assistant turns replay their ``raw``
  content blocks when present so thinking blocks survive the tool loop)
- neutral ``ToolDef`` -> Anthropic tool definitions
- Anthropic content blocks -> ``LLMResponse`` (text + tool calls + raw)
"""

from typing import Any

import anthropic

from app.llm.base import (
    LLMProviderError,
    LLMResponse,
    Message,
    StreamEvent,
    StreamHandler,
    ToolCall,
    ToolDef,
)

DEFAULT_MODEL = "claude-opus-4-8"
# Generous ceiling: replies are chat-sized, but adaptive thinking tokens
# count against the same cap and truncation surfaces as a broken answer.
MAX_TOKENS = 16000
REQUEST_TIMEOUT_SECONDS = 60.0  # chat UI: fail fast rather than hang (SDK default is 10 min)


_UNRESOLVED = object()


class AnthropicProvider:
    def __init__(self, api_key: str, model: str | None = None):
        if not api_key:
            raise LLMProviderError(
                "ANTHROPIC_API_KEY is not set — it is required when LLM_PROVIDER=anthropic."
            )
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
        self._model = model or DEFAULT_MODEL
        self._thinking: Any = _UNRESOLVED  # resolved per model on first call

    async def _thinking_config(self) -> dict[str, str] | None:
        """Adaptive thinking where the model supports it, omitted where it
        doesn't (e.g. Haiku 4.5). Resolved once via the Models API and cached;
        display=summarized because we stream the reasoning to the UI."""
        if self._thinking is _UNRESOLVED:
            try:
                model = await self._client.models.retrieve(self._model)
                caps: dict[str, Any] = model.capabilities.to_dict() if model.capabilities else {}
                supported = bool(caps["thinking"]["types"]["adaptive"]["supported"])
            except Exception:  # capability lookup is best-effort; assume modern
                supported = True
            self._thinking = {"type": "adaptive", "display": "summarized"} if supported else None
        return self._thinking

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolDef],
        on_event: StreamHandler | None = None,
    ) -> LLMResponse:
        # Typed as Any at the SDK boundary: we build plain wire-shape dicts
        # (pinned by unit tests) rather than mirroring the SDK's TypedDict tree.
        api_tools: Any = [to_anthropic_tool(tool) for tool in tools]
        api_messages: Any = to_anthropic_messages(messages)
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=api_tools,
            messages=api_messages,
        )
        thinking = await self._thinking_config()
        if thinking is not None:
            kwargs["thinking"] = thinking
        try:
            if on_event is None:
                response = await self._client.messages.create(**kwargs)
            else:
                async with self._client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        if event.type != "content_block_delta":
                            continue
                        if event.delta.type == "thinking_delta" and event.delta.thinking:
                            await on_event(StreamEvent("thinking", event.delta.thinking))
                        elif event.delta.type == "text_delta" and event.delta.text:
                            await on_event(StreamEvent("text", event.delta.text))
                    response = await stream.get_final_message()
        except anthropic.APIStatusError as exc:
            raise LLMProviderError(_describe_status_error(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMProviderError(
                "Could not reach the Anthropic API — check your network and try again."
            ) from exc
        except anthropic.APIError as exc:
            # Catch-all for SDK errors outside the two families above
            # (e.g. response validation) so no vendor exception escapes.
            raise LLMProviderError(f"Anthropic API error: {exc}") from exc
        return from_anthropic_response(response)


def to_anthropic_tool(tool: ToolDef) -> dict[str, Any]:
    return {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "user":
            out.append({"role": "user", "content": message.content})
        elif message.role == "assistant":
            if message.raw:  # truthiness: an empty block list must fall through
                # Replay the provider's own content blocks verbatim: required
                # for thinking blocks mid tool-turn, and byte-identical for
                # everything else.
                out.append({"role": "assistant", "content": message.raw})
                continue
            blocks: list[dict[str, Any]] = []
            if message.content:  # the API rejects empty text blocks
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                for call in message.tool_calls
            )
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        else:  # "tool" — Anthropic expects results as a user turn of tool_result blocks
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result.tool_call_id,
                            "content": result.content,
                            **({"is_error": True} if result.is_error else {}),
                        }
                        for result in message.tool_results
                    ],
                }
            )
    return out


def from_anthropic_response(response: Any) -> LLMResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            arguments = block.input if isinstance(block.input, dict) else {}
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=arguments))

    text = "\n\n".join(text_parts).strip()
    stop_reason = getattr(response, "stop_reason", None)
    if not text and not tool_calls:
        if stop_reason == "refusal":
            text = "I can't help with that request."
        else:
            text = "I didn't manage to produce a response — could you rephrase that?"
    elif stop_reason == "max_tokens":
        text += '\n\n(I ran out of room there — say "continue" and I\'ll pick up where I left off.)'
    return LLMResponse(text=text, tool_calls=tool_calls, raw=list(response.content))


def _describe_status_error(exc: anthropic.APIStatusError) -> str:
    if "credit balance" in str(exc.message).lower():
        return (
            "Your Anthropic account is out of credits — top up at console.anthropic.com "
            "(Plans & Billing), or set LLM_PROVIDER=mock to keep demoing without a key."
        )
    if isinstance(exc, anthropic.AuthenticationError):
        return "Anthropic rejected the API key — check ANTHROPIC_API_KEY in your .env."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "The Anthropic API key doesn't have permission for this model."
    if isinstance(exc, anthropic.RateLimitError):
        return "The assistant is rate-limited right now — wait a moment and try again."
    if exc.status_code >= 500:
        return "The Anthropic API is temporarily unavailable — try again shortly."
    return f"Anthropic API error {exc.status_code}: {exc.message}"
