"""Provider-agnostic LLM types.

The agent loop speaks only these types; each provider adapter translates
to/from its vendor SDK. This keeps the LLM vendor swappable (and lets the
whole app run on the deterministic mock provider with no API key).
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


class LLMProviderError(Exception):
    """A provider call failed (bad credentials, rate limit, network, ...).

    Adapters raise this with a user-presentable message so the web layer can
    surface it without knowing which vendor is behind the protocol.
    """


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the tool's arguments


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    # Opaque provider payload round-tripped verbatim (e.g. Anthropic thinking
    # blocks, which must be resent during a tool-use turn). The agent loop
    # never inspects it; providers other than the one that set it ignore it.
    raw: Any = None


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # see Message.raw


@dataclass
class StreamEvent:
    """One incremental piece of a completion, pushed to ``on_event`` mid-call.

    ``kind`` is "thinking" (the model's reasoning) or "text" (answer prose);
    ``delta`` is the new fragment. The final, complete answer is still the
    returned ``LLMResponse`` — stream events are presentation-only.
    """

    kind: Literal["thinking", "text"]
    delta: str


StreamHandler = Callable[[StreamEvent], Awaitable[None]]


class LLMProvider(Protocol):
    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolDef],
        on_event: StreamHandler | None = None,
    ) -> LLMResponse: ...
