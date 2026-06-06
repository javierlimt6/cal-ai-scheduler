"""Provider-agnostic LLM types.

The agent loop speaks only these types; each provider adapter translates
to/from its vendor SDK. This keeps the LLM vendor swappable (and lets the
whole app run on the deterministic mock provider with no API key).
"""

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


class LLMProvider(Protocol):
    async def complete(
        self, system: str, messages: list[Message], tools: list[ToolDef]
    ) -> LLMResponse: ...
