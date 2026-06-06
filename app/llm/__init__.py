from app.llm.base import LLMProvider, LLMResponse, Message, ToolCall, ToolDef, ToolResult
from app.llm.mock import MockProvider


def get_provider(name: str) -> LLMProvider:
    if name == "mock":
        return MockProvider()
    raise ValueError(f"Unknown LLM provider: {name!r} (supported: mock)")


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "Message",
    "MockProvider",
    "ToolCall",
    "ToolDef",
    "ToolResult",
    "get_provider",
]
