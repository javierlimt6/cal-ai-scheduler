from app.config import Settings
from app.llm.base import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    Message,
    StreamEvent,
    StreamHandler,
    ToolCall,
    ToolDef,
    ToolResult,
)
from app.llm.mock import MockProvider


def get_provider(name: str, settings: Settings) -> LLMProvider:
    if name == "mock":
        return MockProvider()
    if name == "anthropic":
        # Lazy import keeps the vendor SDK out of the process unless selected
        from app.llm.anthropic import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.anthropic_api_key, model=settings.llm_model or None
        )
    raise ValueError(f"Unknown LLM provider: {name!r} (supported: mock, anthropic)")


__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "LLMResponse",
    "Message",
    "MockProvider",
    "StreamEvent",
    "StreamHandler",
    "ToolCall",
    "ToolDef",
    "ToolResult",
    "get_provider",
]
