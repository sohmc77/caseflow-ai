import os

from .base import CompletionRequest, CompletionResponse, LLMProvider, Message, ProviderError, ToolCall, ToolSpec
from .mock import MockProvider


def get_provider(name: str | None = None) -> LLMProvider:
    """Build the provider selected by ``name`` or the LLM_PROVIDER env var (default: mock)."""
    name = (name or os.environ.get("LLM_PROVIDER") or "mock").lower()
    if name == "mock":
        return MockProvider()
    if name == "openai":
        from .openai_provider import OpenAIProvider  # optional dependency, imported lazily

        return OpenAIProvider()
    raise ValueError(f"unknown LLM_PROVIDER '{name}' (expected 'mock' or 'openai')")


__all__ = [
    "CompletionRequest",
    "CompletionResponse",
    "LLMProvider",
    "Message",
    "MockProvider",
    "ProviderError",
    "ToolCall",
    "ToolSpec",
    "get_provider",
]
