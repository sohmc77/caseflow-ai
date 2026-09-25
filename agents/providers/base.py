"""Vendor-neutral LLM provider interface.

Agents speak only these types. A provider adapts them to one vendor's API, so
switching vendors (or running with no vendor at all) never touches agent code.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel


class ToolCall(BaseModel):
    id: str
    name: str
    arguments_json: str  # raw; the agent validates it against the tool's args model


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None  # role == "tool" only
    name: str | None = None  # tool name, role == "tool" only


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class CompletionRequest(BaseModel):
    messages: list[Message]
    tools: list[ToolSpec] = []
    output_schema_name: str
    output_json_schema: dict[str, Any]
    temperature: float = 0.0


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class CompletionResponse(BaseModel):
    """Either ``tool_calls`` (the model wants to use tools) or ``content`` (final answer)."""

    content: str | None = None
    tool_calls: list[ToolCall] = []
    usage: Usage = Usage()
    model: str


class ProviderError(Exception):
    """The provider failed to produce a response at all (network, auth, quota...)."""


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse: ...
