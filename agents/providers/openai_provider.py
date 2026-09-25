"""OpenAI Chat Completions adapter. Requires ``pip install openai`` and OPENAI_API_KEY."""

import os

from .base import CompletionRequest, CompletionResponse, LLMProvider, Message, ProviderError, ToolCall, Usage


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 60.0):
        try:
            import openai
        except ImportError as exc:  # optional dependency
            raise ProviderError("LLM_PROVIDER=openai requires `pip install openai`") from exc
        self._openai = openai
        # The SDK already retries transient failures (429/5xx/timeouts) with backoff.
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"), timeout=timeout, max_retries=3
        )
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        kwargs = {
            "model": self.model,
            "temperature": request.temperature,
            "messages": [_to_openai(m) for m in request.messages],
            # Non-strict: pydantic schemas use features strict mode rejects. Our own
            # validation layer is the guarantee; this is only a strong hint.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_schema_name,
                    "schema": request.output_json_schema,
                    "strict": False,
                },
            },
        }
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in request.tools
            ]
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except self._openai.OpenAIError as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        msg = resp.choices[0].message
        return CompletionResponse(
            content=msg.content,
            tool_calls=[
                ToolCall(id=tc.id, name=tc.function.name, arguments_json=tc.function.arguments or "{}")
                for tc in (msg.tool_calls or [])
            ],
            usage=Usage(
                input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            ),
            model=resp.model,
        )


def _to_openai(m: Message) -> dict:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or ""}
    out: dict = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments_json}}
            for tc in m.tool_calls
        ]
    return out
