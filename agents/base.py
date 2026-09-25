"""The agent loop: tool use, then a validated structured output.

    ┌─ provider.complete(messages, tools, output schema)
    │    ├─ tool calls? → validate args → run tool → append results → loop
    │    └─ final JSON  → validate against output_schema (+ run-specific context)
    │                        ├─ ok      → return AgentResult
    └──────────────────────  └─ invalid → append the errors, ask for a fix (bounded)

Subclasses declare what they need: a prompt, tools, an output schema, and
optionally extra validation context. They do not write their own loop.
"""

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from .providers.base import CompletionRequest, LLMProvider, Message
from .schemas import AgentStep, AgentTrace
from .tools import AgentContext, Tool, ToolError

InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)


class AgentError(Exception):
    """The agent could not produce a valid output. Nothing is persisted."""

    def __init__(self, message: str, trace: AgentTrace):
        super().__init__(message)
        self.trace = trace


@dataclass
class ToolInvocation:
    tool: str
    args: BaseModel
    result: BaseModel


@dataclass
class AgentResult(Generic[OutT]):
    output: OutT
    trace: AgentTrace


@dataclass
class _RunState:
    messages: list[Message]
    tool_log: list[ToolInvocation] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    turns: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class Agent(Generic[InT, OutT]):
    name: ClassVar[str]
    system_prompt: ClassVar[str]
    output_schema: ClassVar[type[BaseModel]]
    tools: ClassVar[tuple[Tool, ...]] = ()
    max_turns: ClassVar[int] = 8
    max_validation_retries: ClassVar[int] = 2

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self._tools = {t.name: t for t in self.tools}

    def validation_context(self, task: InT, ctx: AgentContext, tool_log: list[ToolInvocation]) -> dict[str, Any]:
        """Extra facts the output validators check against (see schemas/)."""
        return {}

    def run(self, task: InT, ctx: AgentContext) -> AgentResult[OutT]:
        state = _RunState(
            messages=[
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=task.model_dump_json()),
            ]
        )
        request_base = {
            "tools": [t.spec() for t in self.tools],
            "output_schema_name": self.output_schema.__name__,
            "output_json_schema": self.output_schema.model_json_schema(),
        }

        while state.turns < self.max_turns:
            state.turns += 1
            response = self.provider.complete(CompletionRequest(messages=state.messages, **request_base))
            state.input_tokens += response.usage.input_tokens
            state.output_tokens += response.usage.output_tokens

            if response.tool_calls:
                state.messages.append(Message(role="assistant", tool_calls=response.tool_calls))
                for call in response.tool_calls:
                    content = self._invoke_tool(call.name, call.arguments_json, ctx, state)
                    state.messages.append(Message(role="tool", tool_call_id=call.id, name=call.name, content=content))
                continue

            raw = response.content or ""
            try:
                output = self.output_schema.model_validate_json(
                    raw, context=self.validation_context(task, ctx, state.tool_log)
                )
            except ValidationError as exc:
                errors = _summarize(exc)
                state.steps.append(AgentStep(type="validation_error", detail={"errors": errors}))
                if state.retries >= self.max_validation_retries:
                    raise AgentError(
                        f"{self.name}: output failed validation after {state.retries + 1} attempts: "
                        + "; ".join(errors),
                        self._trace(state),
                    ) from exc
                state.retries += 1
                state.messages.append(Message(role="assistant", content=raw))
                state.messages.append(
                    Message(
                        role="user",
                        content="Your answer failed validation:\n- "
                        + "\n- ".join(errors)
                        + f"\nReturn a corrected {self.output_schema.__name__} as JSON only.",
                    )
                )
                continue

            state.steps.append(AgentStep(type="output"))
            return AgentResult(output=output, trace=self._trace(state))

        raise AgentError(f"{self.name}: no final answer within {self.max_turns} turns", self._trace(state))

    def _invoke_tool(self, name: str, arguments_json: str, ctx: AgentContext, state: _RunState) -> str:
        """Run one tool call. Errors go back to the model as data instead of being raised."""
        tool = self._tools.get(name)
        try:
            if tool is None:
                raise ToolError(f"unknown tool '{name}'; available: {sorted(self._tools)}")
            try:
                args = tool.args_model.model_validate_json(arguments_json or "{}")
            except ValidationError as exc:
                raise ToolError("invalid arguments: " + "; ".join(_summarize(exc))) from exc
            result = tool.fn(ctx, args)
        except ToolError as exc:
            state.steps.append(AgentStep(type="tool_error", detail={"tool": name, "error": str(exc)}))
            return json.dumps({"error": str(exc)})

        state.tool_log.append(ToolInvocation(tool=name, args=args, result=result))
        state.steps.append(AgentStep(type="tool_call", detail={"tool": name, "args": args.model_dump(mode="json")}))
        return result.model_dump_json()

    def _trace(self, state: _RunState) -> AgentTrace:
        return AgentTrace(
            agent=self.name,
            provider=self.provider.name,
            turns=state.turns,
            validation_retries=state.retries,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            steps=state.steps,
        )


def _summarize(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        out.append(f"{loc}: {err['msg']}")
    return out
