"""The agent loop as a LangGraph ``StateGraph``: tool use, then a validated structured output.

    START ─▶ call_model ──tool calls──▶ run_tools ──┐
                 ▲    └──final answer─▶ validate ───┤
                 │                                  ├─ output set ─────────▶ END
                 │                                  ├─ retries exhausted ──▶ END (error)
                 │                                  ├─ turn budget spent ──▶ give_up ─▶ END
                 └──────────── otherwise ───────────┘

Validation is a first-class node, not something tacked on after the loop. Its
outcome (accept, retry with feedback, or give up) is an explicit edge in the
graph. Subclasses declare a prompt, tools, an output schema, and optionally
extra validation context. They do not write their own loop.

Run ``python -m agents.visualize`` to print this graph as Mermaid.
"""

import json
import operator
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Generic, Literal, TypedDict, TypeVar

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from .providers.base import CompletionRequest, CompletionResponse, LLMProvider, Message
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


class AgentState(TypedDict, total=False):
    # Inputs, fixed for the run.
    task: BaseModel
    ctx: AgentContext
    # Append-only channels: nodes return new items, LangGraph concatenates them.
    messages: Annotated[list[Message], operator.add]
    steps: Annotated[list[AgentStep], operator.add]
    tool_log: Annotated[list[ToolInvocation], operator.add]
    # Overwritten by the node that owns them.
    response: CompletionResponse
    turns: int
    retries: int
    input_tokens: int
    output_tokens: int
    output: BaseModel
    error: str


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
        self._request_base = {
            "tools": [t.spec() for t in self.tools],
            "output_schema_name": self.output_schema.__name__,
            "output_json_schema": self.output_schema.model_json_schema(),
        }
        self.graph = self._build_graph()

    def validation_context(self, task: InT, ctx: AgentContext, tool_log: list[ToolInvocation]) -> dict[str, Any]:
        """Extra facts the output validators check against (see schemas/)."""
        return {}

    def run(self, task: InT, ctx: AgentContext) -> AgentResult[OutT]:
        initial: AgentState = {
            "task": task,
            "ctx": ctx,
            "messages": [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=task.model_dump_json()),
            ],
            "steps": [],
            "tool_log": [],
            "turns": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        # Each turn visits at most three nodes; the turn budget, not LangGraph's
        # recursion limit, is what should stop a runaway loop.
        final = self.graph.invoke(initial, config={"recursion_limit": self.max_turns * 3 + 5})
        trace = self._trace(final)
        if "output" in final:
            return AgentResult(output=final["output"], trace=trace)
        raise AgentError(final["error"], trace)

    # ------------------------------------------------------------------ graph

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("call_model", self._call_model)
        graph.add_node("run_tools", self._run_tools)
        graph.add_node("validate", self._validate)
        graph.add_node("give_up", self._give_up)

        graph.add_edge(START, "call_model")
        graph.add_conditional_edges("call_model", self._route_response, ["run_tools", "validate"])
        graph.add_conditional_edges("run_tools", self._route_next, ["call_model", "give_up", END])
        graph.add_conditional_edges("validate", self._route_next, ["call_model", "give_up", END])
        graph.add_edge("give_up", END)
        return graph.compile(name=self.name)

    def _call_model(self, state: AgentState) -> AgentState:
        response = self.provider.complete(CompletionRequest(messages=state["messages"], **self._request_base))
        return {
            "response": response,
            "turns": state["turns"] + 1,
            "input_tokens": state["input_tokens"] + response.usage.input_tokens,
            "output_tokens": state["output_tokens"] + response.usage.output_tokens,
        }

    def _run_tools(self, state: AgentState) -> AgentState:
        calls = state["response"].tool_calls
        messages = [Message(role="assistant", tool_calls=calls)]
        steps, log = [], []
        for call in calls:
            content, step, invocation = self._invoke_tool(call.name, call.arguments_json, state["ctx"])
            messages.append(Message(role="tool", tool_call_id=call.id, name=call.name, content=content))
            steps.append(step)
            if invocation:
                log.append(invocation)
        return {"messages": messages, "steps": steps, "tool_log": log}

    def _validate(self, state: AgentState) -> AgentState:
        raw = state["response"].content or ""
        context = self.validation_context(state["task"], state["ctx"], state["tool_log"])
        try:
            output = self.output_schema.model_validate_json(raw, context=context)
        except ValidationError as exc:
            errors = _summarize(exc)
            step = AgentStep(type="validation_error", detail={"errors": errors})
            if state["retries"] >= self.max_validation_retries:
                attempts = state["retries"] + 1
                return {
                    "steps": [step],
                    "error": f"{self.name}: output failed validation after {attempts} attempts: " + "; ".join(errors),
                }
            feedback = (
                "Your answer failed validation:\n- "
                + "\n- ".join(errors)
                + f"\nReturn a corrected {self.output_schema.__name__} as JSON only."
            )
            return {
                "steps": [step],
                "retries": state["retries"] + 1,
                "messages": [Message(role="assistant", content=raw), Message(role="user", content=feedback)],
            }
        return {"output": output, "steps": [AgentStep(type="output")]}

    def _give_up(self, state: AgentState) -> AgentState:
        return {"error": f"{self.name}: no final answer within {self.max_turns} turns"}

    @staticmethod
    def _route_response(state: AgentState) -> Literal["run_tools", "validate"]:
        return "run_tools" if state["response"].tool_calls else "validate"

    def _route_next(self, state: AgentState) -> str:
        if "output" in state or "error" in state:
            return END
        if state["turns"] >= self.max_turns:
            return "give_up"
        return "call_model"

    # ------------------------------------------------------------------ helpers

    def _invoke_tool(
        self, name: str, arguments_json: str, ctx: AgentContext
    ) -> tuple[str, AgentStep, ToolInvocation | None]:
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
            step = AgentStep(type="tool_error", detail={"tool": name, "error": str(exc)})
            return json.dumps({"error": str(exc)}), step, None

        step = AgentStep(type="tool_call", detail={"tool": name, "args": args.model_dump(mode="json")})
        return result.model_dump_json(), step, ToolInvocation(tool=name, args=args, result=result)

    def _trace(self, state: AgentState) -> AgentTrace:
        return AgentTrace(
            agent=self.name,
            provider=self.provider.name,
            turns=state["turns"],
            validation_retries=state["retries"],
            input_tokens=state["input_tokens"],
            output_tokens=state["output_tokens"],
            steps=state["steps"],
        )


def _summarize(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        out.append(f"{loc}: {err['msg']}")
    return out
