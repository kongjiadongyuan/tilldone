from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tilldone.core.contract import CompletionContract, ContractState, matches_type_hint
from tilldone.core.tools import ToolCall, ToolDef, ToolHandler, ToolResult


class ToolRegistry:
    """The core's ToolExecutor implementation: routes a ToolCall to a handler.

    Contract-tool handlers (return/todo/finish) mutate ContractState and enforce
    the declared contract (declared names + type hints). Custom-tool handlers run
    host-side. Unknown tool names return an error result (never silently ignored).
    """

    def __init__(self, state: ContractState, contract: CompletionContract | None = None):
        self.state = state
        self.contract = contract
        self._handlers: dict[str, ToolHandler] = {}
        self._defs: list[ToolDef] = []

    def add_contract_tools(self) -> None:
        declared_returns = {
            r.name: r for r in (self.contract.returns if self.contract else ())
        }
        declared_todos = {t.name for t in (self.contract.todos if self.contract else ())}

        async def _return(call: ToolCall) -> ToolResult:
            recorded: list[str] = []
            rejected: list[str] = []
            for k, v in call.arguments.items():
                r = declared_returns.get(k)
                if r is None:
                    rejected.append(f"{k} (undeclared)")
                    continue
                if not matches_type_hint(v, r.type_hint):
                    rejected.append(f"{k} (expected {r.type_hint})")
                    continue
                self.state.returns[k] = v
                recorded.append(k)
            if rejected:
                return ToolResult(
                    content=f"recorded={recorded}; rejected={rejected}", is_error=True
                )
            return ToolResult(content=f"recorded={recorded}")

        async def _todo(call: ToolCall) -> ToolResult:
            name = call.arguments.get("name")
            if name not in declared_todos:
                return ToolResult(content=f"unknown todo: {name!r}", is_error=True)
            self.state.todos_done[name] = call.arguments.get("note", "")
            return ToolResult(content=f"todo done: {name}")

        async def _finish(call: ToolCall) -> ToolResult:
            self.state.finished = True
            return ToolResult(content="finished")

        self._handlers.update({"return": _return, "todo": _todo, "finish": _finish})

    def add_custom(self, name: str, handler: ToolHandler,
                   description: str = "", input_schema: Mapping[str, Any] | None = None) -> None:
        self._handlers[name] = handler
        self._defs.append(
            ToolDef(name, description, input_schema or {"type": "object"}, kind="custom")
        )

    def tool_defs(self, contract_defs: Sequence[ToolDef]) -> list[ToolDef]:
        return list(contract_defs) + self._defs

    async def invoke(self, call: ToolCall) -> ToolResult:
        handler = self._handlers.get(call.name)
        if handler is None:
            return ToolResult(content=f"unknown tool: {call.name}", is_error=True)
        return await handler(call)
