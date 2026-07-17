from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    kind: Literal["contract", "custom"] = "custom"
    namespace: str | None = None


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any]
    raw_name: str | None = None


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False
    data: Mapping[str, Any] | None = None


ToolHandler = Callable[[ToolCall], Awaitable[ToolResult]]


@runtime_checkable
class ToolExecutor(Protocol):
    async def invoke(self, call: ToolCall) -> ToolResult: ...
