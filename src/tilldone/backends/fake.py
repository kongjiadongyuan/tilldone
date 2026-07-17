from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from tilldone.core.enums import EventKind
from tilldone.core.events import AgentEvent
from tilldone.core.spec import (
    AgentRunSpec, BackendCapabilities, BackendError, ResumeHandle, RunOutcome,
)
from tilldone.core.tools import ToolCall, ToolExecutor


@dataclass
class EmitText:
    text: str


@dataclass
class CallTool:
    name: str
    arguments: dict


@dataclass
class WriteFile:
    path: Path
    content: str


@dataclass
class Finish:
    pass


class Script(list):
    """An ordered list of scripted steps for FakeBackend to replay."""


class FakeRunHandle:
    def __init__(self, steps, executor: ToolExecutor, structured: dict | None = None,
                 fail_error: BackendError | None = None):
        self._steps = steps
        self._executor = executor
        self._structured = structured
        self._fail_error = fail_error
        self._seq = 0
        self._final = ""

    @property
    async def events(self) -> AsyncIterator[AgentEvent]:
        if self._fail_error is not None:
            self._seq += 1
            yield AgentEvent(EventKind.RUN_FAILED, self._seq,
                             {"error": self._fail_error.message, "code": self._fail_error.code})
            return
        for step in self._steps:
            self._seq += 1
            if isinstance(step, EmitText):
                self._final = step.text
                yield AgentEvent(EventKind.ASSISTANT_TEXT, self._seq, {"text": step.text})
            elif isinstance(step, WriteFile):
                step.path.parent.mkdir(parents=True, exist_ok=True)
                step.path.write_text(step.content)
                yield AgentEvent(EventKind.FILE_CHANGED, self._seq, {"path": str(step.path)})
            elif isinstance(step, CallTool):
                call_id = f"call-{self._seq}"
                res = await self._executor.invoke(ToolCall(call_id, step.name, step.arguments))
                yield AgentEvent(EventKind.TOOL_RESULT, self._seq,
                                 {"name": step.name, "content": res.content,
                                  "is_error": res.is_error}, call_id=call_id)
            elif isinstance(step, Finish):
                call_id = f"call-{self._seq}"
                res = await self._executor.invoke(ToolCall(call_id, "finish", {}))
                yield AgentEvent(EventKind.TOOL_RESULT, self._seq,
                                 {"name": "finish", "content": res.content}, call_id=call_id)
        self._seq += 1
        yield AgentEvent(EventKind.RUN_COMPLETED, self._seq, {"final": self._final})

    async def outcome(self) -> RunOutcome:
        if self._fail_error is not None:
            return RunOutcome(status="failed",
                              resume=ResumeHandle("fake", {"id": "1"}),
                              error=self._fail_error)
        return RunOutcome(status="completed",
                          resume=ResumeHandle("fake", {"id": "1"}),
                          final_text=self._final,
                          structured_output=self._structured)

    async def cancel(self) -> None:
        ...


class FakeBackend:
    name = "fake"

    def __init__(self, script: Script, *, host_tool_calls: bool = True,
                 structured_output: dict | None = None,
                 fail_sequence: "list[BackendError | None] | None" = None):
        self._script = script
        self._host = host_tool_calls
        self._structured = structured_output
        self._fail_sequence = list(fail_sequence or [])

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            host_tool_calls=self._host,
            structured_final_output=self._structured is not None,
            session_resume=True,
        )

    async def start(self, spec: AgentRunSpec, executor: ToolExecutor) -> FakeRunHandle:
        fail_error = self._fail_sequence.pop(0) if self._fail_sequence else None
        return FakeRunHandle(list(self._script), executor,
                             structured=self._structured, fail_error=fail_error)

    async def aclose(self) -> None:
        ...
