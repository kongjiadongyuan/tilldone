"""ClaudeBackend — drives a single turn over `claude-agent-sdk` (live).

Per-turn, stateless driver: one AgentRunSpec -> one `query()` -> a normalized
AgentEvent stream + a RunOutcome. NO GATING: launches `permission_mode="bypassPermissions"`.
Contract + custom tools are projected to an in-process SDK MCP server whose handlers
bridge back into the core ToolExecutor. Verified against claude-agent-sdk 0.2.107.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

import claude_agent_sdk as sdk
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
)
from claude_agent_sdk import tool as sdk_tool

from tilldone.core.enums import EventKind
from tilldone.core.events import AgentEvent
from tilldone.core.inputs import TextInput, normalize_prompt
from tilldone.core.spec import (
    AgentRunSpec,
    BackendCapabilities,
    BackendError,
    ResumeHandle,
    RunOutcome,
)
from tilldone.core.tools import ToolCall, ToolDef, ToolExecutor

MCP_SERVER_NAME = "tilldone"
TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"

# HTTP statuses -> normalized BackendError.code (RT-CFG-4: classify by code, not text).
_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 529})
_AUTH_STATUS = frozenset({401, 403})
_PROTOCOL_STATUS = frozenset({400, 404, 422})

_ASSISTANT_ERROR_CODE = {
    "authentication_failed": "auth",
    "billing_error": "auth",
    "rate_limit": "transient",
    "server_error": "transient",
    "invalid_request": "protocol",
    "unknown": "fatal",
}


def _prompt_text(prompt: Any) -> str:
    """Flatten the (possibly multimodal) prompt to text for the CLI string input."""
    return "\n".join(i.text for i in normalize_prompt(prompt) if isinstance(i, TextInput))


def _normalized_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """create_sdk_mcp_server uses a dict schema verbatim ONLY if it has both
    'type' and 'properties'; otherwise it treats the dict as a name->type map.
    Ensure object schemas carry 'properties' so they pass through unchanged."""
    out = dict(schema)
    if out.get("type") == "object" and "properties" not in out:
        out["properties"] = {}
    return out


def project_tools(spec_tools: Sequence[ToolDef], executor: ToolExecutor):
    """ToolDef[] -> (SdkMcpTool[], allowed_tool_wire_names[]).

    Each projected handler bridges an SDK MCP tool call back to the core
    ToolExecutor and converts the ToolResult into MCP content.
    """
    sdk_tools = []
    allowed: list[str] = []
    counter = {"n": 0}

    def make_handler(logical_name: str):
        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            counter["n"] += 1
            call = ToolCall(
                call_id=f"claude-{logical_name}-{counter['n']}",
                name=logical_name,
                arguments=args or {},
                raw_name=TOOL_PREFIX + logical_name,
            )
            result = await executor.invoke(call)
            return {
                "content": [{"type": "text", "text": result.content}],
                "is_error": result.is_error,
            }

        return handler

    for td in spec_tools:
        allowed.append(TOOL_PREFIX + td.name)
        sdk_tools.append(
            sdk_tool(td.name, td.description, _normalized_schema(dict(td.input_schema)))(
                make_handler(td.name)
            )
        )
    return sdk_tools, allowed


def build_options(spec: AgentRunSpec, sdk_tools, allowed: list[str],
                  *, model: str | None = None) -> ClaudeAgentOptions:
    """AgentRunSpec -> ClaudeAgentOptions. NO GATING => permission_mode=bypassPermissions."""
    server = create_sdk_mcp_server(MCP_SERVER_NAME, tools=sdk_tools)
    options = ClaudeAgentOptions(
        mcp_servers={MCP_SERVER_NAME: server},
        allowed_tools=allowed,
        permission_mode="bypassPermissions",  # NO GATING — RT-GATE-1 launch-spec invariant
        cwd=str(spec.cwd),
        env=dict(spec.env),
        add_dirs=[str(cd.path) for cd in spec.context_dirs],
        model=model or spec.model,
        # Hermetic isolation (T5): ignore the user's ~/.claude settings + any project
        # .mcp.json; only our in-process MCP server is loaded. Keeps runs reproducible
        # and prevents ambient host config from polluting a contract run.
        setting_sources=[],
        strict_mcp_config=True,
    )
    instructions = "\n\n".join(
        x for x in (spec.system_instructions, spec.developer_instructions) if x
    )
    if instructions:
        options.system_prompt = {
            "type": "preset",
            "preset": "claude_code",
            "append": instructions,
        }
    if spec.output_schema is not None:
        options.output_format = {"type": "json_schema", "schema": dict(spec.output_schema)}
    if spec.resume is not None:
        options.resume = spec.resume.token.get("session_id")
    # NOTE: spec.timeout_s is enforced by the core loop policy, not forwarded to the
    # CLI. A previous version forwarded it as a value-less `--max-turns` flag, which
    # corrupts CLI arg parsing (claude >= 2.1.x hangs pre-handshake -> initialize
    # control request timeout). Regression: tests/backends/test_claude_unit.py.
    return options


def _map_result_error(msg: ResultMessage) -> BackendError:
    status = msg.api_error_status
    if status is not None:
        if status in _TRANSIENT_STATUS:
            code = "transient"
        elif status in _AUTH_STATUS:
            code = "auth"
        elif status in _PROTOCOL_STATUS:
            code = "protocol"
        else:
            code = "fatal"
        return BackendError(code=code, message=f"api error {status} ({msg.subtype})",
                            status=status, raw=msg)
    return BackendError(code="fatal", message=f"result error: {msg.subtype}", raw=msg)


class ClaudeRunHandle:
    """Streams normalized AgentEvents from an SDK message iterator; resolves a RunOutcome."""

    def __init__(self, message_stream: AsyncIterator[Any]):
        self._messages = message_stream
        self._seq = 0
        self._session_id: str | None = None
        self._final_text = ""
        self._structured: Any = None
        self._usage: dict[str, Any] | None = None
        self._error: BackendError | None = None
        self._failed = False
        self._terminated = False

    def _ev(self, kind: EventKind, data: dict[str, Any], **kw) -> AgentEvent:
        self._seq += 1
        return AgentEvent(kind, self._seq, data, session_id=self._session_id, **kw)

    @property
    async def events(self) -> AsyncIterator[AgentEvent]:
        try:
            async for msg in self._messages:
                for ev in self._adapt(msg):
                    yield ev
            if not self._terminated:
                # Stream ended without a ResultMessage — a protocol violation, surfaced.
                self._failed = True
                self._error = BackendError(code="protocol", message="stream ended without result")
                self._terminated = True
                yield self._ev(EventKind.RUN_FAILED, {"error": self._error.message})
        finally:
            # ensure the SDK's query() generator (and its `claude` subprocess) is closed,
            # so repeated runs in one process don't leak child processes.
            aclose = getattr(self._messages, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()

    def _adapt(self, msg: Any) -> list[AgentEvent]:
        evs: list[AgentEvent] = []
        if isinstance(msg, SystemMessage):
            data = msg.data if isinstance(msg.data, dict) else {}
            sid = data.get("session_id")
            if sid:
                self._session_id = sid
            if msg.subtype == "init":
                evs.append(self._ev(EventKind.RUN_STARTED, {"subtype": msg.subtype}, raw=msg))
        elif isinstance(msg, AssistantMessage):
            if msg.session_id:
                self._session_id = msg.session_id
            if msg.error:
                self._error = BackendError(
                    code=_ASSISTANT_ERROR_CODE.get(msg.error, "fatal"),
                    message=f"assistant error: {msg.error}",
                )
            for block in msg.content:
                if isinstance(block, TextBlock):
                    self._final_text = block.text
                    evs.append(self._ev(EventKind.ASSISTANT_TEXT, {"text": block.text}, raw=msg))
                elif isinstance(block, ThinkingBlock):
                    evs.append(self._ev(EventKind.REASONING, {"thinking": block.thinking}, raw=msg))
                elif isinstance(block, ToolUseBlock):
                    name = block.name
                    if name.startswith(TOOL_PREFIX):
                        name = name[len(TOOL_PREFIX):]
                    evs.append(self._ev(EventKind.TOOL_CALL, {"name": name, "input": block.input},
                                        call_id=block.id, raw=msg))
        elif isinstance(msg, UserMessage):
            content = msg.content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, ToolResultBlock):
                        evs.append(self._ev(EventKind.TOOL_RESULT,
                                            {"is_error": bool(block.is_error)},
                                            call_id=block.tool_use_id, raw=msg))
        elif isinstance(msg, ResultMessage):
            if msg.session_id:
                self._session_id = msg.session_id
            self._structured = msg.structured_output
            self._usage = msg.usage
            if msg.result:
                self._final_text = msg.result
            self._terminated = True
            if msg.is_error:
                self._failed = True
                self._error = _map_result_error(msg)
                evs.append(self._ev(EventKind.RUN_FAILED,
                                    {"subtype": msg.subtype, "status": msg.api_error_status}, raw=msg))
            else:
                evs.append(self._ev(EventKind.RUN_COMPLETED, {"final": self._final_text}, raw=msg))
        # other message kinds (StreamEvent, Task*) are not part of the v1 stable view
        return evs

    async def outcome(self) -> RunOutcome:
        resume = (
            ResumeHandle("claude", {"session_id": self._session_id})
            if self._session_id
            else None
        )
        if self._failed:
            return RunOutcome(status="failed", resume=resume, final_text=self._final_text,
                              structured_output=self._structured, usage=self._usage,
                              error=self._error)
        return RunOutcome(status="completed", resume=resume, final_text=self._final_text,
                          structured_output=self._structured, usage=self._usage)

    async def cancel(self) -> None:
        aclose = getattr(self._messages, "aclose", None)
        if aclose is not None:
            await aclose()


class ClaudeBackend:
    """Backend over claude-agent-sdk `query()` (one turn per start())."""

    name = "claude"

    def __init__(self, query_fn: Callable[..., AsyncIterator[Any]] = sdk.query,
                 *, model: str | None = None):
        self._query = query_fn
        # model: explicit arg wins, else TILLDONE_CLAUDE_MODEL env (gateway routing),
        # else None (CLI default / spec.model). Lets live tests route to a gateway model.
        self._model = model or os.environ.get("TILLDONE_CLAUDE_MODEL")

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            host_tool_calls=True,
            structured_final_output=True,
            session_resume=True,
            streaming_events=True,
            reasoning_events=True,
            in_process_mcp=True,
            mcp_http=True,
            mcp_stdio=True,
            multimodal_input=True,
            skills=True,
        )

    async def start(self, spec: AgentRunSpec, executor: ToolExecutor) -> ClaudeRunHandle:
        if spec.resume is not None and spec.resume.backend != self.name:
            # RT-SESS-5: a resume handle minted by another backend must NOT be silently used as
            # an opaque string (cross-backend resume is a non-goal). Fail fast — before any SDK
            # work — so a caller can't believe a Codex session was resumed here.
            raise ValueError(
                f"resume handle is for backend {spec.resume.backend!r}, not {self.name!r} — "
                "cross-backend resume is a non-goal (RT-SESS-5)")
        sdk_tools, allowed = project_tools(spec.tools, executor)
        options = build_options(spec, sdk_tools, allowed, model=spec.model or self._model)
        stream = self._query(prompt=_prompt_text(spec.prompt), options=options)
        return ClaudeRunHandle(stream)

    async def aclose(self) -> None:
        ...
