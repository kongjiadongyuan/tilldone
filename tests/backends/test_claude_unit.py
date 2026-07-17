"""ClaudeBackend unit tests — NO live SDK calls (a fake `query` is injected)."""

from __future__ import annotations

from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from tilldone.backends.claude import (
    TOOL_PREFIX,
    ClaudeBackend,
    build_options,
    project_tools,
)
from tilldone.core.contract import (
    CompletionContract,
    ContractEvaluator,
    ContractState,
    ReturnExpectation,
)
from tilldone.core.registry import ToolRegistry
from tilldone.core.spec import AgentRunSpec, ResumeHandle
from tilldone.core.tools import ToolCall, ToolResult
from tilldone.testing.conformance import assert_protocol_invariants


def _contract():
    return CompletionContract(outputs=(), returns=(ReturnExpectation("ok", "bool", True),),
                              todos=(), require_finish=True)


def _contract_tool_defs():
    return ContractEvaluator(_contract()).tool_defs(host_tool_calls=True)


class _StubExecutor:
    def __init__(self):
        self.calls: list[ToolCall] = []

    async def invoke(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return ToolResult(content="recorded", is_error=False)


def _fake_query(messages):
    async def query_fn(*, prompt, options):  # matches sdk.query signature shape
        for m in messages:
            yield m
    return query_fn


# ---------------- tool projection ----------------

async def test_project_tools_wire_names_and_bridge():
    ex = _StubExecutor()
    sdk_tools, allowed = project_tools(_contract_tool_defs(), ex)
    assert allowed == [TOOL_PREFIX + n for n in ("return", "todo", "finish")]
    names = {t.name for t in sdk_tools}
    assert names == {"return", "todo", "finish"}
    # open-object 'return' schema must carry 'properties' so the SDK uses it verbatim
    ret = next(t for t in sdk_tools if t.name == "return")
    assert "properties" in ret.input_schema
    # handler bridges to executor.invoke and converts ToolResult -> MCP content
    out = await ret.handler({"ok": True})
    assert out == {"content": [{"type": "text", "text": "recorded"}], "is_error": False}
    assert ex.calls and ex.calls[0].name == "return" and ex.calls[0].arguments == {"ok": True}


# ---------------- launch-spec (RT-GATE-1) ----------------

def test_build_options_bypass_permissions_snapshot(tmp_path):
    ex = _StubExecutor()
    sdk_tools, allowed = project_tools(_contract_tool_defs(), ex)
    spec = AgentRunSpec(prompt="go", tools=(), cwd=tmp_path)
    opts = build_options(spec, sdk_tools, allowed)
    # RT-GATE-1: the launch spec MUST request bypassPermissions
    assert opts.permission_mode == "bypassPermissions"
    assert "tilldone" in opts.mcp_servers
    assert opts.mcp_servers["tilldone"]["type"] == "sdk"
    assert opts.allowed_tools == allowed
    assert opts.cwd == str(tmp_path)
    assert opts.output_format is None and opts.resume is None
    # hermetic isolation (T5): no host settings, only our MCP server
    assert opts.setting_sources == [] and opts.strict_mcp_config is True


def test_build_options_output_schema_resume_instructions(tmp_path):
    ex = _StubExecutor()
    sdk_tools, allowed = project_tools((), ex)
    spec = AgentRunSpec(
        prompt="go", tools=(), cwd=tmp_path,
        output_schema={"type": "object", "properties": {}},
        resume=ResumeHandle("claude", {"session_id": "sess-X"}),
        system_instructions="be terse", developer_instructions="use tools",
    )
    opts = build_options(spec, sdk_tools, allowed)
    assert opts.output_format == {"type": "json_schema", "schema": {"type": "object", "properties": {}}}
    assert opts.resume == "sess-X"
    assert opts.system_prompt["type"] == "preset" and opts.system_prompt["preset"] == "claude_code"
    assert "be terse" in opts.system_prompt["append"] and "use tools" in opts.system_prompt["append"]


def test_build_options_timeout_not_forwarded_to_cli(tmp_path):
    """timeout_s must NOT leak into extra_args: a value-less `--max-turns` flag corrupts
    CLI arg parsing (claude >= 2.1.x hangs pre-handshake -> initialize timeout).
    The core loop policy owns timeout enforcement."""
    ex = _StubExecutor()
    sdk_tools, allowed = project_tools((), ex)
    spec = AgentRunSpec(prompt="go", tools=(), cwd=tmp_path, timeout_s=120.0)
    opts = build_options(spec, sdk_tools, allowed)
    assert "max-turns" not in dict(opts.extra_args)


# ---------------- event adapter + outcome via injected fake query ----------------

async def test_start_adapts_messages_and_is_conformant(tmp_path):
    st = ContractState()
    reg = ToolRegistry(st, _contract())
    reg.add_contract_tools()
    messages = [
        SystemMessage(subtype="init", data={"session_id": "sess-1"}),
        AssistantMessage(
            content=[TextBlock(text="working"),
                     ToolUseBlock(id="tu1", name=TOOL_PREFIX + "return", input={"ok": True})],
            model="claude", session_id="sess-1"),
        UserMessage(content=[ToolResultBlock(tool_use_id="tu1", content="recorded", is_error=False)]),
        ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                      num_turns=1, session_id="sess-1", result="done", structured_output=None),
    ]
    backend = ClaudeBackend(query_fn=_fake_query(messages))
    spec = AgentRunSpec(prompt="go", tools=(), cwd=tmp_path)
    handle = await backend.start(spec, reg)
    events = [ev async for ev in handle.events]
    outcome = await handle.outcome()

    kinds = [e.kind.value for e in events]
    assert kinds == ["run_started", "assistant_text", "tool_call", "tool_result", "run_completed"]
    tc = next(e for e in events if e.kind.value == "tool_call")
    assert tc.data["name"] == "return" and tc.call_id == "tu1"
    tr = next(e for e in events if e.kind.value == "tool_result")
    assert tr.call_id == "tu1"
    assert outcome.status == "completed" and outcome.final_text == "done"
    assert outcome.resume == ResumeHandle("claude", {"session_id": "sess-1"})
    # forward-compatible protocol invariants (same checker used for every backend)
    assert_protocol_invariants(events, outcome)


async def test_reasoning_block_maps_to_reasoning(tmp_path):
    messages = [
        AssistantMessage(content=[ThinkingBlock(thinking="hmm", signature="s")],
                         model="claude", session_id="s1"),
        ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                      num_turns=1, session_id="s1", result="x"),
    ]
    backend = ClaudeBackend(query_fn=_fake_query(messages))
    handle = await backend.start(AgentRunSpec(prompt="go", tools=(), cwd=tmp_path), _StubExecutor())
    kinds = [ev.kind.value async for ev in handle.events]
    assert "reasoning" in kinds


async def test_result_error_maps_to_failed_outcome(tmp_path):
    messages = [
        ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=True,
                      num_turns=1, session_id="s2", api_error_status=429),
    ]
    backend = ClaudeBackend(query_fn=_fake_query(messages))
    handle = await backend.start(AgentRunSpec(prompt="go", tools=(), cwd=tmp_path), _StubExecutor())
    events = [ev async for ev in handle.events]
    outcome = await handle.outcome()
    assert events[-1].kind.value == "run_failed"
    assert outcome.status == "failed" and outcome.error is not None
    assert outcome.error.code == "transient" and outcome.error.status == 429


async def test_stream_without_result_is_protocol_error(tmp_path):
    messages = [AssistantMessage(content=[TextBlock(text="oops")], model="claude", session_id="s3")]
    backend = ClaudeBackend(query_fn=_fake_query(messages))
    handle = await backend.start(AgentRunSpec(prompt="go", tools=(), cwd=tmp_path), _StubExecutor())
    events = [ev async for ev in handle.events]
    outcome = await handle.outcome()
    assert events[-1].kind.value == "run_failed"
    assert outcome.status == "failed" and outcome.error.code == "protocol"


def test_capabilities():
    caps = ClaudeBackend().capabilities()
    assert caps.host_tool_calls is True and caps.structured_final_output is True
    assert caps.session_resume is True and caps.in_process_mcp is True
