"""CodexExecBackend unit tests — NO live codex (a fake subprocess feeds real JSONL fixtures)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tilldone.backends.codex_exec import CodexExecBackend, CodexRunHandle, build_argv
from tilldone.backends.mcp_bridge import BridgeInfo
from tilldone.core.spec import AgentRunSpec, ContextDirView, ResumeHandle
from tilldone.testing.conformance import assert_protocol_invariants

_BRIDGE = BridgeInfo(url="http://127.0.0.1:54321/mcp", token="tok-abc")


# ---------------- launch-spec (RT-GATE-1) ----------------

def test_build_argv_fresh_has_bypass_and_bridge(tmp_path):
    spec = AgentRunSpec(prompt="do it", tools=(), cwd=tmp_path,
                        context_dirs=(ContextDirView(id="d", path=tmp_path / "ctx"),))
    argv = build_argv(spec, _BRIDGE)
    assert argv[:2] == ["codex", "exec"]
    assert "--dangerously-bypass-approvals-and-sandbox" in argv  # RT-GATE-1
    assert "--json" in argv and "--skip-git-repo-check" in argv
    assert "-C" in argv and str(tmp_path) in argv
    assert "--add-dir" in argv and str(tmp_path / "ctx") in argv
    joined = " ".join(argv)
    assert 'mcp_servers.tilldone.url="http://127.0.0.1:54321/mcp"' in joined
    assert 'mcp_servers.tilldone.bearer_token_env_var="TILLDONE_MCP_TOKEN"' in joined
    assert argv[-1] == "do it"


def test_build_argv_allows_user_supplied_mcp_server_extra_config(tmp_path):
    spec = AgentRunSpec(prompt="do it", tools=(), cwd=tmp_path)
    extra = [
        'mcp_servers.files.url="http://127.0.0.1:7777/mcp"',
        'mcp_servers.files.bearer_token_env_var="FILES_MCP_TOKEN"',
    ]
    argv = build_argv(spec, _BRIDGE, extra_config=extra)
    joined = " ".join(argv)
    assert 'mcp_servers.tilldone.url="http://127.0.0.1:54321/mcp"' in joined
    assert 'mcp_servers.files.url="http://127.0.0.1:7777/mcp"' in joined
    assert 'mcp_servers.files.bearer_token_env_var="FILES_MCP_TOKEN"' in joined


def test_build_argv_rejects_reserved_tilldone_mcp_override(tmp_path):
    spec = AgentRunSpec(prompt="do it", tools=(), cwd=tmp_path)
    with pytest.raises(ValueError, match="tilldone.*reserved"):
        build_argv(
            spec,
            _BRIDGE,
            extra_config=['mcp_servers.tilldone.url="http://127.0.0.1:9/mcp"'],
        )


def test_build_argv_resume_uses_subcommand_no_cwd(tmp_path):
    spec = AgentRunSpec(prompt="again", tools=(), cwd=tmp_path,
                        resume=ResumeHandle("codex", {"session_id": "uuid-9"}))
    argv = build_argv(spec, _BRIDGE)
    assert argv[:3] == ["codex", "exec", "resume"]
    assert "--dangerously-bypass-approvals-and-sandbox" in argv  # RT-GATE-1 on resume too
    assert "-C" not in argv  # resume reuses the session cwd
    assert "uuid-9" in argv and argv[-1] == "again"


# ---------------- event adapter via fake subprocess ----------------

class _FakeStream:
    def __init__(self, lines):
        self._lines = [(l + "\n").encode() for l in lines]
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._i]
        self._i += 1
        return line


class _FakeProc:
    def __init__(self, lines, returncode=0):
        self.stdout = _FakeStream(lines)
        self.stderr = _FakeStream([])
        self.returncode = returncode

    async def wait(self):
        return self.returncode

    def kill(self):
        ...


class _FakeBridge:
    def __init__(self):
        self.stopped = False

    async def stop(self):
        self.stopped = True


def _lines(*objs):
    return [json.dumps(o) for o in objs]


async def test_adapter_command_and_agent_message_conformant():
    bridge = _FakeBridge()
    lines = _lines(
        {"type": "thread.started", "thread_id": "019ef-aaaa"},
        {"type": "turn.started"},
        {"type": "item.started", "item": {"type": "command_execution", "id": "c1",
                                          "command": "echo hi", "status": "in_progress"}},
        {"type": "item.completed", "item": {"type": "command_execution", "id": "c1",
                                            "command": "echo hi", "exit_code": 0,
                                            "status": "completed"}},
        {"type": "item.completed", "item": {"type": "agent_message", "id": "a0", "text": "DONE"}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}},
    )
    handle = CodexRunHandle(_FakeProc(lines), bridge)
    events = [ev async for ev in handle.events]
    outcome = await handle.outcome()

    kinds = [e.kind.value for e in events]
    assert kinds == ["run_started", "command", "command", "assistant_text", "run_completed"]
    assert outcome.status == "completed" and outcome.final_text == "DONE"
    assert outcome.resume == ResumeHandle("codex", {"session_id": "019ef-aaaa"})
    assert bridge.stopped is True  # bridge torn down (RT-MCP-1 cleanup)
    assert_protocol_invariants(events, outcome)


async def test_adapter_mcp_tool_call_pairs_call_and_result():
    lines = _lines(
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "item.started", "item": {"type": "mcp_tool_call", "id": "m1",
                                          "server": "tilldone", "tool": "return"}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "id": "m1",
                                            "server": "tilldone", "tool": "return",
                                            "status": "completed",
                                            "result": {"content": [{"type": "text", "text": "recorded"}]}}},
        {"type": "turn.completed", "usage": {}},
    )
    handle = CodexRunHandle(_FakeProc(lines), _FakeBridge())
    events = [ev async for ev in handle.events]
    outcome = await handle.outcome()
    tc = next(e for e in events if e.kind.value == "tool_call")
    tr = next(e for e in events if e.kind.value == "tool_result")
    assert tc.data["name"] == "return" and tc.call_id == "m1"
    assert tr.call_id == "m1" and tr.data["is_error"] is False
    assert_protocol_invariants(events, outcome)


async def test_adapter_turn_failed_is_failed_outcome():
    lines = _lines(
        {"type": "thread.started", "thread_id": "t2"},
        {"type": "turn.failed", "error": {"message": "model error"}},
    )
    handle = CodexRunHandle(_FakeProc(lines, returncode=1), _FakeBridge())
    events = [ev async for ev in handle.events]
    outcome = await handle.outcome()
    assert events[-1].kind.value == "run_failed"
    assert outcome.status == "failed" and outcome.error.code == "fatal"


async def test_adapter_exit_without_turn_completed_fails():
    lines = _lines({"type": "thread.started", "thread_id": "t3"},
                   {"type": "item.completed", "item": {"type": "agent_message", "id": "a", "text": "x"}})
    handle = CodexRunHandle(_FakeProc(lines, returncode=1), _FakeBridge())
    events = [ev async for ev in handle.events]
    outcome = await handle.outcome()
    assert events[-1].kind.value == "run_failed"
    assert outcome.status == "failed" and outcome.error.code == "fatal"


async def test_error_then_turn_failed_emits_single_terminal():
    # Codex emits `error` then `turn.failed` for one failure; the adapter must emit exactly
    # ONE terminal (conformance) — `error` is a notice, `turn.failed` is the terminal.
    lines = _lines(
        {"type": "thread.started", "thread_id": "tt"},
        {"type": "error", "message": "Selected model is at capacity. Please try a different model."},
        {"type": "turn.failed", "error": {"message": "at capacity"}},
    )
    handle = CodexRunHandle(_FakeProc(lines, returncode=0), _FakeBridge())
    events = [ev async for ev in handle.events]
    outcome = await handle.outcome()
    terminals = [e for e in events if e.kind.value in ("run_completed", "run_failed")]
    assert len(terminals) == 1 and terminals[0].kind.value == "run_failed"
    assert any(e.kind.value == "backend_notice" for e in events)
    assert outcome.status == "failed" and outcome.error.code == "transient"  # capacity -> retry
    assert_protocol_invariants(events, outcome)


def test_capabilities():
    caps = CodexExecBackend().capabilities()
    assert caps.host_tool_calls is True and caps.session_resume is True
    assert caps.command_events is True and caps.mcp_http is True


def test_codex_backend_public_default_is_exec_backend():
    from tilldone.backends.codex import CodexBackend

    backend = CodexBackend(extra_config=['model_reasoning_effort="low"'])
    assert isinstance(backend, CodexExecBackend)
    assert backend.name == "codex"
    caps = backend.capabilities()
    assert caps.host_tool_calls is True
    assert caps.mcp_http is True
    assert caps.session_resume is True


def test_codex_backend_rejects_reserved_mcp_server_name():
    from tilldone.backends.codex import CodexBackend

    with pytest.raises(ValueError, match="tilldone.*reserved"):
        CodexBackend(extra_config=['mcp_servers."tilldone".url="http://127.0.0.1:9/mcp"'])
