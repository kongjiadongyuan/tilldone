"""Unit tests for the structured-output CodexAppServerBackend.

All tests drive a *real* :class:`AppServerTransport` through the in-memory
``FakePeer``: we script the SERVER side (thread/started → turn/started →
item/* → thread/tokenUsage/updated → turn/completed) exactly as the live
``codex app-server`` would, using frame shapes matching the live protocol.
NO real codex spawn, no gateway.

The backend is injected the fake transport via its documented ``transport_factory``
seam, so ``start()`` never spawns a subprocess yet exercises the entire
initialize→thread/start→turn/start handshake + event normalization + outcome
settlement path end-to-end.

asyncio_mode="auto" (pyproject) → async tests need no decorator.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from tilldone.backends.app_server.backend import CodexAppServerBackend
from tilldone.backends.app_server.events import map_notification
from tilldone.core.enums import EventKind
from tilldone.core.spec import AgentRunSpec, BackendError, ResumeHandle
from tilldone.core.tools import ToolExecutor, ToolResult
from tilldone.testing.conformance import assert_protocol_invariants, collect_run

from .fake_peer import FakePeer

THREAD_ID = "019ef56d-9a48-7f23-8fe6-201acd4c6965"
TURN_ID = "019ef56d-9a55-7c41-9ae0-c9958f25576b"
MSG_ID = "msg_007461865ca561f0016a3abbd4c95481968e7d2baa6fdb9229"

# A schema-valid final answer, exactly as the structured probe emitted it.
FINAL_JSON_TEXT = '{"returns":{"answer":"DELTA-9082"},"todos_done":[],"finished":true}'

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "returns": {"type": "object", "properties": {"answer": {"type": "string"}}},
        "todos_done": {"type": "array", "items": {"type": "string"}},
        "finished": {"type": "boolean"},
    },
    "required": ["returns", "todos_done", "finished"],
    "additionalProperties": False,
}


class _NullExecutor:
    """No tools in M2 → invoke must never be called. Asserts that contract."""

    def __init__(self) -> None:
        self.invoked = False

    async def invoke(self, call) -> ToolResult:  # pragma: no cover - must never run
        self.invoked = True
        raise AssertionError("structured-mode backend must not invoke host tools")


def _write_fake_codex_app_server(tmp_path: Path) -> Path:
    """Write a tiny stdio app-server executable for lifecycle/env tests.

    It deliberately writes Codex-like state files under CODEX_HOME on startup. If
    the backend fails to provide CODEX_HOME, the script falls back to ~/.codex,
    making default-home pollution observable without running the real Codex CLI.
    """

    script = tmp_path / "fake_codex_app_server.py"
    script.write_text(
        f"""#!/usr/bin/env python3
import json
import os
import pathlib
import sys

THREAD_ID = {THREAD_ID!r}
TURN_ID = {TURN_ID!r}
MSG_ID = {MSG_ID!r}
FINAL_JSON_TEXT = {FINAL_JSON_TEXT!r}

home = pathlib.Path(os.environ.get("CODEX_HOME") or pathlib.Path.home() / ".codex")
home.mkdir(parents=True, exist_ok=True)
for rel in ("state_5.sqlite", "logs_2.sqlite", "installation_id"):
    (home / rel).write_text("fake-codex-state", encoding="utf-8")
(home / "skills").mkdir(exist_ok=True)

def send(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\\n")
    sys.stdout.flush()

for raw in sys.stdin:
    if not raw.strip():
        continue
    msg = json.loads(raw)
    method = msg.get("method")
    mid = msg.get("id")
    if method == "initialize":
        send({{"id": mid, "result": {{
            "userAgent": "tilldone/fake",
            "codexHome": str(home),
            "platformFamily": "unix",
            "platformOs": "linux",
        }}}})
    elif method == "initialized":
        continue
    elif method == "thread/start":
        send({{"id": mid, "result": {{
            "thread": {{"id": THREAD_ID, "sessionId": THREAD_ID, "ephemeral": True}},
            "model": "fake-model",
            "sandbox": {{"type": "dangerFullAccess"}},
            "approvalPolicy": "never",
        }}}})
        send({{"method": "thread/started", "params": {{"thread": {{"id": THREAD_ID}}}}}})
    elif method == "turn/start":
        send({{"id": mid, "result": {{"turn": {{"id": TURN_ID, "status": "inProgress"}}}}}})
        send({{"method": "turn/started", "params": {{
            "threadId": THREAD_ID,
            "turn": {{"id": TURN_ID, "items": [], "itemsView": "notLoaded",
                     "status": "inProgress", "error": None}},
        }}}})
        send({{"method": "item/started", "params": {{
            "item": {{"type": "agentMessage", "id": MSG_ID, "text": "",
                     "phase": "final_answer"}},
            "threadId": THREAD_ID,
            "turnId": TURN_ID,
        }}}})
        send({{"method": "item/completed", "params": {{
            "item": {{"type": "agentMessage", "id": MSG_ID, "text": FINAL_JSON_TEXT,
                     "phase": "final_answer"}},
            "threadId": THREAD_ID,
            "turnId": TURN_ID,
        }}}})
        send({{"method": "thread/tokenUsage/updated", "params": {{
            "threadId": THREAD_ID,
            "turnId": TURN_ID,
            "tokenUsage": {{
                "total": {{"totalTokens": 3, "inputTokens": 2, "outputTokens": 1}},
                "last": {{"totalTokens": 3}},
                "modelContextWindow": 128,
            }},
        }}}})
        send({{"method": "turn/completed", "params": {{
            "threadId": THREAD_ID,
            "turn": {{"id": TURN_ID, "items": [], "itemsView": "notLoaded",
                     "status": "completed", "error": None, "durationMs": 1}},
        }}}})
    elif mid is not None:
        send({{"id": mid, "result": {{}}}})
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


_FAKE_CODEX_STATE_MARKERS = ("state_5.sqlite", "logs_2.sqlite", "installation_id", "skills")


async def _complete_fake_app_server_run(be: CodexAppServerBackend, cwd: Path) -> Path:
    handle = await be.start(_spec(cwd), _NullExecutor())
    transport = next(iter(be._live.values()))
    assert transport._env is not None
    used_home = Path(transport._env["CODEX_HOME"])
    # Drain the fake server's completed turn so the transport is in the same
    # post-run resident state a real caller would leave for backend.aclose().
    async for _ in handle.events:
        pass
    assert (await handle.outcome()).status == "completed"
    return used_home


async def _run_fake_app_server_once(
    codex_bin: Path,
    cwd: Path,
    *,
    codex_home: Path | None = None,
) -> Path:
    be = CodexAppServerBackend(
        codex_bin=str(codex_bin), codex_home=codex_home, mode="structured")
    try:
        return await _complete_fake_app_server_run(be, cwd)
    finally:
        await be.aclose()


def _has_fake_codex_state(home: Path) -> bool:
    return any((home / name).exists() for name in _FAKE_CODEX_STATE_MARKERS)


def _spec(tmp_path: Path, **kw) -> AgentRunSpec:
    return AgentRunSpec(
        prompt=kw.pop("prompt", "Produce the structured output."),
        tools=(),
        cwd=tmp_path,
        output_schema=kw.pop("output_schema", _OUTPUT_SCHEMA),
        **kw,
    )


# --------------------------------------------------------------------------- #
# Server-side scripts (mirror the golden transcripts frame-for-frame)          #
# --------------------------------------------------------------------------- #

def _respond_initialize(peer: FakePeer, init_id):
    peer.respond(
        init_id,
        {
            "userAgent": "tilldone/test",
            "codexHome": "/tmp/x",
            "platformFamily": "unix",
            "platformOs": "linux",
        },
    )


def _respond_thread_start(peer: FakePeer, start_id, *, thread_id=THREAD_ID):
    peer.respond(
        start_id,
        {
            "thread": {"id": thread_id, "sessionId": thread_id, "ephemeral": True},
            "model": "gw/model-a",
            "sandbox": {"type": "dangerFullAccess"},
            "approvalPolicy": "never",
        },
    )
    peer.notify("thread/started", {"thread": {"id": thread_id}})


async def _serve_handshake(peer: FakePeer, *, thread_id=THREAD_ID):
    """Answer initialize → initialized → thread/start → turn/start IN ORDER.

    Crucial ordering: ``start()`` only sends ``initialized`` AFTER its ``initialize``
    request is answered, so the server must respond to ``initialize`` BEFORE reading
    ``initialized`` (otherwise deadlock). Returns the parsed (init, start, turn) msgs.
    """
    init = await peer.read_client_message()
    assert init["method"] == "initialize"
    _respond_initialize(peer, init["id"])
    initialized = await peer.read_client_message()
    assert initialized["method"] == "initialized"
    start = await peer.read_client_message()
    assert start["method"] == "thread/start"
    _respond_thread_start(peer, start["id"], thread_id=thread_id)
    turn = await peer.read_client_message()
    assert turn["method"] == "turn/start"
    peer.respond(turn["id"], {"turn": {"id": TURN_ID, "status": "inProgress"}})
    return init, start, turn


def _notify_turn_started(peer: FakePeer, turn_id=TURN_ID, thread_id=THREAD_ID):
    peer.notify(
        "turn/started",
        {
            "threadId": thread_id,
            "turn": {"id": turn_id, "items": [], "itemsView": "notLoaded",
                     "status": "inProgress", "error": None},
        },
    )


def _notify_final_agent_message(peer: FakePeer, text=FINAL_JSON_TEXT,
                                *, phase="final_answer", turn_id=TURN_ID,
                                thread_id=THREAD_ID, item_id=MSG_ID):
    peer.notify(
        "item/started",
        {"item": {"type": "agentMessage", "id": item_id, "text": "", "phase": phase},
         "threadId": thread_id, "turnId": turn_id},
    )
    peer.notify(
        "item/completed",
        {"item": {"type": "agentMessage", "id": item_id, "text": text, "phase": phase},
         "threadId": thread_id, "turnId": turn_id},
    )


def _notify_token_usage(peer: FakePeer, *, total_tokens=11655, turn_id=TURN_ID,
                        thread_id=THREAD_ID):
    peer.notify(
        "thread/tokenUsage/updated",
        {"threadId": thread_id, "turnId": turn_id,
         "tokenUsage": {
             "total": {"totalTokens": total_tokens, "inputTokens": total_tokens - 24,
                       "outputTokens": 24},
             "last": {"totalTokens": total_tokens},
             "modelContextWindow": 258400}},
    )


def _notify_turn_completed(peer: FakePeer, *, status="completed", error=None,
                           turn_id=TURN_ID, thread_id=THREAD_ID):
    turn = {"id": turn_id, "items": [], "itemsView": "notLoaded", "status": status,
            "error": error, "durationMs": 8331}
    peer.notify("turn/completed", {"threadId": thread_id, "turn": turn})


async def _drive_happy(peer: FakePeer, *, final_text=FINAL_JSON_TEXT,
                       status="completed", error=None):
    """Read the client handshake/turn requests and script a full happy server flow."""
    init, start, turn = await _serve_handshake(peer)
    _notify_turn_started(peer)
    _notify_final_agent_message(peer, text=final_text)
    _notify_token_usage(peer)
    _notify_turn_completed(peer, status=status, error=error)
    return init, start, turn


# --------------------------------------------------------------------------- #
# events.py — pure mapper unit tests                                           #
# --------------------------------------------------------------------------- #

def test_map_thread_started_to_run_started():
    ev = map_notification(
        {"method": "thread/started", "params": {"thread": {"id": THREAD_ID}}}, seq=1
    )
    assert ev is not None
    assert ev.kind == EventKind.RUN_STARTED
    assert ev.thread_id == THREAD_ID
    assert ev.seq == 1


def test_map_turn_started_is_dropped():
    assert map_notification(
        {"method": "turn/started",
         "params": {"threadId": THREAD_ID, "turn": {"id": TURN_ID}}}, seq=2
    ) is None


def test_map_agent_message_completed_to_assistant_text():
    ev = map_notification(
        {"method": "item/completed",
         "params": {"item": {"type": "agentMessage", "id": MSG_ID,
                             "text": FINAL_JSON_TEXT, "phase": "final_answer"},
                    "threadId": THREAD_ID, "turnId": TURN_ID}}, seq=3,
    )
    assert ev is not None
    assert ev.kind == EventKind.ASSISTANT_TEXT
    assert ev.data["text"] == FINAL_JSON_TEXT
    assert ev.data["phase"] == "final_answer"
    assert ev.thread_id == THREAD_ID
    assert ev.turn_id == TURN_ID
    assert ev.item_id == MSG_ID


def test_map_reasoning_completed_to_reasoning():
    ev = map_notification(
        {"method": "item/completed",
         "params": {"item": {"type": "reasoning", "id": "rs_1", "summary": [], "content": []},
                    "threadId": THREAD_ID, "turnId": TURN_ID}}, seq=4,
    )
    assert ev is not None
    assert ev.kind == EventKind.REASONING


def test_map_command_execution_started_and_completed():
    started = map_notification(
        {"method": "item/started",
         "params": {"item": {"type": "commandExecution", "id": "c1",
                             "command": "ls", "status": "inProgress"},
                    "threadId": THREAD_ID, "turnId": TURN_ID}}, seq=5,
    )
    assert started is not None and started.kind == EventKind.COMMAND
    assert started.data["phase"] == "begin"
    assert started.call_id == "c1"

    done = map_notification(
        {"method": "item/completed",
         "params": {"item": {"type": "commandExecution", "id": "c1",
                             "command": "ls", "exit_code": 0, "status": "completed"},
                    "threadId": THREAD_ID, "turnId": TURN_ID}}, seq=6,
    )
    assert done is not None and done.kind == EventKind.COMMAND
    assert done.data["phase"] == "end"


def test_map_mcp_tool_call_started_is_tool_call_completed_is_result():
    call = map_notification(
        {"method": "item/started",
         "params": {"item": {"type": "mcpToolCall", "id": "t1", "tool": "x",
                             "server": "s", "status": "inProgress"},
                    "threadId": THREAD_ID, "turnId": TURN_ID}}, seq=7,
    )
    assert call is not None and call.kind == EventKind.TOOL_CALL
    assert call.call_id == "t1"

    res = map_notification(
        {"method": "item/completed",
         "params": {"item": {"type": "mcpToolCall", "id": "t1", "tool": "x",
                             "status": "completed", "result": {}},
                    "threadId": THREAD_ID, "turnId": TURN_ID}}, seq=8,
    )
    assert res is not None and res.kind == EventKind.TOOL_RESULT
    assert res.call_id == "t1"
    assert res.data["is_error"] is False


def test_map_file_change_completed_to_file_changed():
    ev = map_notification(
        {"method": "item/completed",
         "params": {"item": {"type": "fileChange", "id": "f1"},
                    "threadId": THREAD_ID, "turnId": TURN_ID}}, seq=9,
    )
    assert ev is not None and ev.kind == EventKind.FILE_CHANGED


def test_map_agent_message_delta_to_assistant_text_streaming():
    ev = map_notification(
        {"method": "item/agentMessage/delta",
         "params": {"threadId": THREAD_ID, "turnId": TURN_ID,
                    "itemId": MSG_ID, "delta": "{\""}}, seq=10,
    )
    assert ev is not None
    assert ev.kind == EventKind.ASSISTANT_TEXT
    assert ev.data["delta"] == "{\""
    assert ev.data.get("streaming") is True
    assert ev.item_id == MSG_ID


def test_map_turn_completed_completed_to_run_completed():
    ev = map_notification(
        {"method": "turn/completed",
         "params": {"threadId": THREAD_ID,
                    "turn": {"id": TURN_ID, "status": "completed", "error": None}}}, seq=11,
    )
    assert ev is not None
    assert ev.kind == EventKind.RUN_COMPLETED
    assert ev.turn_id == TURN_ID


def test_map_turn_completed_failed_to_run_failed():
    ev = map_notification(
        {"method": "turn/completed",
         "params": {"threadId": THREAD_ID,
                    "turn": {"id": TURN_ID, "status": "failed",
                             "error": {"message": "boom"}}}}, seq=12,
    )
    assert ev is not None
    assert ev.kind == EventKind.RUN_FAILED


def test_map_turn_completed_interrupted_to_run_failed():
    ev = map_notification(
        {"method": "turn/completed",
         "params": {"threadId": THREAD_ID,
                    "turn": {"id": TURN_ID, "status": "interrupted", "error": None}}}, seq=13,
    )
    assert ev is not None
    assert ev.kind == EventKind.RUN_FAILED


def test_map_token_usage_to_usage():
    ev = map_notification(
        {"method": "thread/tokenUsage/updated",
         "params": {"threadId": THREAD_ID, "turnId": TURN_ID,
                    "tokenUsage": {"total": {"totalTokens": 100}}}}, seq=14,
    )
    assert ev is not None
    assert ev.kind == EventKind.USAGE
    assert ev.data["usage"]["total"]["totalTokens"] == 100


def test_map_noise_frames_are_dropped():
    for method in ("warning", "thread/status/changed", "account/rateLimits/updated",
                   "remoteControl/status/changed", "thread/goal/cleared"):
        assert map_notification({"method": method, "params": {}}, seq=99) is None


def test_map_user_message_is_dropped():
    """userMessage item/completed must NOT be mapped to assistant text."""
    assert map_notification(
        {"method": "item/completed",
         "params": {"item": {"type": "userMessage", "id": "u1", "content": []},
                    "threadId": THREAD_ID, "turnId": TURN_ID}}, seq=15,
    ) is None


# --------------------------------------------------------------------------- #
# backend — capabilities + name                                                #
# --------------------------------------------------------------------------- #

def test_name_is_distinct_from_codex():
    be = CodexAppServerBackend(mode="structured")
    assert be.name == "codex_app_server"


def test_capabilities_structured_no_host_tools():
    caps = CodexAppServerBackend(mode="structured").capabilities()
    assert caps.host_tool_calls is False
    assert caps.structured_final_output is True
    assert caps.session_resume is True
    assert caps.streaming_events is True
    assert caps.reasoning_events is True
    assert caps.command_events is True
    assert caps.file_change_events is True
    assert caps.multimodal_input is True
    assert caps.dynamic_tools is False


async def test_default_codex_home_is_per_run_ephemeral_and_cleaned(tmp_path, monkeypatch):
    """Default app-server runs use a project-local per-run home and clean it on close."""

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("TILLDONE_CODEX_HOME", raising=False)

    fake_codex = _write_fake_codex_app_server(tmp_path)
    be = CodexAppServerBackend(codex_bin=str(fake_codex), mode="structured")
    used_home: Path | None = None
    try:
        used_home = await _complete_fake_app_server_run(be, project)

        runs_root = (project / ".tilldone" / "codex-home" / "runs").resolve()
        assert used_home.is_relative_to(runs_root)
        assert _has_fake_codex_state(used_home)

        user_codex = fake_home / ".codex"
        assert not _has_fake_codex_state(user_codex), (
            "default app-server run wrote Codex state under user ~/.codex")

        global_tilldone_home = fake_home / ".tilldone" / "codex-home"
        assert not _has_fake_codex_state(global_tilldone_home), (
            "default app-server run wrote Codex state under global ~/.tilldone")

        project_persistent_home = project / ".tilldone" / "codex-home"
        assert not (project_persistent_home / "state_5.sqlite").exists()
        assert not (project_persistent_home / "logs_2.sqlite").exists()
        assert not (project_persistent_home / "installation_id").exists()
        assert not (project_persistent_home / "skills").exists()
    finally:
        await be.aclose()

    assert used_home is not None
    assert not used_home.exists(), "default per-run CODEX_HOME was not cleaned"


async def test_default_codex_home_is_unique_per_run_same_project(tmp_path, monkeypatch):
    """Two default app-server runs in one project receive independent CODEX_HOME dirs."""

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("TILLDONE_CODEX_HOME", raising=False)

    fake_codex = _write_fake_codex_app_server(tmp_path)
    be_a = CodexAppServerBackend(codex_bin=str(fake_codex), mode="structured")
    be_b = CodexAppServerBackend(codex_bin=str(fake_codex), mode="structured")
    used: tuple[Path, Path] | None = None
    try:
        used = await asyncio.gather(
            _complete_fake_app_server_run(be_a, project),
            _complete_fake_app_server_run(be_b, project),
        )
        home_a, home_b = used
        runs_root = (project / ".tilldone" / "codex-home" / "runs").resolve()
        assert home_a.is_relative_to(runs_root)
        assert home_b.is_relative_to(runs_root)
        assert home_a != home_b
        assert _has_fake_codex_state(home_a)
        assert _has_fake_codex_state(home_b)
    finally:
        await asyncio.gather(be_a.aclose(), be_b.aclose())

    assert used is not None
    assert not used[0].exists()
    assert not used[1].exists()


async def test_default_codex_home_gc_reaps_stale_crash_residue(tmp_path, monkeypatch):
    """Creating a default home best-effort reaps stale run-* residue from old crashes."""

    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("TILLDONE_CODEX_HOME", raising=False)

    stale = tmp_path / ".tilldone" / "codex-home" / "runs" / "run-stale"
    stale.mkdir(parents=True)
    (stale / "state_5.sqlite").write_text("stale", encoding="utf-8")
    old = time.time() - 2 * 60 * 60
    os.utime(stale / "state_5.sqlite", (old, old))
    os.utime(stale, (old, old))

    be = CodexAppServerBackend(mode="structured")
    default_home: Path | None = None
    transport = be._make_transport(_spec(tmp_path))
    try:
        assert transport._env is not None
        default_home = Path(transport._env["CODEX_HOME"])
        assert not stale.exists()
        assert default_home.exists()
    finally:
        await be.aclose()

    assert default_home is not None
    assert not default_home.exists()


async def test_default_codex_home_ignores_ambient_process_codex_home(tmp_path, monkeypatch):
    """Ambient process CODEX_HOME is not inherited unless the caller opts in explicitly."""

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    ambient_codex_home = tmp_path / "ambient-codex-home"
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("CODEX_HOME", str(ambient_codex_home))
    monkeypatch.delenv("TILLDONE_CODEX_HOME", raising=False)

    fake_codex = _write_fake_codex_app_server(tmp_path)
    used_home = await _run_fake_app_server_once(fake_codex, project)

    assert not _has_fake_codex_state(ambient_codex_home), (
        "default app-server run inherited ambient process CODEX_HOME")
    runs_root = (project / ".tilldone" / "codex-home" / "runs").resolve()
    assert used_home.is_relative_to(runs_root)
    assert not used_home.exists(), "default per-run CODEX_HOME was not cleaned"


async def test_explicit_codex_home_is_persistent_and_not_cleaned(tmp_path, monkeypatch):
    """Explicit codex_home opts in to a persistent shared home that backend does not remove."""

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    persistent_home = tmp_path / "persistent-codex-home"
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("TILLDONE_CODEX_HOME", raising=False)

    fake_codex = _write_fake_codex_app_server(tmp_path)
    used_home = await _run_fake_app_server_once(
        fake_codex, project, codex_home=persistent_home)

    assert used_home == persistent_home.resolve()
    assert _has_fake_codex_state(persistent_home)


def test_persist_or_persisted_resume_requires_explicit_persistent_home(tmp_path, monkeypatch):
    """Persisted app-server state must opt into a fixed home, not the default temp home."""

    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("TILLDONE_CODEX_HOME", raising=False)

    with pytest.raises(ValueError, match="explicit persistent CODEX_HOME"):
        CodexAppServerBackend(mode="structured", persist_thread=True)._make_transport(
            _spec(tmp_path))

    persisted_resume = ResumeHandle(
        "codex_app_server",
        {"thread_id": THREAD_ID, "sandbox": "danger-full-access", "approvalPolicy": "never"},
    )
    with pytest.raises(ValueError, match="explicit persistent CODEX_HOME"):
        CodexAppServerBackend(mode="structured")._make_transport(
            _spec(tmp_path, resume=persisted_resume))

    persistent_home = tmp_path / "persistent-home"
    transport = CodexAppServerBackend(
        mode="structured",
        persist_thread=True,
        codex_home=persistent_home,
    )._make_transport(_spec(tmp_path))
    assert transport._env is not None
    assert transport._env["CODEX_HOME"] == str(persistent_home.resolve())


async def test_codex_home_override_precedence(tmp_path, monkeypatch):
    """Explicit CODEX_HOME channels opt in above the per-run ephemeral default."""

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "ambient-home"))
    monkeypatch.delenv("TILLDONE_CODEX_HOME", raising=False)

    spec = _spec(tmp_path)
    default_backend = CodexAppServerBackend(mode="structured")
    default_transport = default_backend._make_transport(spec)
    second_default_transport = default_backend._make_transport(spec)
    assert default_transport._env is not None
    assert second_default_transport._env is not None
    runs_root = (tmp_path / ".tilldone" / "codex-home" / "runs").resolve()
    default_home = Path(default_transport._env["CODEX_HOME"])
    second_default_home = Path(second_default_transport._env["CODEX_HOME"])
    assert default_home.is_relative_to(runs_root)
    assert second_default_home.is_relative_to(runs_root)
    assert default_home != second_default_home
    await default_backend.aclose()
    assert not default_home.exists()
    assert not second_default_home.exists()

    monkeypatch.setenv("TILLDONE_CODEX_HOME", str(tmp_path / "env-home"))
    tilldone_env_transport = CodexAppServerBackend(mode="structured")._make_transport(spec)
    assert tilldone_env_transport._env is not None
    assert tilldone_env_transport._env["CODEX_HOME"] == str(tmp_path / "env-home")

    param_transport = CodexAppServerBackend(
        mode="structured",
        codex_home=tmp_path / "param-home",
    )._make_transport(spec)
    assert param_transport._env is not None
    assert param_transport._env["CODEX_HOME"] == str(tmp_path / "param-home")

    dot_codex = fake_home / ".codex"
    spec_transport = CodexAppServerBackend(
        mode="structured",
        codex_home=tmp_path / "param-home",
    )._make_transport(_spec(tmp_path, env={"CODEX_HOME": str(dot_codex)}))
    assert spec_transport._env is not None
    assert spec_transport._env["CODEX_HOME"] == str(dot_codex)


# --------------------------------------------------------------------------- #
# backend — happy structured round-trip                                        #
# --------------------------------------------------------------------------- #

async def test_structured_happy_path_events_and_outcome(tmp_path):
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        # The server driver must run CONCURRENTLY with start(): start() blocks on
        # the initialize/thread/start/turn/start handshake, which the driver answers.
        driver = asyncio.ensure_future(_drive_happy(peer))
        handle = await be.start(_spec(tmp_path), _NullExecutor())

        events = [ev async for ev in handle.events]
        await asyncio.wait_for(driver, timeout=2.0)
        outcome = await handle.outcome()

        kinds = [e.kind for e in events]
        # RUN_STARTED first, exactly one terminal last == RUN_COMPLETED.
        assert kinds[0] == EventKind.RUN_STARTED
        assert kinds[-1] == EventKind.RUN_COMPLETED
        assert kinds.count(EventKind.RUN_COMPLETED) == 1
        assert kinds.count(EventKind.RUN_FAILED) == 0
        assert EventKind.ASSISTANT_TEXT in kinds
        assert EventKind.USAGE in kinds

        # seq strictly monotonic.
        seqs = [e.seq for e in events]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)

        # outcome: parsed structured output + status + resume handle id.
        assert outcome.status == "completed"
        assert outcome.structured_output == json.loads(FINAL_JSON_TEXT)
        assert outcome.error is None
        assert outcome.final_text == FINAL_JSON_TEXT
        assert outcome.resume is not None
        assert outcome.resume.backend == "codex_app_server"
        assert outcome.resume.token["thread_id"] == THREAD_ID
        assert outcome.usage is not None
        assert outcome.usage["total"]["totalTokens"] == 11655
    finally:
        await be.aclose()
        await peer.aclose()


async def test_structured_happy_path_passes_conformance(tmp_path):
    """collect_run + assert_protocol_invariants — the cross-backend generalization gate."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        driver = asyncio.ensure_future(_drive_happy(peer))
        collector = asyncio.ensure_future(
            collect_run(be, _spec(tmp_path), _NullExecutor()))
        events, outcome = await asyncio.wait_for(collector, timeout=2.0)
        await asyncio.wait_for(driver, timeout=2.0)
        assert_protocol_invariants(events, outcome)
        assert outcome.structured_output == json.loads(FINAL_JSON_TEXT)
    finally:
        await be.aclose()
        await peer.aclose()


async def test_initialize_handshake_shape(tmp_path):
    """initialize carries experimentalApi:true; thread/start carries string sandbox +
    ephemeral + cwd; turn/start carries the OBJECT sandboxPolicy + input + outputSchema."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        # start() runs concurrently with our inline reads/asserts of the handshake.
        start_task = asyncio.ensure_future(be.start(
            _spec(tmp_path, system_instructions="SYS", developer_instructions="DEV",
                  model="gpt-5.5"),
            _NullExecutor(),
        ))

        init = await peer.read_client_message()
        assert init["params"]["capabilities"]["experimentalApi"] is True
        assert init["params"]["clientInfo"]["name"] == "tilldone"
        # Respond to initialize BEFORE reading initialized (start() gates on it).
        _respond_initialize(peer, init["id"])

        initialized = await peer.read_client_message()
        assert initialized["method"] == "initialized"

        start = await peer.read_client_message()
        sp = start["params"]
        assert sp["sandbox"] == "danger-full-access"  # STRING form on thread/start
        assert sp["approvalPolicy"] == "never"
        assert sp["ephemeral"] is True
        assert sp["cwd"] == str(tmp_path)
        assert sp["baseInstructions"] == "SYS"
        assert sp["developerInstructions"] == "DEV"
        assert sp["model"] == "gpt-5.5"
        assert "dynamicTools" not in sp  # M2: NO tools wired

        _respond_thread_start(peer, start["id"])

        turn = await peer.read_client_message()
        tp = turn["params"]
        assert tp["threadId"] == THREAD_ID
        assert tp["input"] == [{"type": "text", "text": "Produce the structured output."}]
        assert tp["outputSchema"] == _OUTPUT_SCHEMA
        assert tp["sandboxPolicy"] == {"type": "dangerFullAccess"}  # OBJECT form on turn/start
        peer.respond(turn["id"], {"turn": {"id": TURN_ID, "status": "inProgress"}})

        handle = await asyncio.wait_for(start_task, timeout=2.0)
        consume = asyncio.ensure_future(_collect(handle))
        # finish the run so the consumer task ends cleanly.
        _notify_turn_started(peer)
        _notify_final_agent_message(peer)
        _notify_turn_completed(peer)
        await asyncio.wait_for(consume, timeout=2.0)
    finally:
        await be.aclose()
        await peer.aclose()


async def _collect(handle):
    return [ev async for ev in handle.events]


async def test_no_server_request_handler_registered_unexpected_is_protocol_error(tmp_path):
    """Structured path registers NO server-request handler; an item/tool/call arriving
    lands in transport.unexpected_server_requests → outcome is a protocol error."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(
            be.start(_spec(tmp_path), _NullExecutor()))

        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            # An UNEXPECTED server-request (no handler in structured mode).
            peer.server_request("item/tool/call", {"callId": "c0", "name": "x"}, id=0)
            await asyncio.sleep(0.02)
            _notify_final_agent_message(peer)
            _notify_turn_completed(peer, status="completed")

        driver = asyncio.ensure_future(drive())
        handle = await asyncio.wait_for(start_task, timeout=2.0)
        events = [ev async for ev in handle.events]
        await asyncio.wait_for(driver, timeout=2.0)
        outcome = await handle.outcome()

        assert outcome.status == "failed"
        assert outcome.error is not None
        assert outcome.error.code == "protocol"
        # terminal still present + single
        assert sum(1 for e in events if e.kind in
                   (EventKind.RUN_COMPLETED, EventKind.RUN_FAILED)) == 1
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# backend — failure / interrupt / non-JSON                                     #
# --------------------------------------------------------------------------- #

async def test_turn_failed_yields_run_failed_and_backend_error(tmp_path):
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(
            be.start(_spec(tmp_path), _NullExecutor()))

        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            _notify_turn_completed(
                peer, status="failed",
                error={"message": "stream disconnected before completion",
                       "codexErrorInfo": "serverOverloaded"})

        driver = asyncio.ensure_future(drive())
        handle = await asyncio.wait_for(start_task, timeout=2.0)
        events = [ev async for ev in handle.events]
        await asyncio.wait_for(driver, timeout=2.0)
        outcome = await handle.outcome()

        assert events[-1].kind == EventKind.RUN_FAILED
        assert outcome.status == "failed"
        assert outcome.error is not None
        assert outcome.error.code == "transient"  # serverOverloaded => transient
        assert "stream disconnected" in outcome.error.message
        assert outcome.structured_output is None
    finally:
        await be.aclose()
        await peer.aclose()


async def test_turn_failed_unauthorized_is_fatal(tmp_path):
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(
            be.start(_spec(tmp_path), _NullExecutor()))

        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            _notify_turn_completed(
                peer, status="failed",
                error={"message": "401", "codexErrorInfo": "unauthorized"})

        driver = asyncio.ensure_future(drive())
        handle = await asyncio.wait_for(start_task, timeout=2.0)
        events = [ev async for ev in handle.events]
        await asyncio.wait_for(driver, timeout=2.0)
        outcome = await handle.outcome()
        assert outcome.error is not None
        assert outcome.error.code == "auth"
    finally:
        await be.aclose()
        await peer.aclose()


async def test_turn_error_codexErrorInfo_object_httpstatus(tmp_path):
    """codexErrorInfo as an object with a 5xx httpStatusCode => transient."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(
            be.start(_spec(tmp_path), _NullExecutor()))

        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            _notify_turn_completed(
                peer, status="failed",
                error={"message": "upstream", "codexErrorInfo":
                       {"variant": {"httpStatusCode": 503}}})

        driver = asyncio.ensure_future(drive())
        handle = await asyncio.wait_for(start_task, timeout=2.0)
        _ = [ev async for ev in handle.events]
        await asyncio.wait_for(driver, timeout=2.0)
        outcome = await handle.outcome()
        assert outcome.error is not None
        assert outcome.error.code == "transient"
        assert outcome.error.status == 503
    finally:
        await be.aclose()
        await peer.aclose()


async def test_turn_interrupted_yields_run_failed(tmp_path):
    """interrupt => status=interrupted, error=null; RUN_FAILED terminal."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(
            be.start(_spec(tmp_path), _NullExecutor()))

        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            _notify_turn_completed(peer, status="interrupted", error=None)

        driver = asyncio.ensure_future(drive())
        handle = await asyncio.wait_for(start_task, timeout=2.0)
        events = [ev async for ev in handle.events]
        await asyncio.wait_for(driver, timeout=2.0)
        outcome = await handle.outcome()

        assert events[-1].kind == EventKind.RUN_FAILED
        assert outcome.status == "failed"
        assert outcome.error is not None
        assert outcome.error.code in ("fatal", "protocol")
    finally:
        await be.aclose()
        await peer.aclose()


async def test_non_json_final_message_is_protocol_error(tmp_path):
    """A schema-illegal (non-JSON) final agentMessage => protocol error, NEVER a
    natural-language guess."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        driver = asyncio.ensure_future(
            _drive_happy(peer, final_text="Sorry, I cannot do that."))
        handle = await be.start(_spec(tmp_path), _NullExecutor())

        events = [ev async for ev in handle.events]
        await asyncio.wait_for(driver, timeout=2.0)
        outcome = await handle.outcome()

        # turn.status was completed, but parse failure flips status to failed.
        assert outcome.status == "failed"
        assert outcome.error is not None
        assert outcome.error.code == "protocol"
        assert outcome.structured_output is None
        # terminal stays single & last (the event stream reflects turn/completed).
        assert sum(1 for e in events if e.kind in
                   (EventKind.RUN_COMPLETED, EventKind.RUN_FAILED)) == 1
    finally:
        await be.aclose()
        await peer.aclose()


async def test_buffers_final_answer_not_trailing_non_final(tmp_path):
    """Prefer phase==final_answer: a later non-final agentMessage must NOT clobber the
    buffered final JSON used for structured_output."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(
            be.start(_spec(tmp_path), _NullExecutor()))

        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            _notify_final_agent_message(peer, text=FINAL_JSON_TEXT, phase="final_answer",
                                        item_id="m_final")
            # A trailing NON-final narrative agentMessage (must be ignored for output).
            _notify_final_agent_message(peer, text="just chatting", phase="answer",
                                        item_id="m_extra")
            _notify_turn_completed(peer, status="completed")

        driver = asyncio.ensure_future(drive())
        handle = await asyncio.wait_for(start_task, timeout=2.0)
        _ = [ev async for ev in handle.events]
        await asyncio.wait_for(driver, timeout=2.0)
        outcome = await handle.outcome()
        assert outcome.structured_output == json.loads(FINAL_JSON_TEXT)
        assert outcome.final_text == FINAL_JSON_TEXT
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# backend — resume (cross-backend fail-fast + same-backend issues thread/resume)#
# --------------------------------------------------------------------------- #

async def test_resume_foreign_backend_id_raises(tmp_path):
    """A resume handle minted by another backend must fail-fast (RT-SESS-5 mirror)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        spec = _spec(tmp_path, resume=ResumeHandle("codex", {"session_id": "abc"}))
        with pytest.raises(ValueError, match="codex_app_server"):
            await be.start(spec, _NullExecutor())
    finally:
        await be.aclose()
        await peer.aclose()


async def test_resume_same_backend_reuses_resident_thread_no_thread_resume(tmp_path):
    """A resume handle for THIS backend pointing at a RESIDENT thread reuses the
    live transport: it issues another ``turn/start`` on the same thread and NEVER a
    ``thread/resume`` ( — an ephemeral thread has no rollout
    to resume). Full multi-round reuse / fail-fast coverage lives in
    ``test_backend_multiround.py``; this asserts the LaneA-level contract that the
    same-backend resume path does NOT emit ``thread/resume``."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        # Round 1 — open a real resident thread via the normal handshake.
        driver = asyncio.ensure_future(_serve_handshake(peer))
        handle1 = await be.start(_spec(tmp_path), _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)
        consume1 = asyncio.ensure_future(_collect(handle1))
        _notify_turn_started(peer)
        _notify_final_agent_message(peer)
        _notify_turn_completed(peer)
        await asyncio.wait_for(consume1, timeout=2.0)
        outcome1 = await handle1.outcome()
        assert outcome1.resume is not None
        assert outcome1.resume.token["thread_id"] == THREAD_ID

        # Round 2 — resume against the resident thread. The ONLY client message is a
        # turn/start on the same thread (no initialize / thread/start / thread/resume).
        spec2 = _spec(tmp_path, resume=outcome1.resume)
        start_task = asyncio.ensure_future(be.start(spec2, _NullExecutor()))
        msg = await peer.read_client_message()
        assert msg["method"] == "turn/start", (
            f"resident resume must reuse via turn/start, not {msg['method']!r}")
        assert msg["params"]["threadId"] == THREAD_ID
        peer.respond(msg["id"], {"turn": {"id": TURN_ID, "status": "inProgress"}})

        handle2 = await asyncio.wait_for(start_task, timeout=2.0)
        consume2 = asyncio.ensure_future(_collect(handle2))
        _notify_turn_started(peer)
        _notify_final_agent_message(peer)
        _notify_turn_completed(peer)
        await asyncio.wait_for(consume2, timeout=2.0)

        outcome2 = await handle2.outcome()
        assert outcome2.status == "completed"
        assert outcome2.resume.token["thread_id"] == THREAD_ID
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# backend — cancel issues turn/interrupt                                       #
# --------------------------------------------------------------------------- #

async def test_cancel_issues_turn_interrupt(tmp_path):
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(be.start(_spec(tmp_path), _NullExecutor()))

        await _serve_handshake(peer)
        handle = await asyncio.wait_for(start_task, timeout=2.0)
        _notify_turn_started(peer)
        # Give the events loop a tick to observe turn/started so turn_id is known.
        events_task = asyncio.ensure_future(_collect(handle))
        await asyncio.sleep(0.02)

        await handle.cancel()
        interrupt = await peer.read_client_message()
        assert interrupt["method"] == "turn/interrupt"
        assert interrupt["params"]["threadId"] == THREAD_ID
        assert interrupt["params"]["turnId"] == TURN_ID
        # cancel also closes the transport → events iterator terminates.
        await asyncio.wait_for(events_task, timeout=2.0)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 0-core-diff boundary: backend/events do NOT cause core→app_server import      #
# --------------------------------------------------------------------------- #

def test_core_does_not_import_app_server():
    """Importing tilldone.core (+ submodules) must not pull app_server in."""
    import importlib

    for name in list(__import__("sys").modules):
        if name.startswith("tilldone"):
            del __import__("sys").modules[name]
    importlib.import_module("tilldone.core.backend")
    importlib.import_module("tilldone.core.spec")
    importlib.import_module("tilldone.core.events")
    leaked = [n for n in __import__("sys").modules
              if n.startswith("tilldone.backends.app_server")]
    assert leaked == [], f"core import leaked app_server modules: {leaked}"
