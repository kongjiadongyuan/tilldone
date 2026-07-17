"""RT-SESS-5 — cross-backend resume is a v1 NON-GOAL; both backends must fail fast.

A ResumeHandle minted by one backend (its ``.backend`` field) MUST NOT be silently
fed to the other backend as an opaque string. ``start()`` raises ValueError BEFORE any
subprocess/bridge/SDK work — these tests are gateway-free (no live, no real subprocess):
the guard trips before any I/O.

The symmetric assertions:
  (a) ClaudeBackend.start(resume.backend="codex")  -> ValueError mentioning RT-SESS-5
  (b) CodexExecBackend.start(resume.backend="claude") -> ValueError mentioning RT-SESS-5
  (c) A SAME-backend handle does NOT trip THIS guard — it proceeds past it (and only
      then may fail for unrelated, clearly-labelled reasons).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tilldone.backends.claude import ClaudeBackend
from tilldone.backends.codex_exec import CodexExecBackend
from tilldone.core.spec import AgentRunSpec, ResumeHandle
from tilldone.core.tools import ToolCall, ToolResult


class _StubExecutor:
    async def invoke(self, call: ToolCall) -> ToolResult:  # pragma: no cover - never reached
        return ToolResult(content="ok", is_error=False)


def _spec(resume_backend: str, cwd: Path) -> AgentRunSpec:
    """Minimal valid spec carrying a resume handle minted by `resume_backend`."""
    return AgentRunSpec(
        prompt="go",
        tools=(),
        cwd=cwd,
        resume=ResumeHandle(resume_backend, {"session_id": "sess-foreign-1"}),
    )


# ---------------- (a) Claude rejects a Codex-minted handle ----------------

async def test_claude_rejects_codex_resume_handle(tmp_path):
    backend = ClaudeBackend()  # real query_fn — but the guard trips before it's ever called
    with pytest.raises(ValueError) as exc:
        await backend.start(_spec("codex", tmp_path), _StubExecutor())
    msg = str(exc.value)
    assert "RT-SESS-5" in msg
    assert "codex" in msg and "claude" in msg  # names the wrong backend AND the right one


# ---------------- (b) Codex rejects a Claude-minted handle ----------------

async def test_codex_rejects_claude_resume_handle(tmp_path):
    backend = CodexExecBackend()  # guard trips before any bridge bind / subprocess
    with pytest.raises(ValueError) as exc:
        await backend.start(_spec("claude", tmp_path), _StubExecutor())
    msg = str(exc.value)
    assert "RT-SESS-5" in msg
    assert "claude" in msg and "codex" in msg


# ---------------- (c) same-backend handle does NOT trip THIS guard ----------------

class _Sentinel(Exception):
    """A unique, clearly-unrelated failure used to prove we got PAST the RT-SESS-5 guard."""


def _exploding_query(*, prompt, options):  # matches sdk.query signature shape
    raise _Sentinel("past the guard (claude)")


class _ExplodingBridge:
    def __init__(self, *_args, **_kwargs):
        ...

    async def start(self):
        raise _Sentinel("past the guard (codex)")

    async def stop(self):  # pragma: no cover - start() raised before a stop is needed
        ...


async def test_claude_same_backend_handle_passes_guard(tmp_path):
    # A claude-minted handle must NOT raise the cross-backend ValueError. We inject a
    # query_fn that raises a sentinel so start() fails AFTER the guard for an unrelated
    # reason — proving the guard itself did not trip on a same-backend handle.
    backend = ClaudeBackend(query_fn=_exploding_query)
    with pytest.raises(_Sentinel):
        await backend.start(_spec("claude", tmp_path), _StubExecutor())


async def test_codex_same_backend_handle_passes_guard(tmp_path):
    # A codex-minted handle must NOT raise the cross-backend ValueError. The injected
    # bridge_factory raises a sentinel at bridge.start() — i.e. AFTER the guard — proving
    # the guard let the same-backend handle through.
    backend = CodexExecBackend(bridge_factory=_ExplodingBridge)
    with pytest.raises(_Sentinel):
        await backend.start(_spec("codex", tmp_path), _StubExecutor())


async def test_no_resume_does_not_trip_guard_claude(tmp_path):
    # Sanity: with no resume handle at all, the guard is a no-op (fails later via sentinel).
    backend = ClaudeBackend(query_fn=_exploding_query)
    spec = AgentRunSpec(prompt="go", tools=(), cwd=tmp_path)
    with pytest.raises(_Sentinel):
        await backend.start(spec, _StubExecutor())
