"""RT-CAP-1: capability is runtime-probed; a down bridge fails fast (never hangs)."""

from __future__ import annotations

import asyncio

import pytest

from tilldone.backends.codex_exec import CodexExecBackend
from tilldone.core.spec import AgentRunSpec
from tilldone.core.tools import ToolCall, ToolResult


class _StubExecutor:
    async def invoke(self, call: ToolCall) -> ToolResult:
        return ToolResult(content="ok")


class _FailingBridge:
    """A bridge whose start() never binds — simulates a killed/unavailable MCP bridge."""

    def __init__(self, tools, executor):
        ...

    async def start(self):
        raise OSError("bridge cannot bind")

    async def stop(self):
        ...


async def test_probe_host_tools_true_with_real_bridge():
    # Spins a REAL localhost MCP bridge (no codex/model) and tears it down.
    assert await CodexExecBackend().probe_host_tools() is True


async def test_probe_host_tools_false_when_bridge_down():
    backend = CodexExecBackend(bridge_factory=_FailingBridge)
    # bounded: must resolve to False, not hang
    assert await asyncio.wait_for(backend.probe_host_tools(), timeout=5) is False


async def test_start_fails_fast_when_bridge_cannot_bind(tmp_path):
    backend = CodexExecBackend(bridge_factory=_FailingBridge)
    spec = AgentRunSpec(prompt="go", tools=(), cwd=tmp_path)
    with pytest.raises(OSError):
        # fail-fast (raises) within the timeout — RT-CAP-1: does NOT hang
        await asyncio.wait_for(backend.start(spec, _StubExecutor()), timeout=5)
