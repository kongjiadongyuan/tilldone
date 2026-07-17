"""AR-3 — cancel/interrupt cleanup unit tests for CodexRunHandle.cancel().

Verifies RT-MCP-1 / G2: on cancel, the child process is killed and the MCP
bridge (localhost port) is released. No orphan process, no leaked port.
"""

from __future__ import annotations

from tilldone.backends.codex_exec import CodexRunHandle


# --------------------------------------------------------------------------- #
# Fakes — own copies (NOT imported from test_codex_unit)                      #
# --------------------------------------------------------------------------- #

class _FakeProc:
    """Minimal fake asyncio subprocess for cancel-path testing."""

    def __init__(self, *, kill_raises: bool = False):
        self.killed: bool = False
        self.returncode: int | None = None
        self._kill_raises = kill_raises

    def kill(self) -> None:
        if self._kill_raises:
            raise ProcessLookupError("already dead")
        self.killed = True

    async def wait(self) -> int:
        return 0


class _FakeBridge:
    """Minimal fake MCP bridge for cancel-path testing."""

    def __init__(self):
        self.stopped: bool = False

    async def stop(self) -> None:
        self.stopped = True


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

async def test_cancel_kills_process_and_stops_bridge():
    """cancel() must kill the process AND stop the bridge (G2 / RT-MCP-1)."""
    proc = _FakeProc()
    bridge = _FakeBridge()
    handle = CodexRunHandle(proc, bridge)

    await handle.cancel()

    assert proc.killed is True, "process was not killed"
    assert bridge.stopped is True, "bridge was not stopped"


async def test_cancel_is_safe_when_process_already_exited():
    """cancel() must not raise when .kill() raises ProcessLookupError (process already dead),
    AND must still call bridge.stop() — no port leak."""
    proc = _FakeProc(kill_raises=True)
    bridge = _FakeBridge()
    handle = CodexRunHandle(proc, bridge)

    # Must not propagate ProcessLookupError
    await handle.cancel()

    assert bridge.stopped is True, "bridge was not stopped even though process was already dead"


async def test_cancel_then_no_hang():
    """Calling cancel() twice is safe (idempotent-ish) — no exception raised."""
    proc = _FakeProc()
    bridge = _FakeBridge()
    handle = CodexRunHandle(proc, bridge)

    await handle.cancel()
    # Second call: kill() on an already-killed fake is fine (killed=True again);
    # bridge.stop() is called again — that's acceptable for idempotency.
    await handle.cancel()  # must not raise
