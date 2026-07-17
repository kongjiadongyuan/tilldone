"""RT-SDK-1 — the Flow SDK drives ``CodexAppServerBackend``.

``run()`` may pass while ``run_iter()`` does not truly STREAM, and a concurrent run
may corrupt the Flow's active-run state. Unit tests prove this for ``FakeBackend``
(``tests/core/test_flow.py``); this lands the SAME guarantee for
``CodexAppServerBackend(mode="tools")`` driven end-to-end through the real
``run_task`` contract loop via :class:`Flow`:

* ``Flow.run`` over the app-server backend reaches a contract-satisfying verdict and
  returns the typed return value.
* ``Flow.run_iter`` STREAMS intermediate :class:`AgentEvent`s (not just a terminal) and
  ``.result()`` resolves to the SAME value ``run()`` produced.
* the active-run guard holds — a second run while one is in flight raises
  :class:`ActiveRunError`, and the guard is released after completion.

The backend runs over the in-memory :class:`FakePeer` (NO codex spawn, NO gateway): a
``_peer_server`` coroutine serves the handshake, then drives a return+finish contract by
issuing the ``item/tool/call`` server-requests core's executor answers, then completes the
turn. So this is the FULL ``Flow → run_task → backend.start → item/tool/call → ToolRegistry
→ ContractState`` path on the app-server backend — the tools-mode (host_tool_calls=True)
seam, exactly as the three-backend parity uses it.

``asyncio_mode="auto"`` (pyproject) → async tests need no decorator.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path

import pytest

from tilldone.backends.app_server.backend import CodexAppServerBackend
from tilldone.core.contract import (
    CompletionContract,
    ReturnExpectation,
)
from tilldone.core.enums import EventKind
from tilldone.core.flow import ActiveRunError, Flow

from .fake_peer import FakePeer
from .test_backend import (
    THREAD_ID,
    TURN_ID,
    _respond_initialize,
    _respond_thread_start,
    _write_fake_codex_app_server,
)
from .test_backend_tools import (
    _notify_narration,
    _notify_tool_call_completed,
    _notify_tool_call_started,
    _notify_turn_completed,
    _notify_turn_started,
)

# The value the contract's `return answer` must carry (a fixed scalar, asserted equal
# across run() and run_iter().result()).
_ANSWER = 7


def _contract() -> CompletionContract:
    """A minimal tools-mode contract: one required int return + a finish gate."""
    return CompletionContract(
        outputs=(),
        returns=(ReturnExpectation("answer", "int", True),),
        todos=(),
        require_finish=True,
    )


async def _peer_server(peer: FakePeer, *, answer: int = _ANSWER) -> None:
    """Serve ONE full tools-mode contract turn over the fake peer.

    handshake (initialize → initialized → thread/start[dynamicTools] → turn/start),
    then drive the contract: issue ``item/tool/call`` for ``return {answer}`` then
    ``finish`` (core's executor answers each → mutates ContractState), then narrate +
    ``turn/completed``. This is what makes ``run_task`` judge the contract satisfied and
    return ``answer``.
    """
    # -- handshake (tools mode sends dynamicTools on thread/start) --------------
    init = await peer.read_client_message()
    assert init["method"] == "initialize"
    _respond_initialize(peer, init["id"])
    initialized = await peer.read_client_message()
    assert initialized["method"] == "initialized"
    start = await peer.read_client_message()
    assert start["method"] == "thread/start"
    assert "dynamicTools" in start["params"]  # tools mode wired the contract tools
    _respond_thread_start(peer, start["id"])
    turn = await peer.read_client_message()
    assert turn["method"] == "turn/start"
    peer.respond(turn["id"], {"turn": {"id": TURN_ID, "status": "inProgress"}})

    # -- drive the contract via item/tool/call server-requests ------------------
    _notify_turn_started(peer)

    async def _one_tool_call(server_id: int, call_id: str, tool: str, arguments: dict):
        _notify_tool_call_started(peer, call_id=call_id, tool=tool, arguments=arguments)
        peer.server_request(
            "item/tool/call",
            {"threadId": THREAD_ID, "turnId": TURN_ID, "callId": call_id,
             "namespace": "contract", "tool": tool, "arguments": arguments},
            id=server_id)
        reply = await peer.read_client_message()  # the DynamicToolCallResponse
        _notify_tool_call_completed(peer, call_id=call_id, tool=tool, arguments=arguments)
        return reply

    await _one_tool_call(1, "call_return", "return", {"answer": answer})
    await _one_tool_call(2, "call_finish", "finish", {})

    _notify_narration(peer, text="Done.")
    _notify_turn_completed(peer, status="completed")


def _flow(peer: FakePeer) -> Flow:
    """A Flow bound to a tools-mode app-server backend wired to ``peer``."""
    backend = CodexAppServerBackend(transport_factory=peer.make_transport, mode="tools")
    return Flow(backend)


class _SwitchboardFactory:
    """A ``transport_factory`` that gives each ``backend.start()`` a FRESH FakePeer and
    auto-drives it with a full contract turn.

    Separate ``Flow.run`` / ``run_iter`` calls are independent ``run_task``s with no
    resume → each opens a fresh thread, so each needs its own connection. This factory
    builds a new ``FakePeer`` per ``start()`` and launches a ``_peer_server`` for it, so
    two sequential runs never share pipes. :meth:`aclose` reaps every peer + server."""

    def __init__(self) -> None:
        self._peers: list[FakePeer] = []
        self._servers: list[asyncio.Task] = []

    def __call__(self, **kwargs):
        peer = FakePeer()
        transport = peer.make_transport(**kwargs)
        self._peers.append(peer)
        self._servers.append(asyncio.ensure_future(_peer_server(peer)))
        return transport

    async def aclose(self) -> None:
        for srv in self._servers:
            srv.cancel()
        for srv in self._servers:
            try:
                await srv  # may be cancelled, already-done, or already-raised
            except BaseException:
                pass
        for peer in self._peers:
            await peer.aclose()


def _process_tree(root_pid: int) -> set[int]:
    out = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    children: dict[int, list[int]] = {}
    seen_in_table: set[int] = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        pid, ppid = int(parts[0]), int(parts[1])
        seen_in_table.add(pid)
        children.setdefault(ppid, []).append(pid)

    tree: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid not in seen_in_table or pid in tree:
            continue
        tree.add(pid)
        stack.extend(children.get(pid, ()))
    return tree


def _live_pids(pids: set[int]) -> set[int]:
    return {pid for pid in pids if Path(f"/proc/{pid}").exists()}


def _wait_reaped(pids: set[int], *, timeout_s: float = 3.0) -> set[int]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        live = _live_pids(pids)
        if not live:
            return set()
        time.sleep(0.05)
    return _live_pids(pids)


# --------------------------------------------------------------------------- #
# 1. Flow.run drives the app-server backend to a contract-satisfying result     #
# --------------------------------------------------------------------------- #

async def test_flow_run_drives_app_server_to_typed_result(tmp_path):
    """Flow.run → run_task → CodexAppServerBackend(tools): the contract is satisfied
    by the item/tool/call dispatches and run() returns the scalar int return value."""
    peer = FakePeer()
    flow = _flow(peer)
    server = asyncio.ensure_future(_peer_server(peer))
    try:
        result = await asyncio.wait_for(
            flow.run(_contract(), prompt="return answer=7 then finish", cwd=tmp_path),
            timeout=5.0)
        await asyncio.wait_for(server, timeout=2.0)
    finally:
        await flow._backend.aclose()
        await peer.aclose()

    # Single declared return → scalar (int), NOT a dict and NOT the string "7".
    assert result == _ANSWER
    assert type(result) is int


async def test_flow_async_context_manager_reaps_app_server_child_process(tmp_path, monkeypatch):
    """Leaving ``async with Flow(...)`` closes resident app-server transports."""

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("TILLDONE_CODEX_HOME", raising=False)

    fake_codex = _write_fake_codex_app_server(tmp_path)
    backend = CodexAppServerBackend(codex_bin=str(fake_codex), mode="structured")
    contract = CompletionContract(
        outputs=(),
        returns=(ReturnExpectation("answer", "str", True),),
        todos=(),
        require_finish=False,
    )

    tracked: set[int] = set()
    async with Flow(backend) as flow:
        result = await asyncio.wait_for(
            flow.run(contract, prompt="return the answer", cwd=tmp_path),
            timeout=5.0,
        )
        transports = list(backend._live.values())
        assert len(transports) == 1
        proc = transports[0]._proc
        assert proc is not None and proc.returncode is None
        tracked = _process_tree(proc.pid)
        assert proc.pid in tracked

    assert result == "DELTA-9082"
    assert not _wait_reaped(tracked), (
        "app-server child process(es) survived Flow.__aexit__ backend.aclose()")


# --------------------------------------------------------------------------- #
# 2. Flow.run_iter STREAMS intermediate events; .result() == run()'s value      #
# --------------------------------------------------------------------------- #

async def test_flow_run_iter_streams_events_and_result_matches(tmp_path):
    """run_iter yields intermediate AgentEvents (a real stream, incl. a tool-call/result
    pair and the RUN_COMPLETED terminal), and .result() resolves to the same int run()
    would return — the RT-SDK-1 run/run_iter parity on the app-server backend."""
    peer = FakePeer()
    flow = _flow(peer)
    server = asyncio.ensure_future(_peer_server(peer))
    try:
        it = flow.run_iter(_contract(), prompt="return answer=7 then finish", cwd=tmp_path)
        events = []
        async for ev in it:
            events.append(ev)
        result = await asyncio.wait_for(it.result(), timeout=5.0)
        await asyncio.wait_for(server, timeout=2.0)
    finally:
        await flow._backend.aclose()
        await peer.aclose()

    assert result == _ANSWER
    # A real stream: at least one tool-call and one tool-result flowed (not just a
    # terminal), and exactly one terminal arrived last.
    kinds = [e.kind for e in events]
    assert EventKind.TOOL_CALL in kinds, f"no streamed TOOL_CALL in {kinds}"
    assert EventKind.TOOL_RESULT in kinds, f"no streamed TOOL_RESULT in {kinds}"
    assert any(k == EventKind.RUN_COMPLETED for k in kinds), f"no RUN_COMPLETED in {kinds}"


async def test_flow_run_and_run_iter_agree_on_value(tmp_path):
    """run() and run_iter().result() on the app-server backend produce the SAME value
    (two independent Flows + peers, same contract)."""
    peer_a = FakePeer()
    flow_a = _flow(peer_a)
    srv_a = asyncio.ensure_future(_peer_server(peer_a))
    try:
        run_result = await asyncio.wait_for(
            flow_a.run(_contract(), prompt="go", cwd=tmp_path), timeout=5.0)
        await asyncio.wait_for(srv_a, timeout=2.0)
    finally:
        await flow_a._backend.aclose()
        await peer_a.aclose()

    peer_b = FakePeer()
    flow_b = _flow(peer_b)
    srv_b = asyncio.ensure_future(_peer_server(peer_b))
    try:
        it = flow_b.run_iter(_contract(), prompt="go", cwd=tmp_path)
        async for _ in it:
            pass
        iter_result = await asyncio.wait_for(it.result(), timeout=5.0)
        await asyncio.wait_for(srv_b, timeout=2.0)
    finally:
        await flow_b._backend.aclose()
        await peer_b.aclose()

    assert run_result == iter_result == _ANSWER


# --------------------------------------------------------------------------- #
# 3. active-run guard holds on the app-server-backed Flow                        #
# --------------------------------------------------------------------------- #

async def test_active_run_guard_second_run_raises_on_app_server_flow(tmp_path):
    """Starting a second run while run_iter holds the Flow busy raises ActiveRunError —
    the RT-SDK-1 concurrent-run guard, on the app-server backend."""
    peer = FakePeer()
    flow = _flow(peer)
    server = asyncio.ensure_future(_peer_server(peer))
    try:
        # run_iter acquires the guard synchronously (before any await).
        it = flow.run_iter(_contract(), prompt="first", cwd=tmp_path)
        # A second run() must raise immediately while the iter is in flight.
        with pytest.raises(ActiveRunError):
            await flow.run(_contract(), prompt="second", cwd=tmp_path)
        # Drain the iter so the background run_task completes cleanly.
        async for _ in it:
            pass
        await asyncio.wait_for(it.result(), timeout=5.0)
        await asyncio.wait_for(server, timeout=2.0)
    finally:
        await flow._backend.aclose()
        await peer.aclose()


async def test_active_run_guard_released_after_completion(tmp_path):
    """After a run_iter is fully consumed, the Flow guard is RELEASED — a SECOND run on
    the SAME Flow proceeds (does not raise :class:`ActiveRunError`) and completes.

    A :class:`_SwitchboardFactory` hands each ``backend.start()`` its OWN fresh FakePeer
    (each ``Flow.run``/``run_iter`` is an independent ``run_task`` with no resume → a fresh
    thread, so a fresh connection), and a per-peer ``_peer_server`` drives each to a
    completed contract. This proves the guard reset WITHOUT the two runs colliding on one
    peer's pipes (mirrors v1 ``test_guard_released_after_run_iter_completes``)."""
    factory = _SwitchboardFactory()
    backend = CodexAppServerBackend(transport_factory=factory, mode="tools")
    flow = Flow(backend)
    try:
        # Round 1 via run_iter.
        it = flow.run_iter(_contract(), prompt="first", cwd=tmp_path)
        async for _ in it:
            pass
        r1 = await asyncio.wait_for(it.result(), timeout=5.0)
        assert r1 == _ANSWER

        # Guard released → a second run() must NOT raise ActiveRunError, and completes.
        try:
            r2 = await asyncio.wait_for(
                flow.run(_contract(), prompt="second", cwd=tmp_path), timeout=5.0)
        except ActiveRunError:
            pytest.fail("ActiveRunError after a completed run_iter — guard was NOT released")
        assert r2 == _ANSWER
    finally:
        await backend.aclose()
        await factory.aclose()
