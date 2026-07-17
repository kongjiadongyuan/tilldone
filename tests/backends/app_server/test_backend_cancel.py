"""cancel() correctness for CodexAppServerBackend (M4 LaneC).

Closes the M2 small-debt and proves RT-APP-CANCEL's three states.

The debt: ``RunHandle.cancel`` issued ``turn/interrupt`` as a NOTIFICATION. Live
the schema (``TurnInterruptParams``/``TurnInterruptResponse``)
confirm ``turn/interrupt`` is a **ClientRequest** (params ``{threadId,turnId}``,
result ``{}``) and the turn's terminal flips to ``status=interrupted``. So cancel
must SEND A REQUEST — but it must NEVER block on a dead/slow peer, hence a SHORT
timeout + ``contextlib.suppress`` (timeout / JsonRpcError / TransportClosed are all
fine; ``aclose`` is the unconditional backstop).

RT-APP-CANCEL three states — cancel must be correct + non-hanging in ALL:
  1. **turnId unknown** (turn/start hadn't yielded ``turn.id``) → skip the interrupt
     request (no turnId to target), go straight to aclose. No hang, no error.
  2. **tool call pending** (an ``item/tool/call`` handler is mid-flight) → the
     interrupt request + aclose must not deadlock; aclose fails pending request
     futures and cancels in-flight handler tasks. No orphaned pending server-request.
  3. **turn already completed** → interrupt is a harmless no-op / idempotent; cancel
     must not raise even if the peer errors the interrupt (e.g. -32600 unknown turn).

In all states: exactly one terminal on the event stream, pending cleaned, no hang,
and cancel is idempotent (a second cancel is a no-op).

All tests drive a *real* :class:`AppServerTransport` through the in-memory
``FakePeer`` (NO codex spawn, NO gateway). asyncio_mode="auto" → no decorator.
"""

from __future__ import annotations

import asyncio

import pytest

from tilldone.backends.app_server.backend import CodexAppServerBackend
from tilldone.core.enums import EventKind
from tilldone.core.spec import AgentRunSpec
from tilldone.core.tools import ToolCall, ToolResult

from .fake_peer import FakePeer
from .test_backend import (
    THREAD_ID,
    TURN_ID,
    _notify_turn_completed,
    _notify_turn_started,
    _respond_initialize,
    _respond_thread_start,
    _serve_handshake,
)

# A bound (seconds) within which cancel MUST return even when the peer never
# answers the interrupt — comfortably below any real request timeout, proving the
# short-timeout + suppress (not a 120s transport default) is what unblocks cancel.
_CANCEL_BOUND_S = 2.0


class _NullExecutor:
    """No host tools needed for most cancel tests (structured mode)."""

    async def invoke(self, call) -> ToolResult:  # pragma: no cover - never called here
        raise AssertionError("executor must not be invoked in this test")


class _BlockingToolExecutor:
    """A host executor whose ``invoke`` blocks forever (until cancelled).

    Models State 2: an ``item/tool/call`` handler that is mid-flight when cancel
    fires. The backend dispatches the call to this executor; ``invoke`` parks on an
    event that is never set, so the handler task is genuinely in-flight when
    ``aclose`` cancels it. ``entered`` lets the test await until the handler is
    actually running before firing cancel (no racy sleep).
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = False

    async def invoke(self, call: ToolCall) -> ToolResult:
        self.entered.set()
        try:
            await asyncio.Event().wait()  # park forever
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return ToolResult(content="unreachable")  # pragma: no cover


def _spec(tmp_path, **kw) -> AgentRunSpec:
    return AgentRunSpec(
        prompt=kw.pop("prompt", "Do the thing."),
        tools=kw.pop("tools", ()),
        cwd=tmp_path,
        output_schema=kw.pop("output_schema", None),
        **kw,
    )


async def _collect(handle):
    return [ev async for ev in handle.events]


async def _serve_handshake_no_turn_id(peer: FakePeer):
    """Handshake whose ``turn/start`` response carries NO ``turn.id``.

    Mirrors ``_serve_handshake`` but the turn/start result omits ``turn.id``, so the
    handle's ``_turn_id`` stays ``None`` (State 1: turnId unknown). The model never
    progressed far enough to mint a turn id.
    """
    init = await peer.read_client_message()
    assert init["method"] == "initialize"
    _respond_initialize(peer, init["id"])
    initialized = await peer.read_client_message()
    assert initialized["method"] == "initialized"
    start = await peer.read_client_message()
    assert start["method"] == "thread/start"
    _respond_thread_start(peer, start["id"])
    turn = await peer.read_client_message()
    assert turn["method"] == "turn/start"
    # No turn.id in the response → handle never learns its turnId.
    peer.respond(turn["id"], {"turn": {"status": "inProgress"}})
    return init, start, turn


# --------------------------------------------------------------------------- #
# The core debt fix: turn/interrupt is a REQUEST (id-bearing), not a notify     #
# --------------------------------------------------------------------------- #

async def test_cancel_sends_turn_interrupt_as_request(tmp_path):
    """cancel() issues ``turn/interrupt`` as a JSON-RPC REQUEST: the frame carries an
    ``id`` (id+method ⇒ request) and ``params={threadId,turnId}``. The peer answers
    ``{}`` (TurnInterruptResponse); cancel returns after aclose closes the transport
    (so the events iterator terminates)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(be.start(_spec(tmp_path), _NullExecutor()))
        await _serve_handshake(peer)
        handle = await asyncio.wait_for(start_task, timeout=2.0)
        _notify_turn_started(peer)
        events_task = asyncio.ensure_future(_collect(handle))
        await asyncio.sleep(0.02)  # let the events loop observe turn/started

        # Answer the interrupt as a real response, concurrently with cancel.
        async def answer_interrupt():
            msg = await peer.read_client_message()
            # THE debt fix: id present (REQUEST), not a bare notification.
            assert "id" in msg, f"turn/interrupt must be a REQUEST (id-bearing), got {msg!r}"
            assert msg["method"] == "turn/interrupt"
            assert msg["params"] == {"threadId": THREAD_ID, "turnId": TURN_ID}
            peer.respond(msg["id"], {})  # TurnInterruptResponse == {}
            return msg

        answerer = asyncio.ensure_future(answer_interrupt())
        await asyncio.wait_for(handle.cancel(), timeout=_CANCEL_BOUND_S)
        await asyncio.wait_for(answerer, timeout=2.0)
        # aclose ran → the events iterator terminates.
        await asyncio.wait_for(events_task, timeout=2.0)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# State 1 — turnId unknown: skip the interrupt request, aclose anyway           #
# --------------------------------------------------------------------------- #

async def test_state1_turn_id_unknown_skips_interrupt_and_acloses(tmp_path):
    """State 1: turn/start never yielded a turn.id, so the handle has no turnId to
    target → cancel sends NO turn/interrupt request and goes straight to aclose.
    No hang, no error, transport closed."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(be.start(_spec(tmp_path), _NullExecutor()))
        await _serve_handshake_no_turn_id(peer)
        handle = await asyncio.wait_for(start_task, timeout=2.0)
        assert handle._turn_id is None  # precondition for State 1

        events_task = asyncio.ensure_future(_collect(handle))
        await asyncio.sleep(0.01)
        await asyncio.wait_for(handle.cancel(), timeout=_CANCEL_BOUND_S)

        # NO interrupt was sent: the only thing the peer can read now is EOF (the
        # client side closed on aclose), never a turn/interrupt frame.
        with pytest.raises((EOFError, asyncio.TimeoutError)):
            await peer.read_client_message(timeout=0.3)
        # aclose still happened → transport closed, events iterator terminated.
        assert be._make_transport  # sanity: backend intact
        await asyncio.wait_for(events_task, timeout=2.0)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# State 2 — tool call pending: no deadlock, pending cleaned, prompt return      #
# --------------------------------------------------------------------------- #

async def test_state2_tool_call_pending_no_deadlock(tmp_path):
    """State 2: an ``item/tool/call`` handler is mid-flight (parked forever) when
    cancel fires. cancel's interrupt request + aclose must NOT deadlock: aclose
    cancels the in-flight handler task and fails pending request futures, so cancel
    returns promptly. No orphaned pending server-request."""
    peer = FakePeer()
    ex = _BlockingToolExecutor()
    # Tools mode so the backend registers the item/tool/call handler.
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="tools")
    try:
        start_task = asyncio.ensure_future(be.start(_spec(tmp_path), ex))
        await _serve_handshake(peer)
        handle = await asyncio.wait_for(start_task, timeout=2.0)
        _notify_turn_started(peer)
        events_task = asyncio.ensure_future(_collect(handle))

        # Fire a server→client item/tool/call: the backend dispatches it to the
        # blocking executor, whose handler parks (genuinely in-flight).
        peer.server_request(
            "item/tool/call",
            {"threadId": THREAD_ID, "turnId": TURN_ID, "callId": "c0",
             "namespace": "contract", "tool": "return", "arguments": {"answer": "x"}},
            id=0,
        )
        # Wait until the handler is actually running before cancelling (no race).
        await asyncio.wait_for(ex.entered.wait(), timeout=2.0)

        # cancel must return promptly even with a tool handler parked + an
        # unanswered interrupt request (peer ignores the interrupt entirely).
        await asyncio.wait_for(handle.cancel(), timeout=_CANCEL_BOUND_S)

        # The parked handler task was cancelled by aclose (no orphan).
        assert ex.cancelled is True
        # No pending request future left hanging on the transport.
        assert handle._t._pending == {}
        await asyncio.wait_for(events_task, timeout=2.0)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# State 3 — turn already completed: interrupt suppressed even if peer errors     #
# --------------------------------------------------------------------------- #

async def test_state3_turn_already_completed_interrupt_suppressed(tmp_path):
    """State 3: the turn already reached turn/completed; cancel still fires (the
    caller doesn't know the turn finished) but the interrupt is a harmless no-op.
    Even if the peer ERRORS it (-32600 unknown turn), cancel must not raise — the
    error is suppressed and aclose still runs."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(be.start(_spec(tmp_path), _NullExecutor()))
        await _serve_handshake(peer)
        handle = await asyncio.wait_for(start_task, timeout=2.0)
        _notify_turn_started(peer)
        _notify_turn_completed(peer, status="completed")
        # Drain the stream to its single terminal (turn/completed consumed).
        events = await asyncio.wait_for(_collect(handle), timeout=2.0)
        assert events[-1].kind == EventKind.RUN_COMPLETED

        # The peer ERRORS the interrupt (turn already gone) — must be suppressed.
        async def error_interrupt():
            msg = await peer.read_client_message()
            assert "id" in msg and msg["method"] == "turn/interrupt"
            peer.respond_error(msg["id"], -32600, "no such turn (already completed)")

        errer = asyncio.ensure_future(error_interrupt())
        # cancel must NOT raise despite the JsonRpcError on the interrupt.
        await asyncio.wait_for(handle.cancel(), timeout=_CANCEL_BOUND_S)
        await asyncio.wait_for(errer, timeout=2.0)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# No-hang: a peer that NEVER answers the interrupt must not block cancel         #
# --------------------------------------------------------------------------- #

async def test_cancel_does_not_hang_when_peer_ignores_interrupt(tmp_path):
    """The interrupt request must NOT make cancel hang if the peer never responds:
    a SHORT timeout + suppress unblocks it, and aclose is the backstop. We assert
    cancel returns within a tight bound even though the peer ignores the interrupt
    entirely (never calls respond)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(be.start(_spec(tmp_path), _NullExecutor()))
        await _serve_handshake(peer)
        handle = await asyncio.wait_for(start_task, timeout=2.0)
        _notify_turn_started(peer)
        events_task = asyncio.ensure_future(_collect(handle))
        await asyncio.sleep(0.02)

        # Peer NEVER answers the interrupt. cancel must still return within bound.
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await asyncio.wait_for(handle.cancel(), timeout=_CANCEL_BOUND_S)
        elapsed = loop.time() - t0
        assert elapsed < _CANCEL_BOUND_S, f"cancel hung {elapsed:.2f}s on an unanswered interrupt"
        await asyncio.wait_for(events_task, timeout=2.0)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# Idempotency — a second cancel is a no-op                                      #
# --------------------------------------------------------------------------- #

async def test_cancel_is_idempotent(tmp_path):
    """A second cancel() is a harmless no-op: the transport is already closed, so the
    interrupt request can't be sent (suppressed) and aclose short-circuits. Neither
    call raises."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(be.start(_spec(tmp_path), _NullExecutor()))
        await _serve_handshake(peer)
        handle = await asyncio.wait_for(start_task, timeout=2.0)
        _notify_turn_started(peer)
        events_task = asyncio.ensure_future(_collect(handle))
        await asyncio.sleep(0.02)

        # First cancel: peer ignores the interrupt; short timeout unblocks it.
        await asyncio.wait_for(handle.cancel(), timeout=_CANCEL_BOUND_S)
        assert handle._t._closed is True
        # Second cancel: transport already closed → pure no-op, must not raise.
        await asyncio.wait_for(handle.cancel(), timeout=_CANCEL_BOUND_S)
        await asyncio.wait_for(events_task, timeout=2.0)
    finally:
        await be.aclose()
        await peer.aclose()
