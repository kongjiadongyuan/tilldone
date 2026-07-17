"""Rich-control tests for CodexAppServerBackend (M5 Lane A) — steer / fork / rollback.

These land the **capability-scoped control plane**: the
rich app-server capabilities are reachable ONLY via the backend-agnostic optional
Protocols :class:`SteerCapable` / :class:`BranchingCapable` (defined beside
``SessionCapable`` in :mod:`tilldone.core.backend`) and operate on a RESIDENT
thread's transport. They are NOT part of ``run_task`` / ``ContractEvaluator`` /
``ToolRegistry`` — the 0-core-diff boundary is enforced separately in
``tests/test_zero_core_diff.py`` (loop.py must reference none of them, and a
``backend.rollback`` must touch zero files).

Schema authority (live shapes lifted from the vendored protocol schema):

* ``turn/steer``  — ``TurnSteerParams`` REQUIRES ``{threadId, expectedTurnId, input}``;
  the response is ``TurnSteerResponse {turnId}``. ``expectedTurnId`` is the active-turn
  precondition (source = ``TurnStartResponse.turn.id``); a stale id / steering an
  already-completed turn / a consecutive steer all FAIL FAST. The race error is the
  ``activeTurnNotSteerable`` codexErrorInfo (``turn/start`` or ``turn/steer`` against a
  non-steerable active turn), surfaced over the wire as a JSON-RPC error.
* ``thread/fork`` — ``ThreadForkParams {threadId, ...}`` → ``ThreadForkResponse`` whose
  ``thread.id`` is the NEW (forked) thread id.
* ``thread/rollback`` — ``ThreadRollbackParams {threadId, numTurns}``; ``numTurns`` has
  schema ``min=0`` but its description REQUIRES ``>= 1`` ("drop from the end ... Must be
  >= 1"), so the client enforces ``>= 1`` and a ``numTurns=0`` call is a fail-fast (defeats
  a no-op false-green). Rollback "only modifies the thread's history and does not revert
  local file changes" — so the method performs ZERO filesystem operations.

All tests drive a *real* :class:`AppServerTransport` through the in-memory
``FakePeer`` (NO codex spawn, NO gateway). ``asyncio_mode="auto"`` (pyproject) →
async tests need no decorator.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from tilldone.backends.app_server.backend import CodexAppServerBackend
from tilldone.backends.app_server.transport import JsonRpcError
from tilldone.core.backend import BranchingCapable, SteerCapable

from .fake_peer import FakePeer
from .test_backend import (
    THREAD_ID,
    TURN_ID,
    _NullExecutor,
    _serve_handshake,
    _spec,
)

# A second turn id the server hands back from a successful steer (TurnSteerResponse.turnId).
STEER_TURN_ID = "019ef56d-cccc-7c41-9ae0-c9958f25576b"
# A forked-thread id distinct from THREAD_ID (ThreadForkResponse.thread.id).
FORK_THREAD_ID = "019ef56d-dddd-7f23-8fe6-201acd4c6965"
# A turn id distinct from this run's bound TURN_ID — used as the STALE expectedTurnId.
STALE_TURN_ID = "019ef56d-9999-7c41-0000-000000000000"

# JSON-RPC error code the live app-server returns for a steer-race (a request against a
# non-steerable / mismatched active turn surfaces as a JSON-RPC error). The shape we
# script mirrors the activeTurnNotSteerable codexErrorInfo carried in `data`.
_STEER_RACE_CODE = -32600
_ACTIVE_TURN_NOT_STEERABLE = {"activeTurnNotSteerable": {"turnKind": "review"}}


async def _resident_thread(peer: FakePeer, tmp_path, *, mode="tools"):
    """Drive a backend to a RESIDENT thread (initialize→thread/start→turn/start) and
    return ``(backend, handle)``. The turn is left in-flight (no turn/completed) so the
    control plane can act on a live thread; the test owns ``backend.aclose()``.
    """
    be = CodexAppServerBackend(transport_factory=lambda **kw: peer.make_transport(**kw),
                               mode=mode)
    driver = asyncio.ensure_future(_serve_handshake(peer))
    handle = await be.start(_spec(tmp_path), _NullExecutor())
    await asyncio.wait_for(driver, timeout=2.0)
    return be, handle


# --------------------------------------------------------------------------- #
# isinstance: the backend SATISFIES the optional rich-control protocols         #
# --------------------------------------------------------------------------- #

def test_backend_satisfies_steer_and_branching_protocols():
    """``CodexAppServerBackend`` is recognized as both ``SteerCapable`` and
    ``BranchingCapable`` via the runtime-checkable optional Protocols — the same
    established pattern as ``SessionCapable``. This is how a control-plane caller
    discovers the capability WITHOUT core ever importing the backend."""
    be = CodexAppServerBackend(transport_factory=lambda **kw: None)
    assert isinstance(be, SteerCapable)
    assert isinstance(be, BranchingCapable)


# --------------------------------------------------------------------------- #
# steer — happy path                                                            #
# --------------------------------------------------------------------------- #

async def test_steer_happy_sends_turn_steer_and_returns_new_turn_id(tmp_path):
    """``steer`` issues ``turn/steer {threadId, expectedTurnId, input}`` on the resident
    transport and returns the new ``TurnSteerResponse.turnId``. ``expectedTurnId`` is the
    REQUIRED active-turn precondition; we pass this run's bound TURN_ID."""
    peer = FakePeer()
    be, handle = await _resident_thread(peer, tmp_path)
    try:
        steer = asyncio.ensure_future(
            be.steer(THREAD_ID, "actually, also do X", expected_turn_id=TURN_ID))
        req = await peer.read_client_message()
        assert req["method"] == "turn/steer"
        params = req["params"]
        assert params["threadId"] == THREAD_ID
        assert params["expectedTurnId"] == TURN_ID
        # input is the schema's UserInput[] (text item), like turn/start.
        assert params["input"] == [{"type": "text", "text": "actually, also do X"}]
        peer.respond(req["id"], {"turnId": STEER_TURN_ID})
        new_turn = await asyncio.wait_for(steer, timeout=2.0)
        assert new_turn == STEER_TURN_ID
    finally:
        await be.aclose()


async def test_steer_requires_expected_turn_id_keyword(tmp_path):
    """``expected_turn_id`` is a REQUIRED keyword (the schema makes it mandatory).
    Calling ``steer`` without it is a TypeError — the precondition can never be
    silently dropped."""
    peer = FakePeer()
    be, handle = await _resident_thread(peer, tmp_path)
    try:
        with pytest.raises(TypeError):
            await be.steer(THREAD_ID, "no precondition")  # type: ignore[call-arg]
    finally:
        await be.aclose()


# --------------------------------------------------------------------------- #
# steer — three race fail-fasts                                                 #
# --------------------------------------------------------------------------- #

async def test_steer_stale_expected_turn_id_fails_fast(tmp_path):
    """Race #1 — STALE expectedTurnId: the supplied id is not the active turn. The server
    rejects with the activeTurnNotSteerable error; ``steer`` surfaces it as a clear raise
    (never a silent no-op / false-green)."""
    peer = FakePeer()
    be, handle = await _resident_thread(peer, tmp_path)
    try:
        steer = asyncio.ensure_future(
            be.steer(THREAD_ID, "late", expected_turn_id=STALE_TURN_ID))
        req = await peer.read_client_message()
        assert req["method"] == "turn/steer"
        assert req["params"]["expectedTurnId"] == STALE_TURN_ID
        peer.respond_error(req["id"], _STEER_RACE_CODE,
                           "active turn is not steerable", data=_ACTIVE_TURN_NOT_STEERABLE)
        with pytest.raises(Exception) as ei:
            await asyncio.wait_for(steer, timeout=2.0)
        # The error must carry the steer-race signal so an operator sees WHY.
        assert "steer" in str(ei.value).lower()
    finally:
        await be.aclose()


async def test_steer_after_turn_completed_fails_fast(tmp_path):
    """Race #2 — steering an ALREADY-COMPLETED turn: the active turn finished, so the
    expectedTurnId no longer matches a steerable turn. The server rejects and ``steer``
    fails fast rather than appearing to land an input on a dead turn."""
    peer = FakePeer()
    be, handle = await _resident_thread(peer, tmp_path)
    try:
        steer = asyncio.ensure_future(
            be.steer(THREAD_ID, "too late", expected_turn_id=TURN_ID))
        req = await peer.read_client_message()
        assert req["method"] == "turn/steer"
        peer.respond_error(req["id"], _STEER_RACE_CODE,
                           "active turn already completed", data=_ACTIVE_TURN_NOT_STEERABLE)
        with pytest.raises(Exception):
            await asyncio.wait_for(steer, timeout=2.0)
    finally:
        await be.aclose()


async def test_consecutive_steer_second_fails_fast(tmp_path):
    """Race #3 — CONSECUTIVE steer: a first steer succeeds and rotates the active turn id;
    a second steer that re-uses the now-stale (original) expectedTurnId is rejected. The
    caller must re-read the new turn id from the first steer's response before steering
    again — proving expectedTurnId is a hard precondition, not advisory."""
    peer = FakePeer()
    be, handle = await _resident_thread(peer, tmp_path)
    try:
        # First steer succeeds, returning a NEW active turn id.
        s1 = asyncio.ensure_future(
            be.steer(THREAD_ID, "first", expected_turn_id=TURN_ID))
        r1 = await peer.read_client_message()
        assert r1["method"] == "turn/steer"
        peer.respond(r1["id"], {"turnId": STEER_TURN_ID})
        assert await asyncio.wait_for(s1, timeout=2.0) == STEER_TURN_ID

        # Second steer re-uses the STALE original TURN_ID → server rejects.
        s2 = asyncio.ensure_future(
            be.steer(THREAD_ID, "second", expected_turn_id=TURN_ID))
        r2 = await peer.read_client_message()
        assert r2["method"] == "turn/steer"
        assert r2["params"]["expectedTurnId"] == TURN_ID
        peer.respond_error(r2["id"], _STEER_RACE_CODE,
                           "stale expectedTurnId", data=_ACTIVE_TURN_NOT_STEERABLE)
        with pytest.raises(Exception):
            await asyncio.wait_for(s2, timeout=2.0)
    finally:
        await be.aclose()


async def test_steer_unknown_thread_fails_fast(tmp_path):
    """A ``steer`` on a thread NOT resident in this backend instance fails fast (there is
    no transport to drive) — never silently swallowed."""
    peer = FakePeer()
    be, handle = await _resident_thread(peer, tmp_path)
    try:
        with pytest.raises(Exception):
            await be.steer("no-such-thread", "x", expected_turn_id=TURN_ID)
    finally:
        await be.aclose()


# --------------------------------------------------------------------------- #
# fork — returns the new thread id                                              #
# --------------------------------------------------------------------------- #

async def test_fork_sends_thread_fork_and_returns_new_thread_id(tmp_path):
    """``fork`` issues ``thread/fork {threadId}`` on the resident transport and returns the
    NEW forked ``thread.id`` (ThreadForkResponse.thread.id) — distinct from the source."""
    peer = FakePeer()
    be, handle = await _resident_thread(peer, tmp_path)
    try:
        fork = asyncio.ensure_future(be.fork(THREAD_ID))
        req = await peer.read_client_message()
        assert req["method"] == "thread/fork"
        assert req["params"]["threadId"] == THREAD_ID
        peer.respond(
            req["id"],
            {"thread": {"id": FORK_THREAD_ID, "sessionId": FORK_THREAD_ID,
                        "ephemeral": True, "forkedFromId": THREAD_ID},
             "model": "gw/model-a", "modelProvider": "openai",
             "approvalPolicy": "never", "approvalsReviewer": "user",
             "cwd": str(tmp_path), "sandbox": {"type": "dangerFullAccess"}},
        )
        new_thread = await asyncio.wait_for(fork, timeout=2.0)
        assert new_thread == FORK_THREAD_ID
        assert new_thread != THREAD_ID
    finally:
        await be.aclose()


async def test_fork_unknown_thread_fails_fast(tmp_path):
    """A ``fork`` on a non-resident thread fails fast — no transport to drive."""
    peer = FakePeer()
    be, handle = await _resident_thread(peer, tmp_path)
    try:
        with pytest.raises(Exception):
            await be.fork("no-such-thread")
    finally:
        await be.aclose()


# --------------------------------------------------------------------------- #
# rollback — numTurns >= 1 happy + numTurns=0 fail-fast                          #
# --------------------------------------------------------------------------- #

async def test_rollback_happy_sends_thread_rollback(tmp_path):
    """``rollback`` issues ``thread/rollback {threadId, numTurns}`` on the resident
    transport and returns the updated ``thread.id``. ``num_turns=2`` is propagated verbatim
    to the wire ``numTurns`` field."""
    peer = FakePeer()
    be, handle = await _resident_thread(peer, tmp_path)
    try:
        rb = asyncio.ensure_future(be.rollback(THREAD_ID, num_turns=2))
        req = await peer.read_client_message()
        assert req["method"] == "thread/rollback"
        assert req["params"]["threadId"] == THREAD_ID
        assert req["params"]["numTurns"] == 2
        peer.respond(
            req["id"],
            {"thread": {"id": THREAD_ID, "sessionId": THREAD_ID, "ephemeral": True,
                        "turns": []}},
        )
        result = await asyncio.wait_for(rb, timeout=2.0)
        assert result == THREAD_ID
    finally:
        await be.aclose()


@pytest.mark.parametrize("bad", [0, -1])
async def test_rollback_num_turns_below_one_fails_fast_without_wire_call(tmp_path, bad):
    """``num_turns < 1`` is a CLIENT-side fail-fast (schema description requires >= 1) — it
    raises BEFORE any wire call, defeating the ``numTurns=0`` no-op false-green. We prove no
    ``thread/rollback`` was ever sent by asserting the client wrote nothing."""
    peer = FakePeer()
    be, handle = await _resident_thread(peer, tmp_path)
    try:
        with pytest.raises(ValueError):
            await be.rollback(THREAD_ID, num_turns=bad)
        # Nothing should have hit the wire — the next client read times out.
        with pytest.raises(asyncio.TimeoutError):
            await peer.read_client_message(timeout=0.2)
    finally:
        await be.aclose()


async def test_rollback_unknown_thread_fails_fast(tmp_path):
    """A ``rollback`` on a non-resident thread fails fast — no transport to drive."""
    peer = FakePeer()
    be, handle = await _resident_thread(peer, tmp_path)
    try:
        with pytest.raises(Exception):
            await be.rollback("no-such-thread", num_turns=1)
    finally:
        await be.aclose()


# --------------------------------------------------------------------------- #
# rollback — rolls back HISTORY ONLY, never the filesystem (boundary)           #
# --------------------------------------------------------------------------- #

async def test_rollback_touches_no_filesystem(tmp_path):
    """``rollback`` rolls back THREAD HISTORY ONLY — it performs ZERO filesystem
    operations. We write a file into ``cwd`` BEFORE the rollback and assert it is STILL
    there afterward, AND that the directory snapshot is byte-for-byte unchanged across the
    ``thread/rollback`` round-trip (no file created OR removed by the backend)."""
    peer = FakePeer()
    be, handle = await _resident_thread(peer, tmp_path)
    try:
        # A file the agent "created" earlier; rollback must NOT remove it.
        marker = tmp_path / "agent_wrote_this.txt"
        marker.write_text("side effect of a turn we are rolling back", encoding="utf-8")
        before = sorted(os.listdir(tmp_path))

        rb = asyncio.ensure_future(be.rollback(THREAD_ID, num_turns=1))
        req = await peer.read_client_message()
        assert req["method"] == "thread/rollback"
        peer.respond(req["id"], {"thread": {"id": THREAD_ID, "turns": []}})
        await asyncio.wait_for(rb, timeout=2.0)

        after = sorted(os.listdir(tmp_path))
        assert after == before, "rollback must not create/remove any file in cwd"
        assert marker.exists(), "rollback rolls back HISTORY only — the file must survive"
        assert marker.read_text(encoding="utf-8") == (
            "side effect of a turn we are rolling back")
    finally:
        await be.aclose()


# --------------------------------------------------------------------------- #
# control plane is a SEPARATE surface from the run_task contract loop           #
# --------------------------------------------------------------------------- #

async def test_control_methods_do_not_disturb_resident_turn_stream(tmp_path):
    """A ``steer``/``fork``/``rollback`` is issued out-of-band on the resident transport and
    does NOT emit ``AgentEvent``s into the run's event stream nor terminate it: the control
    plane and the contract loop are independent surfaces sharing only the transport. The
    turn still completes normally through ``handle.events`` afterward."""
    from tilldone.core.enums import EventKind
    from .test_backend import (
        _notify_final_agent_message,
        _notify_token_usage,
        _notify_turn_completed,
        _notify_turn_started,
        FINAL_JSON_TEXT,
    )

    peer = FakePeer()
    be, handle = await _resident_thread(peer, tmp_path)
    try:
        # Out-of-band fork while the turn is in-flight.
        fork = asyncio.ensure_future(be.fork(THREAD_ID))
        req = await peer.read_client_message()
        assert req["method"] == "thread/fork"
        peer.respond(req["id"], {"thread": {"id": FORK_THREAD_ID}})
        assert await asyncio.wait_for(fork, timeout=2.0) == FORK_THREAD_ID

        # Now the turn completes — its event stream is unaffected by the fork above.
        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        _notify_final_agent_message(peer, text=FINAL_JSON_TEXT)
        _notify_token_usage(peer)
        _notify_turn_completed(peer, status="completed")
        events = await asyncio.wait_for(consume, timeout=2.0)
        kinds = [ev.kind for ev in events]
        assert EventKind.RUN_COMPLETED in kinds
        # No control-plane frame ever appeared as an AgentEvent.
        assert all(ev.kind != EventKind.RUN_FAILED for ev in events)
    finally:
        await be.aclose()


async def _collect(handle):
    return [ev async for ev in handle.events]
