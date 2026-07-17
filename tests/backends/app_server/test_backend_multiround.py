"""Resident-transport lifecycle tests for CodexAppServerBackend (M2 Lane D).

These tests land the ruling that live multi-round runs force:

* **Within-run multi-round** (``run_task`` feeds ``outcome.resume`` back into
  ``backend.start()`` every round) must KEEP the thread + connection RESIDENT and
  issue another ``turn/start`` per round — NOT a native ``thread/resume`` (an
  ``ephemeral:true`` thread has no persisted rollout, so ``thread/resume`` would
  ``-32600 "no rollout found"``), and NOT a fresh spawn (the persistent app-server
  child would leak).
* ``backend.aclose()`` must truly close every resident transport (fixing the
  happy-path child leak reaped by hand in early live runs).
* A resume handle whose ``thread_id`` is NOT resident in this backend instance
  (cross-``run_task`` / different instance / already-closed) must **fail-fast**
  with a clear error — never attempt ``thread/resume`` on the ephemeral thread.

The cross-turn notification-queue reality is also covered: the resident transport
exposes a single shared ``notifications()`` queue; round 1's handle returns after
its ``turn/completed`` and may leave trailing telemetry in the queue; round 2's
handle reads the same queue and MUST filter by its own turnId so round-1 leftovers
never pollute round 2's event stream.

All tests drive a *real* :class:`AppServerTransport` through the in-memory
``FakePeer`` (NO codex spawn, NO gateway). ``asyncio_mode="auto"`` (pyproject) →
async tests need no decorator.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tilldone.backends.app_server.backend import CodexAppServerBackend
from tilldone.core.enums import EventKind
from tilldone.core.spec import ResumeHandle
from tilldone.testing.conformance import assert_protocol_invariants

from .fake_peer import FakePeer
from .test_backend import (
    FINAL_JSON_TEXT,
    THREAD_ID,
    TURN_ID,
    _NullExecutor,
    _notify_final_agent_message,
    _notify_token_usage,
    _notify_turn_completed,
    _notify_turn_started,
    _respond_initialize,
    _respond_thread_start,
    _serve_handshake,
    _spec,
)

# A second turn id for round 2 of a multi-round run (same thread, new turn).
TURN_ID_2 = "019ef56d-bbbb-7c41-9ae0-c9958f25576b"
# A second valid JSON payload so round 2's output is distinguishable from round 1.
FINAL_JSON_TEXT_2 = '{"returns":{"answer":"ROUND-TWO-7777"},"todos_done":[],"finished":true}'


class _CountingFactory:
    """Wrap a FakePeer's ``make_transport`` and count how many transports it built.

    The multi-round REUSE path must build exactly ONE transport across both rounds
    (round 2 reuses the resident one). A fresh spawn would increment this past 1.
    """

    def __init__(self, peer: FakePeer) -> None:
        self._peer = peer
        self.built = 0

    def __call__(self, **kwargs):
        self.built += 1
        return self._peer.make_transport(**kwargs)


async def _collect(handle):
    return [ev async for ev in handle.events]


# --------------------------------------------------------------------------- #
# 1. Multi-round REUSE: round 2 reuses the resident transport — only a second   #
#    turn/start, NO second thread/start / spawn / thread/resume.                #
# --------------------------------------------------------------------------- #

async def test_multiround_reuses_resident_transport_only_second_turn_start(tmp_path):
    """Round 1 = full handshake (initialize→thread/start→turn/start→turn/completed).
    Round 2 = ``start(spec_with_resume)`` where ``resume.token.thread_id`` is the
    resident one → the fake peer observes ONLY a second ``turn/start`` on the SAME
    thread: no second initialize, no second thread/start, no thread/resume, and no
    new transport was built."""
    peer = FakePeer()
    factory = _CountingFactory(peer)
    be = CodexAppServerBackend(transport_factory=factory, mode="structured")
    try:
        # --- round 1: full handshake + a complete turn ---------------------
        driver1 = asyncio.ensure_future(_serve_handshake(peer))
        handle1 = await be.start(_spec(tmp_path), _NullExecutor())
        await asyncio.wait_for(driver1, timeout=2.0)

        consume1 = asyncio.ensure_future(_collect(handle1))
        _notify_turn_started(peer)
        _notify_final_agent_message(peer, text=FINAL_JSON_TEXT)
        _notify_token_usage(peer)
        _notify_turn_completed(peer, status="completed")
        events1 = await asyncio.wait_for(consume1, timeout=2.0)
        outcome1 = await handle1.outcome()

        assert outcome1.status == "completed"
        assert outcome1.structured_output == json.loads(FINAL_JSON_TEXT)
        assert outcome1.resume is not None
        assert outcome1.resume.backend == "codex_app_server"
        assert outcome1.resume.token["thread_id"] == THREAD_ID
        assert factory.built == 1

        # --- round 2: resume with the resident thread_id (the multi-round path)
        resume_spec = _spec(tmp_path, resume=outcome1.resume,
                            prompt="continue: missing X")
        start2 = asyncio.ensure_future(be.start(resume_spec, _NullExecutor()))

        # The ONLY client message round 2 may send before its turn/completed is a
        # turn/start on the SAME thread — NOT initialize / thread/start / thread/resume.
        msg = await peer.read_client_message(timeout=2.0)
        assert msg["method"] == "turn/start", (
            f"round-2 reuse must issue turn/start directly, got {msg['method']!r}")
        assert msg["params"]["threadId"] == THREAD_ID
        assert msg["params"]["input"] == [
            {"type": "text", "text": "continue: missing X"}]
        peer.respond(msg["id"], {"turn": {"id": TURN_ID_2, "status": "inProgress"}})

        handle2 = await asyncio.wait_for(start2, timeout=2.0)
        # No new transport was built for round 2 (resident reuse).
        assert factory.built == 1, "round-2 reuse must NOT build a new transport"

        consume2 = asyncio.ensure_future(_collect(handle2))
        _notify_turn_started(peer, turn_id=TURN_ID_2)
        _notify_final_agent_message(peer, text=FINAL_JSON_TEXT_2, turn_id=TURN_ID_2,
                                    item_id="msg_round2")
        _notify_token_usage(peer, turn_id=TURN_ID_2)
        _notify_turn_completed(peer, status="completed", turn_id=TURN_ID_2)
        events2 = await asyncio.wait_for(consume2, timeout=2.0)
        outcome2 = await handle2.outcome()

        assert outcome2.status == "completed"
        assert outcome2.structured_output == json.loads(FINAL_JSON_TEXT_2)
        # Round 2's terminal is single & last.
        assert events2[-1].kind == EventKind.RUN_COMPLETED
        assert sum(1 for e in events2 if e.kind in
                   (EventKind.RUN_COMPLETED, EventKind.RUN_FAILED)) == 1
        # Round 2's stream still satisfies the cross-backend generalization gate
        # (no RUN_STARTED is fine on a reused thread — invariants don't require it).
        assert_protocol_invariants(events2, outcome2)
    finally:
        await be.aclose()
        await peer.aclose()


async def test_multiround_round2_events_not_polluted_by_round1_trailing(tmp_path):
    """Cross-turn queue bleed: round 1 leaves TRAILING telemetry (a late
    token-usage update tagged with round-1's turnId) in the shared notifications
    queue AFTER its turn/completed. Round 2's handle reads the same queue and must
    SKIP those round-1 leftovers (turnId filtering) — round 2's event stream must
    contain only round-2 turnId events plus thread-level RUN_STARTED-style frames."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        # round 1 handshake + turn.
        driver1 = asyncio.ensure_future(_serve_handshake(peer))
        handle1 = await be.start(_spec(tmp_path), _NullExecutor())
        await asyncio.wait_for(driver1, timeout=2.0)

        consume1 = asyncio.ensure_future(_collect(handle1))
        _notify_turn_started(peer)
        _notify_final_agent_message(peer, text=FINAL_JSON_TEXT)
        _notify_turn_completed(peer, status="completed")
        events1 = await asyncio.wait_for(consume1, timeout=2.0)
        await handle1.outcome()

        # Round 1 left trailing telemetry in the SHARED queue after turn/completed:
        # a late token-usage update + a stray reasoning item, BOTH on round-1 turnId.
        _notify_token_usage(peer, turn_id=TURN_ID)
        peer.notify(
            "item/completed",
            {"item": {"type": "reasoning", "id": "rs_late", "summary": ["leftover"],
                      "content": []},
             "threadId": THREAD_ID, "turnId": TURN_ID},
        )

        # round 2: reuse + a fresh turn.
        resume_spec = _spec(tmp_path, resume=ResumeHandle(
            "codex_app_server", {"thread_id": THREAD_ID}))
        start2 = asyncio.ensure_future(be.start(resume_spec, _NullExecutor()))
        msg = await peer.read_client_message(timeout=2.0)
        assert msg["method"] == "turn/start"
        peer.respond(msg["id"], {"turn": {"id": TURN_ID_2, "status": "inProgress"}})
        handle2 = await asyncio.wait_for(start2, timeout=2.0)

        consume2 = asyncio.ensure_future(_collect(handle2))
        _notify_turn_started(peer, turn_id=TURN_ID_2)
        _notify_final_agent_message(peer, text=FINAL_JSON_TEXT_2, turn_id=TURN_ID_2,
                                    item_id="msg_round2")
        _notify_turn_completed(peer, status="completed", turn_id=TURN_ID_2)
        events2 = await asyncio.wait_for(consume2, timeout=2.0)
        outcome2 = await handle2.outcome()

        # NO round-1-turnId event leaked into round 2's stream.
        leaked = [e for e in events2
                  if e.turn_id is not None and e.turn_id == TURN_ID]
        assert not leaked, f"round-1 turnId events polluted round 2: {leaked}"
        # Round 2's structured output is round 2's payload (not clobbered by the
        # round-1 leftover reasoning / usage).
        assert outcome2.structured_output == json.loads(FINAL_JSON_TEXT_2)
        assert events2[-1].kind == EventKind.RUN_COMPLETED
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 2. aclose closes the resident transport(s) — fixes the happy-path leak.       #
# --------------------------------------------------------------------------- #

async def test_aclose_closes_resident_transport_idempotent(tmp_path):
    """After a run, ``await backend.aclose()`` closes the resident transport (it is
    no longer owned by the RunHandle). The transport's ``_closed`` flag flips and a
    second ``aclose()`` is a harmless no-op."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        driver = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        _notify_final_agent_message(peer)
        _notify_turn_completed(peer, status="completed")
        await asyncio.wait_for(consume, timeout=2.0)
        await handle.outcome()

        # The transport the peer wired is still OPEN after a happy run: the backend
        # owns it now, and run_task never closes it — exactly the LaneC leak.
        transport = peer._transport
        assert transport is not None
        assert transport._closed is False, (
            "resident transport must stay open after a happy run (backend owns it)")

        # backend.aclose() is what the Flow/harness calls in a finally → it must
        # truly close the resident transport (the leak fix).
        await be.aclose()
        assert transport._closed is True, "backend.aclose() must close the resident transport"

        # Idempotent: a second aclose is harmless.
        await be.aclose()
        assert transport._closed is True
    finally:
        await be.aclose()
        await peer.aclose()


async def test_aclose_with_no_runs_is_noop(tmp_path):
    """``aclose()`` on a backend that never ran is a harmless no-op (no transport
    to close, no error)."""
    be = CodexAppServerBackend(mode="structured")
    await be.aclose()
    await be.aclose()


# --------------------------------------------------------------------------- #
# 3. Resume MISS → fail-fast (NOT thread/resume on an ephemeral thread).         #
# --------------------------------------------------------------------------- #

async def test_resume_miss_not_resident_fails_fast(tmp_path):
    """A resume handle for THIS backend whose ``thread_id`` is not resident in this
    instance (cross-run_task / different instance / closed) must fail-fast with a
    clear error — and must NOT spawn a transport or send any thread/resume."""
    peer = FakePeer()
    factory = _CountingFactory(peer)
    be = CodexAppServerBackend(transport_factory=factory, mode="structured")
    try:
        spec = _spec(tmp_path, resume=ResumeHandle(
            "codex_app_server", {"thread_id": "not-a-resident-thread-id"}))
        with pytest.raises(RuntimeError, match="not resident"):
            await be.start(spec, _NullExecutor())
        # Fail-fast must NOT have built / spawned a transport.
        assert factory.built == 0, "resume-miss fail-fast must not build a transport"
    finally:
        await be.aclose()
        await peer.aclose()


async def test_resume_after_aclose_fails_fast(tmp_path):
    """Resuming a thread that WAS resident but whose transport has since been closed
    by ``aclose()`` is a miss → fail-fast (the closed transport was evicted from the
    resident registry)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        driver = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)
        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        _notify_final_agent_message(peer)
        _notify_turn_completed(peer, status="completed")
        await asyncio.wait_for(consume, timeout=2.0)
        outcome = await handle.outcome()

        # Close everything → the resident registry is cleared.
        await be.aclose()

        # Resuming the (now-closed) thread is a miss → fail-fast, no thread/resume.
        with pytest.raises(RuntimeError, match="not resident"):
            await be.start(_spec(tmp_path, resume=outcome.resume), _NullExecutor())
    finally:
        await be.aclose()
        await peer.aclose()


async def test_resume_foreign_backend_still_value_error(tmp_path):
    """The pre-existing cross-backend guard (RT-SESS-5) is unchanged: a resume handle
    minted by a DIFFERENT backend id raises ValueError before any resident lookup."""
    be = CodexAppServerBackend(mode="structured")
    spec = _spec(tmp_path, resume=ResumeHandle("codex", {"session_id": "abc"}))
    with pytest.raises(ValueError, match="codex_app_server"):
        await be.start(spec, _NullExecutor())
    await be.aclose()
