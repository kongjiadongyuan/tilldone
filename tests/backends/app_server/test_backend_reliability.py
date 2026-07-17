"""App-server reliability RTs (M7 Lane A): RT-APP-CONC / RT-APP-RETRY / RT-APP-REASON.

All tests drive a *real* :class:`AppServerTransport` through the in-memory
``FakePeer`` (NO codex spawn, NO gateway, NO wall-clock sleeps). Wire shapes are
lifted from live wire shapes (``ErrorNotification`` for the retry notification;
``ReasoningTextDeltaNotification`` for the reasoning delta).

The three reliability properties:

* **RT-APP-CONC (H)** — concurrent runs / turns on ONE backend instance.
    1. Two CONCURRENT runs (different threads, each its OWN transport) each issue a
       tool call; each call must route to ITS OWN run's executor — never cross-wired.
       Routing is by (threadId,turnId,callId) → the owning run, which the per-transport
       ``item/tool/call`` handler closure delivers by construction. The proof uses a
       BLOCKING executor so BOTH tool calls are genuinely in-flight at once (real
       overlap, not a sequential stand-in): if routing cross-wired, an executor would
       see the OTHER run's call and the assertion fails.
    2. A SECOND concurrent turn on a thread that already has an active turn is REJECTED
       (app-server permits one active turn per thread) — never silently interleaved.

* **RT-APP-RETRY (M)** — an ``error`` notification with ``willRetry=true`` is NOT a
  terminal; the run waits for the eventual ``turn/completed``. ``willRetry=false`` +
  a failed terminal IS the terminal (→ RUN_FAILED).

* **RT-APP-REASON (M)** — reasoning delta/item does NOT reach the DEFAULT public sink
  (``experimentalRawEvents`` is not on by default); it stays in raw/debug. With raw
  events opted in, reasoning is promoted to the public feed.

asyncio_mode="auto" (pyproject) → async tests need no decorator.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tilldone.backends.app_server.backend import CodexAppServerBackend
from tilldone.backends.app_server.events import map_notification
from tilldone.core.enums import EventKind
from tilldone.core.event_sink import JsonlEventSink
from tilldone.core.spec import AgentRunSpec
from tilldone.core.tools import ToolCall, ToolDef, ToolResult

from .fake_peer import FakePeer
from .test_backend import (
    THREAD_ID,
    TURN_ID,
    _notify_turn_completed,
    _notify_turn_started,
    _serve_handshake,
)
from .test_backend_tools import (
    _CONTRACT_TOOLS,
    _notify_narration,
    _notify_tool_call_completed,
    _notify_tool_call_started,
    _serve_one_tool_call,
    RecordingExecutor,
)

# A distinct thread/turn id pair for the SECOND concurrent run (its own transport).
THREAD_ID_B = "019ef999-bbbb-7f23-8fe6-201acd4c6965"
TURN_ID_B = "019ef999-cccc-7c41-9ae0-c9958f25576b"


def _spec(tmp_path: Path, *, tools=_CONTRACT_TOOLS, **kw) -> AgentRunSpec:
    """A tools-mode spec (host_tool_calls=True → no output_schema)."""
    return AgentRunSpec(
        prompt=kw.pop("prompt", "Use the tools then finish."),
        tools=tools,
        cwd=tmp_path,
        output_schema=None,
        **kw,
    )


async def _collect(handle):
    return [ev async for ev in handle.events]


# =========================================================================== #
# RT-APP-CONC #1 — concurrent runs (different threads): NO cross-wiring        #
# =========================================================================== #


class _BlockingExecutor:
    """A host ToolExecutor that BLOCKS inside ``invoke`` until released.

    The blocking is what makes the concurrency proof genuine: both runs' tool calls
    are held IN-FLIGHT simultaneously (each ``invoke`` parks on a per-executor Event),
    so the two ``item/tool/call`` dispatches truly overlap rather than running
    sequentially. Each executor records the calls IT received; the test asserts each
    executor saw ONLY its own run's call — a cross-wire would deliver the other run's
    call here and fail the assertion.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[ToolCall] = []
        self.entered = asyncio.Event()   # set when invoke() is first entered
        self.release = asyncio.Event()   # awaited inside invoke() before returning

    async def invoke(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        self.entered.set()
        await self.release.wait()
        # Echo which executor handled it so the wire reply is attributable too.
        return ToolResult(content=f"handled-by-{self.name}:{call.arguments.get('answer')}")


class _PeerPoolFactory:
    """A ``transport_factory`` for ONE backend that hands out a queue of peers' transports.

    The literal RT-APP-CONC scenario is "同一 backend 实例并发多 run" — ONE backend driving
    several concurrent runs. A backend builds a transport per run via ``transport_factory``;
    this factory returns each queued peer's transport in turn, so a single backend instance
    drives BOTH runs (sharing its ``_live`` registry + ``_active_turns`` set) while each run
    still gets its OWN transport + ``item/tool/call`` handler (the per-(threadId,turnId,
    callId) routing boundary). ``**kwargs`` (e.g. request_timeout_s) are forwarded.
    """

    def __init__(self, peers: list[FakePeer]) -> None:
        self._peers = list(peers)

    def __call__(self, **kwargs):
        return self._peers.pop(0).make_transport(**kwargs)


async def test_conc_two_runs_one_backend_no_cross_wiring(tmp_path):
    """RT-APP-CONC #1 (REAL overlap, ONE backend instance): two CONCURRENT runs on a
    SINGLE backend, each with an in-flight tool call, route each call to ITS OWN executor.

    This is the literal RT scenario ("同一 backend 实例并发多 run"): ONE
    ``CodexAppServerBackend`` drives both runs (shared registry), each run on its OWN
    transport/thread. Genuine concurrency: BLOCKING executors park each ``invoke`` so BOTH
    tool calls are in-flight at once (proven via the two ``entered`` events) before either
    is released. The per-transport ``item/tool/call`` handler closure delivers each call to
    the owning run's executor — the (threadId,turnId,callId) routing. A shared/cross-wired
    handler would deliver run B's call to executor A and the per-executor assertions fail
    (verified non-vacuous by injecting a cross-wire bug → this test fails).
    """
    peer_a, peer_b = FakePeer(), FakePeer()
    ex_a, ex_b = _BlockingExecutor("A"), _BlockingExecutor("B")
    # ONE backend instance, two runs — the factory hands out each run's own transport.
    be = CodexAppServerBackend(transport_factory=_PeerPoolFactory([peer_a, peer_b]))
    try:
        drv_a = asyncio.ensure_future(_serve_handshake(peer_a, thread_id=THREAD_ID))
        handle_a = await be.start(_spec(tmp_path, prompt="run A"), ex_a)
        await asyncio.wait_for(drv_a, timeout=2.0)

        drv_b = asyncio.ensure_future(_serve_handshake(peer_b, thread_id=THREAD_ID_B))
        handle_b = await be.start(_spec(tmp_path, prompt="run B"), ex_b)
        await asyncio.wait_for(drv_b, timeout=2.0)

        consume_a = asyncio.ensure_future(_collect(handle_a))
        consume_b = asyncio.ensure_future(_collect(handle_b))

        _notify_turn_started(peer_a, turn_id=TURN_ID, thread_id=THREAD_ID)
        _notify_turn_started(peer_b, turn_id=TURN_ID, thread_id=THREAD_ID_B)

        # Fire BOTH tool calls (each on its own transport) WITHOUT replying — the
        # blocking executors park, so both are genuinely in-flight at once.
        peer_a.server_request(
            "item/tool/call",
            {"threadId": THREAD_ID, "turnId": TURN_ID, "callId": "call_A",
             "namespace": "contract", "tool": "return", "arguments": {"answer": "AAA"}},
            id=0)
        peer_b.server_request(
            "item/tool/call",
            {"threadId": THREAD_ID_B, "turnId": TURN_ID, "callId": "call_B",
             "namespace": "contract", "tool": "return", "arguments": {"answer": "BBB"}},
            id=0)

        # Real overlap proof: BOTH executors entered invoke() before EITHER returns.
        await asyncio.wait_for(ex_a.entered.wait(), timeout=2.0)
        await asyncio.wait_for(ex_b.entered.wait(), timeout=2.0)

        # ROUTING ASSERTION (cross-wire detector), checked WHILE both are in-flight:
        # each executor saw EXACTLY its own run's call, never the other's.
        assert [c.call_id for c in ex_a.calls] == ["call_A"], ex_a.calls
        assert [c.call_id for c in ex_b.calls] == ["call_B"], ex_b.calls
        assert ex_a.calls[0].arguments == {"answer": "AAA"}
        assert ex_b.calls[0].arguments == {"answer": "BBB"}

        # Release both; each reply comes back on its OWN transport, attributable to the
        # executor that (correctly) handled it (a cross-wire would mislabel the text).
        ex_a.release.set()
        ex_b.release.set()
        reply_a = await peer_a.read_client_message()
        reply_b = await peer_b.read_client_message()
        assert reply_a["result"]["contentItems"][0]["text"] == "handled-by-A:AAA"
        assert reply_b["result"]["contentItems"][0]["text"] == "handled-by-B:BBB"

        # Finish both turns cleanly.
        _notify_tool_call_completed(peer_a, call_id="call_A", tool="return",
                                    arguments={"answer": "AAA"}, turn_id=TURN_ID,
                                    thread_id=THREAD_ID)
        _notify_tool_call_completed(peer_b, call_id="call_B", tool="return",
                                    arguments={"answer": "BBB"}, turn_id=TURN_ID,
                                    thread_id=THREAD_ID_B)
        _notify_narration(peer_a, turn_id=TURN_ID, thread_id=THREAD_ID)
        _notify_narration(peer_b, turn_id=TURN_ID, thread_id=THREAD_ID_B)
        _notify_turn_completed(peer_a, turn_id=TURN_ID, thread_id=THREAD_ID)
        _notify_turn_completed(peer_b, turn_id=TURN_ID, thread_id=THREAD_ID_B)

        await asyncio.wait_for(consume_a, timeout=2.0)
        await asyncio.wait_for(consume_b, timeout=2.0)
        assert (await handle_a.outcome()).status == "completed"
        assert (await handle_b.outcome()).status == "completed"
        # Final cross-wire guard: still exactly one call each, still the right one.
        assert [c.call_id for c in ex_a.calls] == ["call_A"]
        assert [c.call_id for c in ex_b.calls] == ["call_B"]
    finally:
        await be.aclose()
        await peer_a.aclose()
        await peer_b.aclose()


async def test_conc_two_runs_gather_start_no_cross_wiring(tmp_path):
    """RT-APP-CONC #1 variant (ONE backend, ``asyncio.gather`` starts): the two runs on a
    SINGLE backend are STARTED via ``asyncio.gather`` (their handshakes truly interleave on
    the event loop), then each in-flight tool call still routes to its own executor. Proves
    isolation holds even when the two starts overlap on one backend instance.
    """
    peer_a, peer_b = FakePeer(), FakePeer()
    ex_a, ex_b = _BlockingExecutor("A"), _BlockingExecutor("B")
    be = CodexAppServerBackend(transport_factory=_PeerPoolFactory([peer_a, peer_b]))
    try:
        # Drive both server handshakes concurrently AND both backend.start()s concurrently
        # on ONE backend instance.
        drv_a = asyncio.ensure_future(_serve_handshake(peer_a, thread_id=THREAD_ID))
        drv_b = asyncio.ensure_future(_serve_handshake(peer_b, thread_id=THREAD_ID_B))
        handle_a, handle_b = await asyncio.gather(
            be.start(_spec(tmp_path, prompt="A"), ex_a),
            be.start(_spec(tmp_path, prompt="B"), ex_b),
        )
        await asyncio.wait_for(asyncio.gather(drv_a, drv_b), timeout=2.0)

        consume_a = asyncio.ensure_future(_collect(handle_a))
        consume_b = asyncio.ensure_future(_collect(handle_b))
        _notify_turn_started(peer_a, turn_id=TURN_ID, thread_id=THREAD_ID)
        _notify_turn_started(peer_b, turn_id=TURN_ID, thread_id=THREAD_ID_B)

        peer_a.server_request(
            "item/tool/call",
            {"threadId": THREAD_ID, "turnId": TURN_ID, "callId": "ca",
             "namespace": "contract", "tool": "return", "arguments": {"answer": "A"}}, id=0)
        peer_b.server_request(
            "item/tool/call",
            {"threadId": THREAD_ID_B, "turnId": TURN_ID, "callId": "cb",
             "namespace": "contract", "tool": "return", "arguments": {"answer": "B"}}, id=0)

        await asyncio.wait_for(ex_a.entered.wait(), timeout=2.0)
        await asyncio.wait_for(ex_b.entered.wait(), timeout=2.0)
        assert [c.call_id for c in ex_a.calls] == ["ca"]
        assert [c.call_id for c in ex_b.calls] == ["cb"]

        ex_a.release.set()
        ex_b.release.set()
        await peer_a.read_client_message()
        await peer_b.read_client_message()
        _notify_tool_call_completed(peer_a, call_id="ca", tool="return",
                                    arguments={"answer": "A"}, turn_id=TURN_ID,
                                    thread_id=THREAD_ID)
        _notify_tool_call_completed(peer_b, call_id="cb", tool="return",
                                    arguments={"answer": "B"}, turn_id=TURN_ID,
                                    thread_id=THREAD_ID_B)
        _notify_narration(peer_a, turn_id=TURN_ID, thread_id=THREAD_ID)
        _notify_narration(peer_b, turn_id=TURN_ID, thread_id=THREAD_ID_B)
        _notify_turn_completed(peer_a, turn_id=TURN_ID, thread_id=THREAD_ID)
        _notify_turn_completed(peer_b, turn_id=TURN_ID, thread_id=THREAD_ID_B)
        await asyncio.wait_for(asyncio.gather(consume_a, consume_b), timeout=2.0)
        assert (await handle_a.outcome()).status == "completed"
        assert (await handle_b.outcome()).status == "completed"
    finally:
        await be.aclose()
        await peer_a.aclose()
        await peer_b.aclose()


# =========================================================================== #
# RT-APP-CONC #2 — same-thread concurrent second turn is REJECTED (lock)       #
# =========================================================================== #


async def test_conc_second_turn_same_thread_rejected_no_interleave(tmp_path):
    """RT-APP-CONC #2 (REAL overlap): while a turn is IN-FLIGHT on a resident thread,
    a SECOND concurrent turn on that SAME thread is REJECTED — never silently interleaved.

    Round 1's turn is held open (no terminal), so its per-thread active-turn slot is
    still held. A concurrent ``start(resume=...)`` targeting the same resident thread
    must fail fast (RuntimeError naming the active-turn rejection) WITHOUT issuing a
    second ``turn/start`` on the wire (no interleave). After round 1 completes (slot
    released), a fresh turn on the thread is accepted again — proving it was a live
    lock, not a permanent ban.
    """
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    try:
        # Round 1 — start a turn and DELIBERATELY leave it in-flight (no terminal yet).
        drv = asyncio.ensure_future(_serve_handshake(peer))
        handle1 = await be.start(_spec(tmp_path), ex)
        await asyncio.wait_for(drv, timeout=2.0)
        consume1 = asyncio.ensure_future(_collect(handle1))
        _notify_turn_started(peer)  # turn is now active; slot held, NO turn/completed.

        # CONCURRENT second turn on the SAME resident thread → must be REJECTED fast.
        resume = (await _peek_resume(be, handle1))
        with pytest.raises(RuntimeError, match="already has an active turn"):
            await be.start(_spec(tmp_path, resume=resume), ex)

        # NO second turn/start hit the wire (the rejection happened before any request).
        # Prove it: the next client message is the round-1 tool round-trip we now drive,
        # NOT a stray turn/start. Drive round 1 to completion.
        await _serve_one_tool_call(
            peer, server_id=0, call_id="call_R1", tool="return",
            arguments={"answer": "ONE"})
        _notify_narration(peer)
        _notify_turn_completed(peer)
        await asyncio.wait_for(consume1, timeout=2.0)
        out1 = await handle1.outcome()
        assert out1.status == "completed"

        # Slot released after the terminal → a NEW turn on the same thread is accepted.
        spec2 = _spec(tmp_path, resume=out1.resume)
        start2 = asyncio.ensure_future(be.start(spec2, ex))
        msg = await peer.read_client_message()
        assert msg["method"] == "turn/start", msg
        peer.respond(msg["id"], {"turn": {"id": TURN_ID_B, "status": "inProgress"}})
        handle2 = await asyncio.wait_for(start2, timeout=2.0)
        consume2 = asyncio.ensure_future(_collect(handle2))
        _notify_turn_started(peer, turn_id=TURN_ID_B)
        _notify_narration(peer, turn_id=TURN_ID_B)
        _notify_turn_completed(peer, turn_id=TURN_ID_B)
        await asyncio.wait_for(consume2, timeout=2.0)
        assert (await handle2.outcome()).status == "completed"
    finally:
        await be.aclose()
        await peer.aclose()


async def _peek_resume(be: CodexAppServerBackend, handle):
    """Build a resume handle pointing at the handle's (resident) thread WITHOUT ending it.

    ``outcome()`` is not yet meaningful mid-turn, so we mint the resume handle from the
    handle's own ``_resume_handle()`` (the same token outcome() would carry) to target
    the resident thread for the rejected concurrent start.
    """
    return handle._resume_handle()


async def test_conc_second_turn_rejection_does_not_release_first_slot(tmp_path):
    """RT-APP-CONC #2 corollary: a REJECTED second turn must not corrupt the first turn's
    slot — after the rejection, round 1 still completes and round 2 (post-terminal) still
    starts. (Guards against the reject path accidentally freeing the active slot.)"""
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    try:
        drv = asyncio.ensure_future(_serve_handshake(peer))
        handle1 = await be.start(_spec(tmp_path), ex)
        await asyncio.wait_for(drv, timeout=2.0)
        consume1 = asyncio.ensure_future(_collect(handle1))
        _notify_turn_started(peer)

        resume = handle1._resume_handle()
        # Reject TWICE — neither attempt may leak/clear the live slot.
        for _ in range(2):
            with pytest.raises(RuntimeError, match="already has an active turn"):
                await be.start(_spec(tmp_path, resume=resume), ex)

        await _serve_one_tool_call(
            peer, server_id=0, call_id="c1", tool="return", arguments={"answer": "ONE"})
        _notify_narration(peer)
        _notify_turn_completed(peer)
        await asyncio.wait_for(consume1, timeout=2.0)
        assert (await handle1.outcome()).status == "completed"
    finally:
        await be.aclose()
        await peer.aclose()


# =========================================================================== #
# RT-APP-RETRY — willRetry=true is NOT a terminal; the real terminal wins      #
# =========================================================================== #


def _error_notification(*, will_retry: bool, message="upstream hiccup",
                        codex_error_info="serverOverloaded",
                        turn_id=TURN_ID, thread_id=THREAD_ID) -> dict:
    """An ErrorNotification frame (schema: {error, threadId, turnId, willRetry})."""
    return {
        "method": "error",
        "params": {
            "threadId": thread_id, "turnId": turn_id, "willRetry": will_retry,
            "error": {"message": message, "codexErrorInfo": codex_error_info},
        },
    }


def test_map_error_notification_willretry_true_is_nonterminal_notice():
    """Pure mapper: error(willRetry=true) → a NON-terminal BACKEND_NOTICE (never a
    terminal), carrying will_retry=True so the retry is observable."""
    ev = map_notification(_error_notification(will_retry=True), seq=1)
    assert ev is not None
    assert ev.kind == EventKind.BACKEND_NOTICE
    assert ev.kind not in (EventKind.RUN_COMPLETED, EventKind.RUN_FAILED)
    assert ev.data["notice"] == "turn_error"
    assert ev.data["will_retry"] is True
    assert ev.data["message"] == "upstream hiccup"
    assert ev.data["codex_error_info"] == "serverOverloaded"
    assert ev.turn_id == TURN_ID


def test_map_error_notification_willretry_false_still_nonterminal():
    """Even willRetry=false maps to a NON-terminal notice — the TERMINAL is the separate
    turn/completed(status=failed). The error notice never ends the stream itself."""
    ev = map_notification(_error_notification(will_retry=False), seq=1)
    assert ev is not None
    assert ev.kind == EventKind.BACKEND_NOTICE
    assert ev.data["will_retry"] is False


async def test_retry_willretry_true_then_completed_single_terminal(tmp_path):
    """RT-APP-RETRY: error(willRetry=true) followed by a real turn/completed → the run
    settles on the COMPLETION; the retry notice did NOT terminate the stream early.

    Exactly one terminal (RUN_COMPLETED) and it is the LAST event; the retry surfaces as
    a non-terminal BACKEND_NOTICE BEFORE it.
    """
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    try:
        drv = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), ex)
        await asyncio.wait_for(drv, timeout=2.0)
        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        # A retryable error notice MID-TURN — must NOT terminate.
        peer.send_line(_error_notification(will_retry=True))
        # The turn recovers and completes for real.
        _notify_narration(peer)
        _notify_turn_completed(peer, status="completed")
        events = await asyncio.wait_for(consume, timeout=2.0)
        outcome = await handle.outcome()

        kinds = [e.kind for e in events]
        # Exactly one terminal, and it is the completion (last event).
        terminals = [k for k in kinds if k in (EventKind.RUN_COMPLETED, EventKind.RUN_FAILED)]
        assert terminals == [EventKind.RUN_COMPLETED], kinds
        assert kinds[-1] == EventKind.RUN_COMPLETED
        # The retry notice appeared, BEFORE the terminal, and is non-terminal.
        notices = [e for e in events if e.kind == EventKind.BACKEND_NOTICE
                   and e.data.get("notice") == "turn_error"]
        assert len(notices) == 1 and notices[0].data["will_retry"] is True
        assert outcome.status == "completed"
    finally:
        await be.aclose()
        await peer.aclose()


async def test_retry_multiple_willretry_true_then_completed(tmp_path):
    """RT-APP-RETRY: SEVERAL willRetry=true notices before the eventual completion — none
    terminate; the single terminal is still the final turn/completed."""
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    try:
        drv = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), ex)
        await asyncio.wait_for(drv, timeout=2.0)
        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        for i in range(3):
            peer.send_line(_error_notification(will_retry=True, message=f"retry {i}"))
        _notify_narration(peer)
        _notify_turn_completed(peer, status="completed")
        events = await asyncio.wait_for(consume, timeout=2.0)
        assert (await handle.outcome()).status == "completed"
        terminals = [e for e in events
                     if e.kind in (EventKind.RUN_COMPLETED, EventKind.RUN_FAILED)]
        assert len(terminals) == 1 and terminals[0].kind == EventKind.RUN_COMPLETED
        notices = [e for e in events if e.kind == EventKind.BACKEND_NOTICE
                   and e.data.get("notice") == "turn_error"]
        assert len(notices) == 3
    finally:
        await be.aclose()
        await peer.aclose()


async def test_retry_willretry_false_then_failed_terminal_run_failed(tmp_path):
    """RT-APP-RETRY: error(willRetry=false) + a turn/completed(status=failed) → RUN_FAILED.

    The non-retryable error notice is observable but the TERMINAL is the failed
    turn/completed (the real terminal), so the run settles failed. Exactly one terminal.
    """
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    try:
        drv = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), ex)
        await asyncio.wait_for(drv, timeout=2.0)
        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        peer.send_line(_error_notification(
            will_retry=False, message="fatal upstream", codex_error_info="internalServerError"))
        _notify_turn_completed(
            peer, status="failed",
            error={"message": "fatal upstream", "codexErrorInfo": "internalServerError"})
        events = await asyncio.wait_for(consume, timeout=2.0)
        outcome = await handle.outcome()

        kinds = [e.kind for e in events]
        terminals = [k for k in kinds if k in (EventKind.RUN_COMPLETED, EventKind.RUN_FAILED)]
        assert terminals == [EventKind.RUN_FAILED], kinds
        assert kinds[-1] == EventKind.RUN_FAILED
        assert outcome.status == "failed"
        assert outcome.error is not None
        # internalServerError → transient/fatal classification (not a crash); message kept.
        assert "fatal upstream" in (outcome.error.message or "")
        notices = [e for e in events if e.kind == EventKind.BACKEND_NOTICE
                   and e.data.get("notice") == "turn_error"]
        assert len(notices) == 1 and notices[0].data["will_retry"] is False
    finally:
        await be.aclose()
        await peer.aclose()


# =========================================================================== #
# RT-APP-REASON — reasoning is debug/raw-only by default; opt-in promotes it   #
# =========================================================================== #


def _reasoning_item_note(*, item_id="rs1", turn_id=TURN_ID, thread_id=THREAD_ID) -> dict:
    return {
        "method": "item/completed",
        "params": {"item": {"type": "reasoning", "id": item_id,
                            "summary": ["thinking"], "content": []},
                   "threadId": thread_id, "turnId": turn_id},
    }


def _reasoning_delta_note(*, item_id="rs1", delta="step…", turn_id=TURN_ID,
                          thread_id=THREAD_ID) -> dict:
    """ReasoningTextDeltaNotification (schema v2) frame."""
    return {
        "method": "item/reasoningText/delta",
        "params": {"contentIndex": 0, "delta": delta, "itemId": item_id,
                   "threadId": thread_id, "turnId": turn_id},
    }


def test_map_reasoning_item_is_tagged_raw_only():
    """Pure mapper: a reasoning ITEM stays a REASONING event but is tagged raw_only=True
    (debug-channel) so a public sink suppresses it from the default feed."""
    ev = map_notification(_reasoning_item_note(), seq=1)
    assert ev is not None
    assert ev.kind == EventKind.REASONING
    assert ev.data.get("raw_only") is True


def test_map_reasoning_delta_is_reasoning_raw_only_not_assistant_text():
    """Pure mapper: a reasoning DELTA maps to a streaming REASONING event tagged
    raw_only=True — NOT an ASSISTANT_TEXT (so it never leaks into the public answer feed)
    and NOT dropped (so the debug layer stays faithful)."""
    ev = map_notification(_reasoning_delta_note(delta="abc"), seq=1)
    assert ev is not None
    assert ev.kind == EventKind.REASONING
    assert ev.data.get("raw_only") is True
    assert ev.data.get("streaming") is True
    assert ev.data.get("delta") == "abc"


def test_public_sink_drops_raw_only_reasoning_keeps_raw_in_messages(tmp_path):
    """RT-APP-REASON: a raw_only reasoning event is EXCLUDED from the public events feed
    by DEFAULT (no public_only needed — the per-event marker drives it), while its raw
    payload still reaches the debug/messages layer. Non-reasoning events pass through."""
    events_path = tmp_path / "events.jsonl"
    messages_path = tmp_path / "messages.jsonl"

    reasoning_ev = map_notification(_reasoning_item_note(), seq=1)
    assistant_ev = map_notification(
        {"method": "item/completed",
         "params": {"item": {"type": "agentMessage", "id": "m", "text": "hi",
                             "phase": "final_answer"},
                    "threadId": THREAD_ID, "turnId": TURN_ID}}, seq=2)

    # DEFAULT sink (public_only=False) — the raw_only marker alone must suppress reasoning.
    with JsonlEventSink(events_path, messages_path=messages_path) as sink:
        sink(reasoning_ev)
        sink(assistant_ev)

    public = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    public_kinds = [r["kind"] for r in public]
    assert "reasoning" not in public_kinds, public_kinds
    assert "assistant_text" in public_kinds
    # Raw/debug layer still has the reasoning frame (faithful debug view).
    msgs = [json.loads(line) for line in messages_path.read_text().splitlines() if line.strip()]
    assert any(m["seq"] == 1 for m in msgs), "reasoning raw must reach the debug layer"


async def test_reason_suppressed_from_default_public_sink_via_run(tmp_path):
    """RT-APP-REASON end-to-end (default backend): a reasoning item emitted DURING a run
    does NOT reach the default public sink, but its raw IS in the messages/debug layer.

    Drives a real backend over fake_peer, feeds the handle stream through a JsonlEventSink
    (the public sink), and asserts the public feed has no reasoning while the debug layer
    does. The handle stream itself still carries REASONING (debug consumers can see it) —
    suppression is at the public-sink boundary, not a silent drop.
    """
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    events_path = tmp_path / "events.jsonl"
    messages_path = tmp_path / "messages.jsonl"
    try:
        drv = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), ex)
        await asyncio.wait_for(drv, timeout=2.0)

        async def consume_into_sink():
            stream_kinds = []
            with JsonlEventSink(events_path, messages_path=messages_path) as sink:
                async for ev in handle.events:
                    stream_kinds.append(ev.kind)
                    sink(ev)
            return stream_kinds

        consume = asyncio.ensure_future(consume_into_sink())
        _notify_turn_started(peer)
        peer.send_line(_reasoning_delta_note(delta="thinking hard"))
        peer.send_line(_reasoning_item_note())
        _notify_narration(peer)
        _notify_turn_completed(peer)
        stream_kinds = await asyncio.wait_for(consume, timeout=2.0)
        assert (await handle.outcome()).status == "completed"

        # The handle's RAW stream DID carry reasoning (so debug/raw consumers see it).
        assert EventKind.REASONING in stream_kinds
        # The PUBLIC sink feed did NOT (suppressed by default — experimentalRawEvents off).
        public = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        assert "reasoning" not in [r["kind"] for r in public], public
        # The DEBUG/messages layer still recorded the reasoning frames' raw.
        assert messages_path.exists() and messages_path.read_text().strip(), \
            "reasoning raw must be observable in the debug layer"
    finally:
        await be.aclose()
        await peer.aclose()


async def test_reason_opt_in_raw_events_promotes_reasoning_to_public(tmp_path):
    """RT-APP-REASON opt-in: with experimental_raw_events=True the backend PROMOTES
    reasoning to the public feed (raw_only=False), so a public sink keeps it.

    This is the explicit ``experimentalRawEvents`` opt-in (NOT on by default) — proves the
    suppression is a default posture, not a hard ban.
    """
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport,
                               experimental_raw_events=True)
    events_path = tmp_path / "events.jsonl"
    try:
        drv = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), ex)
        await asyncio.wait_for(drv, timeout=2.0)

        async def consume_into_sink():
            with JsonlEventSink(events_path) as sink:
                async for ev in handle.events:
                    sink(ev)

        consume = asyncio.ensure_future(consume_into_sink())
        _notify_turn_started(peer)
        peer.send_line(_reasoning_item_note())
        _notify_narration(peer)
        _notify_turn_completed(peer)
        await asyncio.wait_for(consume, timeout=2.0)
        assert (await handle.outcome()).status == "completed"

        public = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        assert "reasoning" in [r["kind"] for r in public], \
            "raw-events opt-in must promote reasoning into the public feed"
        # And the promoted event is no longer raw_only.
        reasoning_rows = [r for r in public if r["kind"] == "reasoning"]
        assert reasoning_rows and reasoning_rows[0]["data"].get("raw_only") is False
    finally:
        await be.aclose()
        await peer.aclose()


def test_reason_opt_in_handle_restamps_raw_only_false():
    """Unit: the RunHandle's channel projection re-stamps a raw_only reasoning event to
    raw_only=False ONLY when raw_events is on (and leaves non-reasoning events alone)."""
    from tilldone.backends.app_server.backend import CodexAppServerRunHandle

    reasoning_ev = map_notification(_reasoning_item_note(), seq=1)
    assert reasoning_ev.data["raw_only"] is True

    # raw_events OFF (default) → unchanged (still raw_only=True).
    h_off = CodexAppServerRunHandle(object(), thread_id=THREAD_ID, raw_events=False)  # type: ignore[arg-type]
    assert h_off._project_channel(reasoning_ev).data["raw_only"] is True

    # raw_events ON → promoted to raw_only=False (a NEW event; original untouched).
    h_on = CodexAppServerRunHandle(object(), thread_id=THREAD_ID, raw_events=True)  # type: ignore[arg-type]
    projected = h_on._project_channel(reasoning_ev)
    assert projected.data["raw_only"] is False
    assert reasoning_ev.data["raw_only"] is True  # original frozen event not mutated

    # A non-reasoning event is never touched by the projection.
    other = map_notification(
        {"method": "item/completed",
         "params": {"item": {"type": "agentMessage", "id": "m", "text": "x",
                             "phase": "final_answer"},
                    "threadId": THREAD_ID, "turnId": TURN_ID}}, seq=2)
    assert h_on._project_channel(other) is other
