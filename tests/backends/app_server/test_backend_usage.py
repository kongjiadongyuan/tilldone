"""RT-APP-USAGE — usage accounting for CodexAppServerBackend.

Pins usage accounting against ``thread/tokenUsage/updated`` (last + total):
use each turn's final ``last``, dedupe by (threadId, turnId) so resume/retry does
not double-count, and fail fast when usage is missing.

Three properties, each driven against a *real* :class:`AppServerTransport` over
the in-memory ``FakePeer`` (NO codex spawn, NO gateway), using the exact wire
shape of ``ThreadTokenUsageUpdatedNotification``: the notification carries
top-level ``threadId`` + ``turnId`` + a ``tokenUsage`` object with required
``last`` and ``total`` ``TokenUsageBreakdown``s (each with cachedInputTokens /
inputTokens / outputTokens / reasoningOutputTokens / totalTokens) plus an
optional ``modelContextWindow``.

1. **per-turn last wins (not summed within a turn)** — a turn that emits SEVERAL
   ``thread/tokenUsage/updated`` frames settles ``outcome.usage`` to the FINAL
   ``last`` for that turn (the updates supersede, never accumulate, within one
   turn).
2. **(threadId,turnId) dedup across rounds** — a resident two-turn run
   accumulates each turn's final ``last`` into the reported ``total``, and a
   turn whose usage update is observed TWICE (a trailing duplicate / a retry)
   is counted ONCE (dedup by (threadId,turnId)) — no double count.
3. **missing-usage fail-fast (G5)** — a COMPLETED turn that emitted NO usage
   update at all surfaces an explicit ``available: False`` marker (a documented
   "usage unavailable" signal a budget layer keys on to fail-fast), NOT a silent
   ``usage=None`` that a budget layer would mistake for zero cost.

``asyncio_mode="auto"`` (pyproject) → async tests need no decorator.
"""

from __future__ import annotations

import asyncio

import pytest

from tilldone.backends.app_server.backend import CodexAppServerBackend
from tilldone.core.spec import ResumeHandle

from .fake_peer import FakePeer
from .test_backend import (
    FINAL_JSON_TEXT,
    THREAD_ID,
    TURN_ID,
    _NullExecutor,
    _notify_final_agent_message,
    _notify_turn_completed,
    _notify_turn_started,
    _serve_handshake,
    _spec,
)
from .test_backend_multiround import (
    FINAL_JSON_TEXT_2,
    TURN_ID_2,
    _collect,
)


# --------------------------------------------------------------------------- #
# A full TokenUsageBreakdown helper (every schema-required field present).      #
# --------------------------------------------------------------------------- #

def _breakdown(*, total, inp=0, out=0, cached=0, reasoning=0):
    """Build a schema-complete ``TokenUsageBreakdown``."""
    return {
        "cachedInputTokens": cached,
        "inputTokens": inp,
        "outputTokens": out,
        "reasoningOutputTokens": reasoning,
        "totalTokens": total,
    }


def _notify_usage(peer: FakePeer, *, last, total, turn_id=TURN_ID,
                  thread_id=THREAD_ID, ctx_window=258400):
    """Emit a ``thread/tokenUsage/updated`` with explicit last/total breakdowns
    (the real wire shape: top-level threadId + turnId + tokenUsage{last,total})."""
    token_usage = {"last": last, "total": total}
    if ctx_window is not None:
        token_usage["modelContextWindow"] = ctx_window
    peer.notify(
        "thread/tokenUsage/updated",
        {"threadId": thread_id, "turnId": turn_id, "tokenUsage": token_usage},
    )


# --------------------------------------------------------------------------- #
# 1. Within one turn, the FINAL `last` wins — multiple updates do NOT sum.       #
# --------------------------------------------------------------------------- #

async def test_single_turn_multiple_updates_keep_final_last(tmp_path):
    """A turn emits THREE ``thread/tokenUsage/updated`` frames (the running totals
    the live app-server streams). ``outcome.usage`` must reflect the FINAL ``last``
    for the turn — NOT the sum of the three ``last`` values."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        driver = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        # Three updates within ONE turn — last is the per-update increment, total is
        # the server's running cumulative. The LAST frame is authoritative.
        _notify_usage(peer, last=_breakdown(total=100, inp=80, out=20),
                      total=_breakdown(total=100, inp=80, out=20))
        _notify_usage(peer, last=_breakdown(total=50, inp=40, out=10),
                      total=_breakdown(total=150, inp=120, out=30))
        _notify_usage(peer, last=_breakdown(total=33, inp=25, out=8),
                      total=_breakdown(total=183, inp=145, out=38))
        _notify_final_agent_message(peer, text=FINAL_JSON_TEXT)
        _notify_turn_completed(peer, status="completed")
        await asyncio.wait_for(consume, timeout=2.0)
        outcome = await handle.outcome()

        assert outcome.status == "completed"
        assert outcome.usage is not None
        assert outcome.usage["available"] is True
        # FINAL last wins (33), NOT the sum of the three lasts (100+50+33=183).
        assert outcome.usage["last"]["totalTokens"] == 33
        # The accumulated total for a single turn == that turn's final last (33),
        # NOT the within-turn sum.
        assert outcome.usage["total"]["totalTokens"] == 33
        # The server's own reported cumulative total is preserved (not lost).
        assert outcome.usage["server_total"]["totalTokens"] == 183
        # modelContextWindow carried through.
        assert outcome.usage["modelContextWindow"] == 258400
        # Exactly one turn contributed to the accumulation.
        assert outcome.usage["turns_counted"] == 1
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 2a. Two turns on one resident thread → accumulated total = Σ each turn's last. #
# --------------------------------------------------------------------------- #

async def test_multiround_accumulates_per_turn_last_by_threadid_turnid(tmp_path):
    """Resident two-turn run (same thread, two turnIds). Turn 1's final last = 100,
    turn 2's final last = 250. The accumulated ``total`` reported on round 2's
    outcome is 100+250 = 350 (Σ each turn's final last, keyed by
    (threadId,turnId)); round 1's outcome saw only its own 100."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        # round 1
        driver1 = asyncio.ensure_future(_serve_handshake(peer))
        handle1 = await be.start(_spec(tmp_path), _NullExecutor())
        await asyncio.wait_for(driver1, timeout=2.0)
        consume1 = asyncio.ensure_future(_collect(handle1))
        _notify_turn_started(peer)
        _notify_usage(peer, last=_breakdown(total=100, inp=80, out=20),
                      total=_breakdown(total=100, inp=80, out=20))
        _notify_final_agent_message(peer, text=FINAL_JSON_TEXT)
        _notify_turn_completed(peer, status="completed")
        await asyncio.wait_for(consume1, timeout=2.0)
        outcome1 = await handle1.outcome()

        assert outcome1.usage["available"] is True
        assert outcome1.usage["last"]["totalTokens"] == 100
        assert outcome1.usage["total"]["totalTokens"] == 100
        assert outcome1.usage["turns_counted"] == 1

        # round 2: resident reuse, a second turn on the SAME thread.
        resume_spec = _spec(tmp_path, resume=outcome1.resume, prompt="round two")
        start2 = asyncio.ensure_future(be.start(resume_spec, _NullExecutor()))
        msg = await peer.read_client_message(timeout=2.0)
        assert msg["method"] == "turn/start"
        peer.respond(msg["id"], {"turn": {"id": TURN_ID_2, "status": "inProgress"}})
        handle2 = await asyncio.wait_for(start2, timeout=2.0)

        consume2 = asyncio.ensure_future(_collect(handle2))
        _notify_turn_started(peer, turn_id=TURN_ID_2)
        # Turn 2's server total is cumulative (350), its last is the turn's own 250.
        _notify_usage(peer, last=_breakdown(total=250, inp=200, out=50),
                      total=_breakdown(total=350, inp=280, out=70), turn_id=TURN_ID_2)
        _notify_final_agent_message(peer, text=FINAL_JSON_TEXT_2, turn_id=TURN_ID_2,
                                    item_id="msg_round2")
        _notify_turn_completed(peer, status="completed", turn_id=TURN_ID_2)
        await asyncio.wait_for(consume2, timeout=2.0)
        outcome2 = await handle2.outcome()

        # round 2's own per-turn last.
        assert outcome2.usage["last"]["totalTokens"] == 250
        # accumulated across BOTH distinct turns: 100 + 250 = 350.
        assert outcome2.usage["total"]["totalTokens"] == 350
        assert outcome2.usage["total"]["inputTokens"] == 280  # 80 + 200
        assert outcome2.usage["total"]["outputTokens"] == 70   # 20 + 50
        assert outcome2.usage["turns_counted"] == 2
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 2b. A turn whose usage update is seen TWICE is counted ONCE (dedup).           #
# --------------------------------------------------------------------------- #

async def test_duplicate_turn_usage_update_is_not_double_counted(tmp_path):
    """The same turn emits its final ``last`` and then a TRAILING DUPLICATE update
    for the SAME (threadId,turnId) (e.g. a retry / a late re-send). The accumulated
    total must count that turn ONCE — dedup by (threadId,turnId) — not 2x."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        driver = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        # Final last for this turn = 120.
        _notify_usage(peer, last=_breakdown(total=120, inp=100, out=20),
                      total=_breakdown(total=120, inp=100, out=20))
        # A trailing DUPLICATE update for the SAME (threadId,turnId): same final
        # last re-sent. It must NOT add a second 120 to the accumulated total.
        _notify_usage(peer, last=_breakdown(total=120, inp=100, out=20),
                      total=_breakdown(total=120, inp=100, out=20))
        _notify_final_agent_message(peer, text=FINAL_JSON_TEXT)
        _notify_turn_completed(peer, status="completed")
        await asyncio.wait_for(consume, timeout=2.0)
        outcome = await handle.outcome()

        # Counted ONCE: total == 120, not 240.
        assert outcome.usage["total"]["totalTokens"] == 120
        assert outcome.usage["last"]["totalTokens"] == 120
        assert outcome.usage["turns_counted"] == 1
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 3. Completed turn with NO usage → explicit "usage unavailable" fail-fast       #
#    marker (NOT a silent usage=None that reads as zero cost).                   #
# --------------------------------------------------------------------------- #

async def test_completed_turn_without_usage_signals_unavailable(tmp_path):
    """A turn completes but emits NO ``thread/tokenUsage/updated`` at all. G5
    requires fail-fast for budgeting — so the backend must surface an explicit
    ``available: False`` marker (with a reason), NOT a bare ``usage=None`` that a
    budget layer would silently treat as zero-cost / unbounded."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        driver = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        _notify_final_agent_message(peer, text=FINAL_JSON_TEXT)
        # NO _notify_usage here — the backend could not report tokens for this turn.
        _notify_turn_completed(peer, status="completed")
        await asyncio.wait_for(consume, timeout=2.0)
        outcome = await handle.outcome()

        # The turn itself still completed (usage absence is a BUDGET fail-fast signal,
        # not a turn failure — the lead decides where the limit is enforced).
        assert outcome.status == "completed"
        # NOT a silent None: an explicit, documented unavailable marker.
        assert outcome.usage is not None
        assert outcome.usage["available"] is False
        assert "reason" in outcome.usage
        # No phantom zero breakdowns masquerading as real usage.
        assert "last" not in outcome.usage or outcome.usage.get("last") is None
        assert "total" not in outcome.usage or outcome.usage.get("total") is None
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 4. A failed turn with NO usage is ALSO unavailable (not silently None).        #
# --------------------------------------------------------------------------- #

async def test_failed_turn_without_usage_also_unavailable(tmp_path):
    """A turn that FAILS and reported no usage still carries the explicit
    ``available: False`` marker — the error path must not regress to a silent
    ``usage=None`` either (the budget layer reads the same signal regardless of
    turn outcome)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        driver = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        _notify_turn_completed(peer, status="failed",
                               error={"message": "boom", "codexErrorInfo": "fatalError"})
        await asyncio.wait_for(consume, timeout=2.0)
        outcome = await handle.outcome()

        assert outcome.status == "failed"
        assert outcome.usage is not None
        assert outcome.usage["available"] is False
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 5. A turn WITH usage that then FAILS still reports the usage it accrued.       #
# --------------------------------------------------------------------------- #

async def test_failed_turn_with_usage_reports_what_it_accrued(tmp_path):
    """A turn that consumed tokens (a usage update arrived) and THEN failed must
    still report that usage — a failed turn still cost money, so the budget signal
    must not be dropped on the failure path."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        driver = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        _notify_usage(peer, last=_breakdown(total=77, inp=60, out=17),
                      total=_breakdown(total=77, inp=60, out=17))
        _notify_turn_completed(peer, status="failed",
                               error={"message": "boom", "codexErrorInfo": "fatalError"})
        await asyncio.wait_for(consume, timeout=2.0)
        outcome = await handle.outcome()

        assert outcome.status == "failed"
        assert outcome.usage is not None
        assert outcome.usage["available"] is True
        assert outcome.usage["last"]["totalTokens"] == 77
        assert outcome.usage["total"]["totalTokens"] == 77
    finally:
        await be.aclose()
        await peer.aclose()
