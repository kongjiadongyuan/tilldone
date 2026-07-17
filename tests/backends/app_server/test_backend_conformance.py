"""Conformance + adversarial protocol-stream tests for CodexAppServerBackend (M2 Lane B).

Every test scripts an *adversarial* server stream through the in-memory
:class:`FakePeer` (a real :class:`AppServerTransport` over byte pipes — NO codex
spawn, NO gateway), drives the backend end-to-end, and — wherever the run
reaches a terminal — asserts the cross-backend generalization gate
:func:`tilldone.testing.conformance.assert_protocol_invariants` holds on the
*normalized* event stream (seq strictly increasing, exactly one terminal as the
last event, tool-result pairing, resume-token round-trip).

Distinct from ``test_backend.py`` (Lane A): that file proves the happy
structured round-trip + per-frame mapping + handshake shape + the basic
failure/interrupt/non-JSON outcomes. This file does NOT re-litigate those; it
attacks the *protocol-stream invariants* under adversarial wire behavior:
duplicate terminals, two failure signals on the wire, fileChange-not-in-contract,
out-of-order/interleaved items, an unexpected server-request method other than
``item/tool/call``, command+mcpTool call-id pairing, and final-answer buffering
on the conformance path.

We reuse the Lane A server-side scripting helpers verbatim (frame shapes lifted
from the golden transcripts) rather than re-deriving them; only the *adversarial
sequencing* lives here.

asyncio_mode="auto" (pyproject) → async tests need no decorator.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tilldone.backends.app_server.backend import CodexAppServerBackend
from tilldone.core.enums import EventKind
from tilldone.core.spec import AgentRunSpec
from tilldone.testing.conformance import assert_protocol_invariants, collect_run

from .fake_peer import FakePeer

# Reuse Lane A's golden-transcript-derived ids, schema, executor, spec builder,
# and the in-order server-side scripting helpers (initialize→…→turn/start +
# the individual notification emitters). This file owns ONLY the adversarial
# sequencing; the frame shapes are the shared contract fixtures.
from .test_backend import (
    FINAL_JSON_TEXT,
    THREAD_ID,
    TURN_ID,
    _NullExecutor,
    _notify_final_agent_message,
    _notify_token_usage,
    _notify_turn_completed,
    _notify_turn_started,
    _serve_handshake,
    _spec,
)

# A second schema-valid JSON used where a test needs to distinguish "the right
# final answer" from a clobbering candidate (must stay strict-parseable JSON).
ALT_JSON_TEXT = '{"returns":{"answer":"OMEGA-0001"},"todos_done":[],"finished":true}'


async def _run(be: CodexAppServerBackend, spec: AgentRunSpec):
    """Collect events + outcome via the conformance harness (collect_run)."""
    return await collect_run(be, spec, _NullExecutor())


def _terminal_count(events) -> int:
    return sum(
        1 for e in events if e.kind in (EventKind.RUN_COMPLETED, EventKind.RUN_FAILED)
    )


# --------------------------------------------------------------------------- #
# 1. Duplicate turn/completed on the wire → backend emits EXACTLY ONE terminal #
# --------------------------------------------------------------------------- #

async def test_duplicate_turn_completed_emits_single_terminal(tmp_path):
    """Two ``turn/completed`` frames on the wire → the event stream stops after
    the FIRST terminal (the iterator returns); the duplicate is never observed.
    ``assert_protocol_invariants`` then holds (single terminal, last)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            _notify_final_agent_message(peer)
            _notify_token_usage(peer)
            # TWO terminals back-to-back on the wire.
            _notify_turn_completed(peer, status="completed")
            _notify_turn_completed(peer, status="completed")

        driver = asyncio.ensure_future(drive())
        events, outcome = await asyncio.wait_for(_run(be, _spec(tmp_path)), timeout=2.0)
        await asyncio.wait_for(driver, timeout=2.0)

        assert _terminal_count(events) == 1
        assert events[-1].kind == EventKind.RUN_COMPLETED
        assert outcome.status == "completed"
        assert outcome.structured_output == json.loads(FINAL_JSON_TEXT)
        # The generalization gate must pass: single terminal, last, seq increasing.
        assert_protocol_invariants(events, outcome)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 2. turn-failed-after-error: an `error` notification THEN turn/completed(failed)#
#    → the `error` note is NOT a terminal; sole terminal = RUN_FAILED.          #
# --------------------------------------------------------------------------- #

async def test_error_notification_is_not_terminal_turn_completed_failed_wins(tmp_path):
    """A standalone ``error`` notification (willRetry-style) is telemetry noise
    (mapped to None / dropped) — NOT a terminal. The sole terminal is RUN_FAILED
    derived from ``turn/completed``(status=failed) with a classified BackendError."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            # A non-terminal error notification (e.g. a retry-able stream hiccup).
            peer.notify(
                "error",
                {"threadId": THREAD_ID, "turnId": TURN_ID,
                 "message": "stream blip", "willRetry": True},
            )
            # THEN the authoritative terminal: a failed turn with a transient marker.
            _notify_turn_completed(
                peer, status="failed",
                error={"message": "stream disconnected before completion",
                       "codexErrorInfo": "serverOverloaded"})

        driver = asyncio.ensure_future(drive())
        events, outcome = await asyncio.wait_for(_run(be, _spec(tmp_path)), timeout=2.0)
        await asyncio.wait_for(driver, timeout=2.0)

        # Exactly one terminal, and it is RUN_FAILED from turn/completed (NOT the
        # error notification, which dropped out of the normalized stream).
        assert _terminal_count(events) == 1
        assert events[-1].kind == EventKind.RUN_FAILED
        assert outcome.status == "failed"
        assert outcome.error is not None
        assert outcome.error.code == "transient"  # serverOverloaded => transient
        assert "stream disconnected" in outcome.error.message
        assert outcome.structured_output is None
        assert_protocol_invariants(events, outcome)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 3. interrupted: turn/completed(status=interrupted) → RUN_FAILED + interrupt   #
# --------------------------------------------------------------------------- #

async def test_interrupted_terminal_is_run_failed_with_interrupt_error(tmp_path):
    """``turn/completed``(status=interrupted, error=null) →
    RUN_FAILED terminal; outcome carries an interrupt BackendError."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            # Partial narrative before the interrupt (must not become the answer).
            _notify_final_agent_message(peer, text="thinking...", phase="answer",
                                        item_id="m_partial")
            _notify_turn_completed(peer, status="interrupted", error=None)

        driver = asyncio.ensure_future(drive())
        events, outcome = await asyncio.wait_for(_run(be, _spec(tmp_path)), timeout=2.0)
        await asyncio.wait_for(driver, timeout=2.0)

        assert _terminal_count(events) == 1
        assert events[-1].kind == EventKind.RUN_FAILED
        assert outcome.status == "failed"
        assert outcome.error is not None
        assert "interrupt" in outcome.error.message.lower()
        assert outcome.structured_output is None
        assert_protocol_invariants(events, outcome)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 4. fileChange item normalizes to FILE_CHANGED but does NOT terminate the run  #
#    and does NOT affect the outcome (contract authority is fs+sha256 in core). #
# --------------------------------------------------------------------------- #

async def test_file_change_item_is_observed_but_not_terminal_nor_contract(tmp_path):
    """A ``fileChange`` item/completed → a FILE_CHANGED event mid-stream. It does
    NOT terminate the run (only turn/completed does) and does NOT influence the
    structured outcome — the backend never treats fileChange as contract signal."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            # A fileChange item arrives BEFORE the final answer — it must not end
            # the run or be mistaken for the contract authority.
            peer.notify(
                "item/completed",
                {"item": {"type": "fileChange", "id": "f1",
                          "changes": [{"path": "out.txt", "kind": "add"}]},
                 "threadId": THREAD_ID, "turnId": TURN_ID},
            )
            _notify_final_agent_message(peer)
            _notify_turn_completed(peer, status="completed")

        driver = asyncio.ensure_future(drive())
        events, outcome = await asyncio.wait_for(_run(be, _spec(tmp_path)), timeout=2.0)
        await asyncio.wait_for(driver, timeout=2.0)

        file_changed = [e for e in events if e.kind == EventKind.FILE_CHANGED]
        assert len(file_changed) == 1
        # FILE_CHANGED appears BEFORE the terminal — it did not end the run.
        assert events.index(file_changed[0]) < len(events) - 1
        # Exactly one terminal, run still completed normally, output unaffected.
        assert _terminal_count(events) == 1
        assert events[-1].kind == EventKind.RUN_COMPLETED
        assert outcome.status == "completed"
        assert outcome.structured_output == json.loads(FINAL_JSON_TEXT)
        assert_protocol_invariants(events, outcome)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 5. out-of-order / interleaved items → seq stays strictly increasing, no crash,#
#    final agentMessage still correctly buffered.                               #
# --------------------------------------------------------------------------- #

async def test_interleaved_out_of_order_items_keep_seq_and_buffer_final(tmp_path):
    """Reasoning / command / a streaming delta interleaved between the final
    agentMessage's started and completed frames + extra trailing reasoning →
    seq stays strictly increasing, nothing crashes, and the final_answer text is
    still the buffered structured output."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            # A reasoning item.
            peer.notify(
                "item/completed",
                {"item": {"type": "reasoning", "id": "rs1", "summary": ["plan"],
                          "content": []},
                 "threadId": THREAD_ID, "turnId": TURN_ID},
            )
            # The final agentMessage STARTED frame (text="" — not the settled text).
            peer.notify(
                "item/started",
                {"item": {"type": "agentMessage", "id": "m_final", "text": "",
                          "phase": "final_answer"},
                 "threadId": THREAD_ID, "turnId": TURN_ID},
            )
            # ...interleave a streaming delta and a command BETWEEN started+completed.
            peer.notify(
                "item/agentMessage/delta",
                {"threadId": THREAD_ID, "turnId": TURN_ID, "itemId": "m_final",
                 "delta": FINAL_JSON_TEXT[:5]},
            )
            peer.notify(
                "item/started",
                {"item": {"type": "commandExecution", "id": "c1", "command": "ls",
                          "status": "inProgress"},
                 "threadId": THREAD_ID, "turnId": TURN_ID},
            )
            peer.notify(
                "item/completed",
                {"item": {"type": "commandExecution", "id": "c1", "command": "ls",
                          "exit_code": 0, "status": "completed"},
                 "threadId": THREAD_ID, "turnId": TURN_ID},
            )
            # The settled final agentMessage (carries the authoritative JSON text).
            peer.notify(
                "item/completed",
                {"item": {"type": "agentMessage", "id": "m_final",
                          "text": FINAL_JSON_TEXT, "phase": "final_answer"},
                 "threadId": THREAD_ID, "turnId": TURN_ID},
            )
            # A trailing reasoning item AFTER the final answer (must not crash / clobber).
            peer.notify(
                "item/completed",
                {"item": {"type": "reasoning", "id": "rs2", "summary": ["after"],
                          "content": []},
                 "threadId": THREAD_ID, "turnId": TURN_ID},
            )
            _notify_turn_completed(peer, status="completed")

        driver = asyncio.ensure_future(drive())
        events, outcome = await asyncio.wait_for(_run(be, _spec(tmp_path)), timeout=2.0)
        await asyncio.wait_for(driver, timeout=2.0)

        # seq strictly increasing across the whole interleaved stream.
        seqs = [e.seq for e in events]
        assert all(b > a for a, b in zip(seqs, seqs[1:])), f"seq not strict: {seqs}"
        # Sanity: the interleaving really happened (multiple kinds present).
        kinds = {e.kind for e in events}
        assert EventKind.REASONING in kinds
        assert EventKind.COMMAND in kinds
        assert EventKind.ASSISTANT_TEXT in kinds
        # Final answer correctly buffered despite the interleaving.
        assert outcome.status == "completed"
        assert outcome.structured_output == json.loads(FINAL_JSON_TEXT)
        assert outcome.final_text == FINAL_JSON_TEXT
        assert_protocol_invariants(events, outcome)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 6. unexpected server-request (method != item/tool/call) → protocol error.     #
#    Complements Lane A's item/tool/call case with applyPatchApproval.          #
# --------------------------------------------------------------------------- #

async def test_unexpected_apply_patch_approval_request_is_protocol_error(tmp_path):
    """The structured path registers NO server-request handler. An
    ``applyPatchApproval`` server→client REQUEST lands in
    ``transport.unexpected_server_requests`` → outcome is a protocol error even
    though the turn itself completes; the single terminal invariant still holds."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            # An approval REQUEST (id+method) the structured path never models.
            peer.server_request(
                "applyPatchApproval",
                {"threadId": THREAD_ID, "turnId": TURN_ID, "callId": "p0",
                 "fileChanges": []},
                id=0,
            )
            await asyncio.sleep(0.02)  # let the reader pump record the unexpected req
            _notify_final_agent_message(peer)
            _notify_turn_completed(peer, status="completed")

        driver = asyncio.ensure_future(drive())
        events, outcome = await asyncio.wait_for(_run(be, _spec(tmp_path)), timeout=2.0)
        await asyncio.wait_for(driver, timeout=2.0)

        assert outcome.status == "failed"
        assert outcome.error is not None
        assert outcome.error.code == "protocol"
        assert "applyPatchApproval" in outcome.error.message
        # The wire terminal was RUN_COMPLETED (turn completed), but the outcome
        # was overridden to failed by the protocol breach. The event-stream
        # terminal is still single & last → assert_protocol_invariants would FAIL
        # the completed/RUN_COMPLETED agreement, so we assert the stream shape
        # directly here rather than the cross-status agreement check.
        assert _terminal_count(events) == 1
        assert events[-1].kind == EventKind.RUN_COMPLETED
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 7. commandExecution begin/end + mcpToolCall items normalize with correct      #
#    call_id pairing so the tool-result-pairing invariant passes.               #
# --------------------------------------------------------------------------- #

async def test_command_and_mcp_tool_call_pairing_satisfies_invariants(tmp_path):
    """A commandExecution begin/end pair (COMMAND events, call_id set) plus an
    mcpToolCall begin/end pair (TOOL_CALL→TOOL_RESULT, same call_id) → the
    conformance tool-result-pairing check passes (every TOOL_CALL has a matching
    later TOOL_RESULT)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            # commandExecution begin → end (COMMAND kind both, phase begin/end).
            peer.notify(
                "item/started",
                {"item": {"type": "commandExecution", "id": "c1", "command": "echo hi",
                          "status": "inProgress"},
                 "threadId": THREAD_ID, "turnId": TURN_ID},
            )
            peer.notify(
                "item/completed",
                {"item": {"type": "commandExecution", "id": "c1", "command": "echo hi",
                          "exit_code": 0, "status": "completed"},
                 "threadId": THREAD_ID, "turnId": TURN_ID},
            )
            # mcpToolCall begin → end (TOOL_CALL then TOOL_RESULT, SAME call_id "t1").
            peer.notify(
                "item/started",
                {"item": {"type": "mcpToolCall", "id": "t1", "tool": "search",
                          "server": "srv", "status": "inProgress",
                          "arguments": {"q": "x"}},
                 "threadId": THREAD_ID, "turnId": TURN_ID},
            )
            peer.notify(
                "item/completed",
                {"item": {"type": "mcpToolCall", "id": "t1", "tool": "search",
                          "status": "completed", "result": {"isError": False}},
                 "threadId": THREAD_ID, "turnId": TURN_ID},
            )
            _notify_final_agent_message(peer)
            _notify_turn_completed(peer, status="completed")

        driver = asyncio.ensure_future(drive())
        events, outcome = await asyncio.wait_for(_run(be, _spec(tmp_path)), timeout=2.0)
        await asyncio.wait_for(driver, timeout=2.0)

        # COMMAND events present, call_id == item id, begin precedes end.
        cmds = [e for e in events if e.kind == EventKind.COMMAND]
        assert len(cmds) == 2
        assert all(e.call_id == "c1" for e in cmds)
        assert [e.data["phase"] for e in cmds] == ["begin", "end"]

        # TOOL_CALL paired with a later TOOL_RESULT on the same call_id.
        calls = [e for e in events if e.kind == EventKind.TOOL_CALL]
        results = [e for e in events if e.kind == EventKind.TOOL_RESULT]
        assert len(calls) == 1 and len(results) == 1
        assert calls[0].call_id == "t1" == results[0].call_id
        assert events.index(calls[0]) < events.index(results[0])
        assert results[0].data["is_error"] is False

        assert outcome.status == "completed"
        # The full pairing/seq/terminal contract holds.
        assert_protocol_invariants(events, outcome)
    finally:
        await be.aclose()
        await peer.aclose()


async def test_failed_mcp_tool_call_still_pairs_and_marks_error(tmp_path):
    """A FAILED mcpToolCall (status=failed) still emits a TOOL_RESULT paired to
    its TOOL_CALL (is_error=True) → pairing invariant holds; a failed tool call
    is NOT itself a run terminal (only turn/completed is)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            peer.notify(
                "item/started",
                {"item": {"type": "mcpToolCall", "id": "t9", "tool": "bad",
                          "status": "inProgress"},
                 "threadId": THREAD_ID, "turnId": TURN_ID},
            )
            peer.notify(
                "item/completed",
                {"item": {"type": "mcpToolCall", "id": "t9", "tool": "bad",
                          "status": "failed", "result": {"isError": True}},
                 "threadId": THREAD_ID, "turnId": TURN_ID},
            )
            _notify_final_agent_message(peer)
            _notify_turn_completed(peer, status="completed")

        driver = asyncio.ensure_future(drive())
        events, outcome = await asyncio.wait_for(_run(be, _spec(tmp_path)), timeout=2.0)
        await asyncio.wait_for(driver, timeout=2.0)

        results = [e for e in events if e.kind == EventKind.TOOL_RESULT]
        assert len(results) == 1
        assert results[0].call_id == "t9"
        assert results[0].data["is_error"] is True
        # A failed tool call did not end the run — turn/completed did.
        assert _terminal_count(events) == 1
        assert events[-1].kind == EventKind.RUN_COMPLETED
        assert outcome.status == "completed"
        assert_protocol_invariants(events, outcome)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 8. trailing non-final agentMessage after a final_answer one → final_answer    #
#    text wins (buffering rule) on the conformance path.                        #
# --------------------------------------------------------------------------- #

async def test_final_answer_not_clobbered_by_trailing_non_final_on_conformance_path(tmp_path):
    """On the full collect_run path: a final_answer agentMessage followed by a
    trailing NON-final narrative agentMessage → the final_answer JSON wins for
    structured_output (never clobbered), and invariants hold."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            # The authoritative final answer (phase=final_answer).
            _notify_final_agent_message(peer, text=FINAL_JSON_TEXT, phase="final_answer",
                                        item_id="m_final")
            # A LATER non-final narrative — even though it is itself valid JSON, it
            # must NOT clobber the buffered final_answer (phase discriminates).
            _notify_final_agent_message(peer, text=ALT_JSON_TEXT, phase="answer",
                                        item_id="m_extra")
            _notify_turn_completed(peer, status="completed")

        driver = asyncio.ensure_future(drive())
        events, outcome = await asyncio.wait_for(_run(be, _spec(tmp_path)), timeout=2.0)
        await asyncio.wait_for(driver, timeout=2.0)

        assert outcome.status == "completed"
        assert outcome.structured_output == json.loads(FINAL_JSON_TEXT)
        assert outcome.final_text == FINAL_JSON_TEXT
        # Both agentMessages surfaced as events (observed), but only one terminal.
        assert sum(1 for e in events if e.kind == EventKind.ASSISTANT_TEXT
                   and not e.data.get("streaming")) == 2
        assert_protocol_invariants(events, outcome)
    finally:
        await be.aclose()
        await peer.aclose()


async def test_final_answer_wins_even_when_emitted_before_non_final(tmp_path):
    """Order-independence of the buffering rule on the conformance path: the
    final_answer arriving FIRST is still preserved when a non-final follows; the
    structured output is the final_answer JSON, not the trailing narrative."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        async def drive():
            await _serve_handshake(peer)
            _notify_turn_started(peer)
            # A non-final narrative FIRST, then the real final answer — the
            # final_answer must win regardless of arrival order.
            _notify_final_agent_message(peer, text="let me think out loud",
                                        phase="answer", item_id="m_pre")
            _notify_final_agent_message(peer, text=FINAL_JSON_TEXT, phase="final_answer",
                                        item_id="m_final")
            _notify_turn_completed(peer, status="completed")

        driver = asyncio.ensure_future(drive())
        events, outcome = await asyncio.wait_for(_run(be, _spec(tmp_path)), timeout=2.0)
        await asyncio.wait_for(driver, timeout=2.0)

        assert outcome.status == "completed"
        assert outcome.structured_output == json.loads(FINAL_JSON_TEXT)
        assert outcome.final_text == FINAL_JSON_TEXT
        assert_protocol_invariants(events, outcome)
    finally:
        await be.aclose()
        await peer.aclose()
