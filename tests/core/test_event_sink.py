"""AR-1 — JsonlEventSink: TDD test suite.

Covers:
- sink(event) writes one JSON line per event to events_path, with kind, seq, data,
  and any non-None correlation ids, but NEVER raw.
- When messages_path is given and event.raw is not None, appends {"seq":..., "raw":...}.
- All 11 EventKind values.
- close() and context-manager support.
- Integration with run_task + FakeBackend.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tilldone.core.enums import EventKind
from tilldone.core.events import AgentEvent
from tilldone.core.event_sink import JsonlEventSink


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(kind: EventKind, seq: int, **kwargs) -> AgentEvent:
    return AgentEvent(kind=kind, seq=seq, **kwargs)


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# 1. Basic: all 11 EventKind values, raw stripped, correlation ids kept
# ---------------------------------------------------------------------------

ALL_KINDS_EVENTS: list[AgentEvent] = [
    _make_event(EventKind.RUN_STARTED,     seq=1,  data={"model": "fake"}, session_id="s1"),
    _make_event(EventKind.ASSISTANT_TEXT,  seq=2,  data={"text": "hello"}, turn_id="t1"),
    _make_event(EventKind.REASONING,       seq=3,  data={"text": "think"}, raw={"native": "reasoning_payload"}),
    _make_event(EventKind.TOOL_CALL,       seq=4,  data={"name": "bash", "args": {}}, call_id="c4", raw={"native": "tool_call_payload"}),
    _make_event(EventKind.TOOL_RESULT,     seq=5,  data={"name": "bash", "content": "ok"}, call_id="c4"),
    _make_event(EventKind.COMMAND,         seq=6,  data={"cmd": "ls"}),
    _make_event(EventKind.FILE_CHANGED,    seq=7,  data={"path": "/tmp/x.txt"}),
    _make_event(EventKind.USAGE,           seq=8,  data={"input_tokens": 100, "output_tokens": 50}),
    _make_event(EventKind.BACKEND_NOTICE,  seq=9,  data={"msg": "rate_limit"}, raw="string_raw"),
    _make_event(EventKind.RUN_COMPLETED,   seq=10, data={"final": "done"}, session_id="s1", thread_id="th1"),
    _make_event(EventKind.RUN_FAILED,      seq=11, data={"error": "timeout"}, raw={"unserializable": object()}),
]


def test_all_11_kinds_written_to_events_file(tmp_path):
    events_path = tmp_path / "events.jsonl"
    sink = JsonlEventSink(events_path)
    for ev in ALL_KINDS_EVENTS:
        sink(ev)
    sink.close()

    rows = _read_jsonl(events_path)
    assert len(rows) == 11, f"Expected 11 lines, got {len(rows)}"

    # Every row has kind (string) and seq.
    for i, (row, ev) in enumerate(zip(rows, ALL_KINDS_EVENTS)):
        assert "kind" in row, f"row {i} missing 'kind'"
        assert isinstance(row["kind"], str), f"row {i} kind not a string"
        assert row["kind"] == ev.kind.value, f"row {i} kind mismatch"
        assert row["seq"] == ev.seq, f"row {i} seq mismatch"
        assert "raw" not in row, f"row {i} must NOT contain 'raw'"


def test_events_file_has_no_raw_field(tmp_path):
    events_path = tmp_path / "events.jsonl"
    sink = JsonlEventSink(events_path)
    # Feed an event with a raw payload
    ev = _make_event(EventKind.TOOL_CALL, seq=1, data={"name": "x"}, raw={"secret": "data"})
    sink(ev)
    sink.close()

    rows = _read_jsonl(events_path)
    assert len(rows) == 1
    assert "raw" not in rows[0]


def test_correlation_ids_preserved(tmp_path):
    events_path = tmp_path / "events.jsonl"
    sink = JsonlEventSink(events_path)

    ev = _make_event(
        EventKind.TOOL_CALL, seq=3,
        data={"name": "bash"},
        session_id="sess-1", thread_id="thr-1",
        turn_id="tur-1", item_id="itm-1", call_id="cal-1",
    )
    sink(ev)
    sink.close()

    rows = _read_jsonl(events_path)
    row = rows[0]
    assert row["session_id"] == "sess-1"
    assert row["thread_id"] == "thr-1"
    assert row["turn_id"] == "tur-1"
    assert row["item_id"] == "itm-1"
    assert row["call_id"] == "cal-1"


def test_none_correlation_ids_omitted(tmp_path):
    events_path = tmp_path / "events.jsonl"
    sink = JsonlEventSink(events_path)
    ev = _make_event(EventKind.ASSISTANT_TEXT, seq=1, data={"text": "hi"})
    # All correlation ids are None by default
    sink(ev)
    sink.close()

    rows = _read_jsonl(events_path)
    row = rows[0]
    for field in ("session_id", "thread_id", "turn_id", "item_id", "call_id"):
        assert field not in row, f"None field '{field}' should not appear in output"


# ---------------------------------------------------------------------------
# 2. messages_path: raw payloads captured only when raw is not None
# ---------------------------------------------------------------------------

def test_messages_file_captures_raw_payloads(tmp_path):
    events_path = tmp_path / "events.jsonl"
    messages_path = tmp_path / "messages.jsonl"
    sink = JsonlEventSink(events_path, messages_path=messages_path)

    # 3 events: two with raw, one without
    sink(_make_event(EventKind.ASSISTANT_TEXT, seq=1, data={"text": "a"}, raw={"model": "gpt"}))
    sink(_make_event(EventKind.REASONING,      seq=2, data={"text": "b"}))           # no raw
    sink(_make_event(EventKind.TOOL_CALL,      seq=3, data={"name": "x"}, raw={"args": [1, 2]}))
    sink.close()

    assert messages_path.exists(), "messages file should be created"
    msg_rows = _read_jsonl(messages_path)
    assert len(msg_rows) == 2, "only 2 events had raw payloads"

    seqs = {r["seq"] for r in msg_rows}
    assert seqs == {1, 3}
    for r in msg_rows:
        assert "seq" in r and "raw" in r


def test_messages_file_not_created_when_no_raw(tmp_path):
    events_path = tmp_path / "events.jsonl"
    messages_path = tmp_path / "messages.jsonl"
    sink = JsonlEventSink(events_path, messages_path=messages_path)

    sink(_make_event(EventKind.ASSISTANT_TEXT, seq=1, data={"text": "a"}))  # no raw
    sink.close()

    # File should not exist (or be empty) when nothing was written
    if messages_path.exists():
        assert messages_path.read_text().strip() == "", \
            "messages file should be empty when no raw payloads"


def test_messages_file_absent_when_no_path_given(tmp_path):
    events_path = tmp_path / "events.jsonl"
    sink = JsonlEventSink(events_path)   # no messages_path

    ev = _make_event(EventKind.TOOL_CALL, seq=1, raw={"x": 1})
    sink(ev)
    sink.close()

    # No messages file should appear
    assert not (tmp_path / "messages.jsonl").exists()


def test_messages_raw_fallback_repr_for_unserializable(tmp_path):
    """When raw is not JSON-serializable, fall back to repr(raw)."""
    events_path = tmp_path / "events.jsonl"
    messages_path = tmp_path / "messages.jsonl"
    sink = JsonlEventSink(events_path, messages_path=messages_path)

    unserializable = object()
    sink(_make_event(EventKind.RUN_FAILED, seq=1, data={"error": "x"}, raw=unserializable))
    sink.close()

    msg_rows = _read_jsonl(messages_path)
    assert len(msg_rows) == 1
    row = msg_rows[0]
    assert row["seq"] == 1
    # raw should be a string (the repr)
    assert isinstance(row["raw"], str)
    assert repr(unserializable) in row["raw"]


# ---------------------------------------------------------------------------
# 3. close() and context manager
# ---------------------------------------------------------------------------

def test_close_is_idempotent(tmp_path):
    events_path = tmp_path / "events.jsonl"
    sink = JsonlEventSink(events_path)
    sink(_make_event(EventKind.RUN_STARTED, seq=1))
    sink.close()
    sink.close()  # Should not raise


def test_context_manager(tmp_path):
    events_path = tmp_path / "events.jsonl"
    messages_path = tmp_path / "messages.jsonl"

    with JsonlEventSink(events_path, messages_path=messages_path) as sink:
        sink(_make_event(EventKind.RUN_STARTED, seq=1, raw={"x": 1}))
        sink(_make_event(EventKind.RUN_COMPLETED, seq=2))

    # Files should be readable after context exit
    rows = _read_jsonl(events_path)
    assert len(rows) == 2
    assert rows[0]["kind"] == "run_started"
    assert rows[1]["kind"] == "run_completed"

    msg_rows = _read_jsonl(messages_path)
    assert len(msg_rows) == 1
    assert msg_rows[0]["seq"] == 1


def test_context_manager_files_flushed_before_exit(tmp_path):
    """Data written inside the context should be readable after __exit__."""
    events_path = tmp_path / "events.jsonl"

    with JsonlEventSink(events_path) as sink:
        for i, ev in enumerate(ALL_KINDS_EVENTS, start=1):
            sink(ev)

    rows = _read_jsonl(events_path)
    assert len(rows) == 11


# ---------------------------------------------------------------------------
# 4. Tail-friendly: each line flushed immediately
# ---------------------------------------------------------------------------

def test_lines_flushed_immediately(tmp_path):
    """A viewer tailing the file sees lines as they arrive (no buffering)."""
    events_path = tmp_path / "events.jsonl"
    sink = JsonlEventSink(events_path)

    sink(_make_event(EventKind.RUN_STARTED, seq=1))
    # Without calling close(), file should already have one line
    rows = _read_jsonl(events_path)
    assert len(rows) == 1

    sink(_make_event(EventKind.ASSISTANT_TEXT, seq=2, data={"text": "hi"}))
    rows = _read_jsonl(events_path)
    assert len(rows) == 2

    sink.close()


# ---------------------------------------------------------------------------
# 5. Integration: run_task with FakeBackend produces run_completed in file
# ---------------------------------------------------------------------------

async def test_run_task_integration(tmp_path):
    """run_task with a trivial contract writes a run_completed line to the events file."""
    from tilldone.core.contract import CompletionContract, OutputExpectation
    from tilldone.core.loop import run_task
    from tilldone.backends.fake import FakeBackend, Script, WriteFile

    # Contract: just create one output file
    out = tmp_path / "result.txt"
    contract = CompletionContract(
        outputs=(OutputExpectation("result", out, "created"),),
        returns=(),
        todos=(),
        require_finish=False,
    )
    backend = FakeBackend(Script([WriteFile(out, "done")]))

    events_path = tmp_path / "events.jsonl"
    messages_path = tmp_path / "messages.jsonl"

    with JsonlEventSink(events_path, messages_path=messages_path) as sink:
        await run_task(contract, backend, prompt="go", cwd=tmp_path, event_sink=sink)

    rows = _read_jsonl(events_path)
    assert len(rows) > 0, "No events written"

    kinds = [r["kind"] for r in rows]
    assert "run_completed" in kinds, f"Expected run_completed in {kinds}"

    # No row should contain 'raw'
    for row in rows:
        assert "raw" not in row, f"row {row} must not contain 'raw'"


# ---------------------------------------------------------------------------
# 6. public_only filter: reasoning excluded from events feed, raw still written
# ---------------------------------------------------------------------------

def test_public_only_excludes_reasoning(tmp_path):
    """With public_only=True, reasoning events are NOT written to events_path
    but their raw payload IS still written to messages_path."""
    events_path = tmp_path / "events.jsonl"
    messages_path = tmp_path / "messages.jsonl"

    reasoning_raw = {"native": "hidden_chain_of_thought"}
    events = [
        _make_event(EventKind.REASONING,      seq=1, data={"text": "think"}, raw=reasoning_raw),
        _make_event(EventKind.ASSISTANT_TEXT,  seq=2, data={"text": "hello"}),
        _make_event(EventKind.RUN_COMPLETED,   seq=3, data={"final": "done"}),
    ]

    with JsonlEventSink(events_path, messages_path=messages_path, public_only=True) as sink:
        for ev in events:
            sink(ev)

    rows = _read_jsonl(events_path)

    # The public feed must NOT contain any reasoning event.
    kinds = [r["kind"] for r in rows]
    assert "reasoning" not in kinds, f"reasoning must be absent from public feed; got {kinds}"

    # The public feed MUST contain the other event kinds.
    assert "assistant_text" in kinds, "assistant_text must be present"
    assert "run_completed" in kinds, "run_completed must be present"

    # The debug/messages layer MUST still contain the reasoning event's raw.
    assert messages_path.exists(), "messages file should be created"
    msg_rows = _read_jsonl(messages_path)
    assert len(msg_rows) == 1, f"Expected 1 raw message (reasoning), got {len(msg_rows)}"
    assert msg_rows[0]["seq"] == 1
    assert msg_rows[0]["raw"] == reasoning_raw


def test_default_includes_reasoning(tmp_path):
    """With default public_only=False, reasoning events ARE written to events_path
    (existing behaviour is preserved)."""
    events_path = tmp_path / "events.jsonl"

    events = [
        _make_event(EventKind.REASONING,     seq=1, data={"text": "think"}),
        _make_event(EventKind.ASSISTANT_TEXT, seq=2, data={"text": "hello"}),
    ]

    with JsonlEventSink(events_path) as sink:  # public_only defaults to False
        for ev in events:
            sink(ev)

    rows = _read_jsonl(events_path)
    kinds = [r["kind"] for r in rows]
    assert "reasoning" in kinds, "reasoning must appear in events file when public_only=False"
    assert len(rows) == 2, f"Expected 2 events, got {len(rows)}"
