from tilldone.core.enums import EventKind
from tilldone.core.events import AgentEvent


def test_event_kind_members():
    assert EventKind.RUN_COMPLETED.value == "run_completed"
    assert EventKind.TOOL_CALL.value == "tool_call"
    assert EventKind.RUN_FAILED.value == "run_failed"
    # EventKind is a str-Enum, so members compare equal to their string value
    assert EventKind.TOOL_CALL == "tool_call"
    expected = {
        "run_started", "assistant_text", "reasoning", "tool_call", "tool_result",
        "command", "file_changed", "usage", "backend_notice",
        "run_completed", "run_failed",
    }
    assert {k.value for k in EventKind} == expected


def test_event_carries_correlation_ids():
    ev = AgentEvent(
        kind=EventKind.TOOL_CALL, seq=3,
        data={"name": "return"},
        session_id="s1", thread_id="t1", turn_id="u1", item_id="i1", call_id="c1",
        raw={"native": True},
    )
    assert ev.kind is EventKind.TOOL_CALL
    assert ev.seq == 3
    assert ev.call_id == "c1" and ev.item_id == "i1"
    assert ev.raw == {"native": True}
