import pytest

from tilldone.testing.conformance import (
    collect_run, assert_protocol_invariants, ScriptedEventsBackend,
)
from tilldone.backends.fake import (
    FakeBackend, Script, EmitText, WriteFile, CallTool, Finish,
)
from tilldone.core.enums import EventKind
from tilldone.core.events import AgentEvent
from tilldone.core.spec import AgentRunSpec, RunOutcome, ResumeHandle, BackendError
from tilldone.core.registry import ToolRegistry
from tilldone.core.contract import ContractState, CompletionContract, ReturnExpectation


def _contract():
    return CompletionContract(outputs=(), returns=(ReturnExpectation("ok", "bool", True),),
                              todos=(), require_finish=True)


def _reg():
    reg = ToolRegistry(ContractState(), _contract())
    reg.add_contract_tools()
    return reg


# ---- positive: the Mock (FakeBackend) is protocol-conformant in several modes ----

async def test_fake_happy_path_is_conformant(tmp_path):
    out = tmp_path / "r.md"
    backend = FakeBackend(Script([EmitText("hi"), WriteFile(out, "x"),
                                  CallTool("return", {"ok": True}), Finish()]))
    spec = AgentRunSpec(prompt="go", tools=(), cwd=tmp_path)
    events, outcome = await collect_run(backend, spec, _reg())
    assert_protocol_invariants(events, outcome)


async def test_fake_structured_is_conformant(tmp_path):
    backend = FakeBackend(Script([EmitText("hi")]), host_tool_calls=False,
                          structured_output={"returns": {"ok": True}, "todos_done": [],
                                             "finished": True})
    spec = AgentRunSpec(prompt="go", tools=(), cwd=tmp_path)
    events, outcome = await collect_run(backend, spec, _reg())
    assert_protocol_invariants(events, outcome)


async def test_fake_failed_run_is_conformant(tmp_path):
    backend = FakeBackend(Script([]),
                          fail_sequence=[BackendError(code="fatal", message="boom")])
    spec = AgentRunSpec(prompt="go", tools=(), cwd=tmp_path)
    events, outcome = await collect_run(backend, spec, _reg())
    assert outcome.status == "failed"
    assert_protocol_invariants(events, outcome)


# ---- negative: the checker has TEETH — adversarial streams must be rejected ----

def _completed():
    return RunOutcome(status="completed", resume=ResumeHandle("scripted", {"id": "1"}))


async def _check_raises(events, outcome):
    with pytest.raises(AssertionError):
        assert_protocol_invariants(events, outcome)


async def test_rejects_duplicate_terminal():
    await _check_raises(
        [AgentEvent(EventKind.RUN_COMPLETED, 1), AgentEvent(EventKind.RUN_COMPLETED, 2)],
        _completed())


async def test_rejects_nonmonotonic_seq():
    await _check_raises(
        [AgentEvent(EventKind.ASSISTANT_TEXT, 5), AgentEvent(EventKind.RUN_COMPLETED, 2)],
        _completed())


async def test_rejects_missing_terminal():
    await _check_raises([AgentEvent(EventKind.ASSISTANT_TEXT, 1)], _completed())


async def test_rejects_terminal_not_last():
    await _check_raises(
        [AgentEvent(EventKind.RUN_COMPLETED, 1), AgentEvent(EventKind.ASSISTANT_TEXT, 2)],
        _completed())


async def test_rejects_tool_result_without_call_id():
    await _check_raises(
        [AgentEvent(EventKind.TOOL_RESULT, 1), AgentEvent(EventKind.RUN_COMPLETED, 2)],
        _completed())


async def test_rejects_completed_outcome_with_failed_terminal():
    await _check_raises([AgentEvent(EventKind.RUN_FAILED, 1)], _completed())


async def test_scripted_backend_drives_through_collect_run(tmp_path):
    events = [AgentEvent(EventKind.ASSISTANT_TEXT, 1, {"text": "hi"}),
              AgentEvent(EventKind.RUN_COMPLETED, 2)]
    backend = ScriptedEventsBackend(events, _completed())
    spec = AgentRunSpec(prompt="go", tools=(), cwd=tmp_path)
    got_events, outcome = await collect_run(backend, spec, _reg())
    assert_protocol_invariants(got_events, outcome)
