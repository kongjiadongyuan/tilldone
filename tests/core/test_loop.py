import pytest

from tilldone.core.loop import run_task, CorePolicy, MaxRoundsExceeded, BackendFailure
from tilldone.core.contract import (
    CompletionContract, ReturnExpectation, OutputExpectation,
)
from tilldone.core.spec import BackendError
from tilldone.backends.fake import FakeBackend, Script, WriteFile, CallTool, Finish


async def test_loop_completes_single_return_is_scalar(tmp_path):
    out = tmp_path / "r.md"
    contract = CompletionContract(
        outputs=(OutputExpectation("rep", out, "created"),),
        returns=(ReturnExpectation("ok", "bool", True),),
        todos=(), require_finish=True)
    backend = FakeBackend(Script([
        WriteFile(out, "x"), CallTool("return", {"ok": True}), Finish(),
    ]))
    result = await run_task(contract, backend, prompt="go", cwd=tmp_path,
                            policy=CorePolicy(max_rounds=3))
    assert result is True   # single declared return -> scalar


async def test_loop_multiple_returns_is_dict(tmp_path):
    contract = CompletionContract(
        outputs=(), returns=(ReturnExpectation("a", "int", True),
                             ReturnExpectation("b", "int", True)),
        todos=(), require_finish=False)
    backend = FakeBackend(Script([CallTool("return", {"a": 1, "b": 2})]))
    result = await run_task(contract, backend, prompt="go", cwd=tmp_path)
    assert result == {"a": 1, "b": 2}


async def test_loop_zero_returns_is_none(tmp_path):
    out = tmp_path / "f.txt"
    contract = CompletionContract(outputs=(OutputExpectation("f", out, "created"),),
                                  returns=(), todos=(), require_finish=False)
    backend = FakeBackend(Script([WriteFile(out, "hi")]))
    result = await run_task(contract, backend, prompt="go", cwd=tmp_path)
    assert result is None


async def test_loop_raises_when_never_satisfied(tmp_path):
    contract = CompletionContract(outputs=(), returns=(ReturnExpectation("ok", "bool", True),),
                                  todos=(), require_finish=False)
    backend = FakeBackend(Script([]))  # does nothing
    with pytest.raises(MaxRoundsExceeded):
        await run_task(contract, backend, prompt="go", cwd=tmp_path,
                       policy=CorePolicy(max_rounds=2))


async def test_loop_structured_output_path(tmp_path):
    contract = CompletionContract(outputs=(), returns=(ReturnExpectation("ok", "bool", True),),
                                  todos=(), require_finish=True)
    backend = FakeBackend(Script([]), host_tool_calls=False,
                          structured_output={"returns": {"ok": True},
                                             "todos_done": [], "finished": True})
    result = await run_task(contract, backend, prompt="go", cwd=tmp_path)
    assert result is True


async def test_loop_event_sink_receives_events(tmp_path):
    out = tmp_path / "r.md"
    contract = CompletionContract(outputs=(OutputExpectation("rep", out, "created"),),
                                  returns=(), todos=(), require_finish=False)
    seen = []
    backend = FakeBackend(Script([WriteFile(out, "x")]))
    await run_task(contract, backend, prompt="go", cwd=tmp_path, event_sink=seen.append)
    assert any(ev.kind.value == "run_completed" for ev in seen)


async def test_transient_retry_does_not_consume_contract_round(tmp_path):
    out = tmp_path / "r.md"
    contract = CompletionContract(outputs=(OutputExpectation("rep", out, "created"),),
                                  returns=(), todos=(), require_finish=False)
    # One transient failure, then the retried run executes the script and succeeds —
    # all within ONE contract round (max_rounds=1 proves retry != round).
    backend = FakeBackend(Script([WriteFile(out, "x")]),
                          fail_sequence=[BackendError(code="transient", message="socket closed")])
    result = await run_task(contract, backend, prompt="go", cwd=tmp_path,
                            policy=CorePolicy(max_rounds=1, api_retry_max=2))
    assert result is None  # 0 declared returns
    assert out.read_text() == "x"


async def test_fatal_error_raises_immediately(tmp_path):
    contract = CompletionContract(outputs=(), returns=(), todos=(), require_finish=False)
    backend = FakeBackend(Script([]),
                          fail_sequence=[BackendError(code="fatal", message="boom")])
    with pytest.raises(BackendFailure):
        await run_task(contract, backend, prompt="go", cwd=tmp_path,
                       policy=CorePolicy(max_rounds=3, api_retry_max=2))


async def test_transient_exhausts_retries_then_raises(tmp_path):
    contract = CompletionContract(outputs=(), returns=(), todos=(), require_finish=False)
    backend = FakeBackend(Script([]),
                          fail_sequence=[BackendError(code="transient", message="x")] * 10)
    with pytest.raises(BackendFailure):
        await run_task(contract, backend, prompt="go", cwd=tmp_path,
                       policy=CorePolicy(max_rounds=2, api_retry_max=2))
