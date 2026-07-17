"""Multi-todo + finish-gate behavior through the driver loop (RT-TODO-2 / RT-FIN-1).

Unit-level (FakeBackend): a run that completes only a subset of declared todos — even if it
'finishes' — must NOT satisfy the contract. Exact-name todos; finish does not substitute.
"""

from __future__ import annotations

import pytest

from tilldone.core.contract import CompletionContract, TodoExpectation
from tilldone.core.loop import CorePolicy, MaxRoundsExceeded, run_task
from tilldone.backends.fake import CallTool, FakeBackend, Finish, Script


async def test_partial_todos_with_finish_does_not_satisfy(tmp_path):
    contract = CompletionContract(
        outputs=(), returns=(),
        todos=(TodoExpectation("a"), TodoExpectation("b"), TodoExpectation("c")),
        require_finish=True)
    # completes only 2 of 3 todos and calls finish -> still missing todo:c -> never satisfied
    backend = FakeBackend(Script([
        CallTool("todo", {"name": "a"}),
        CallTool("todo", {"name": "b"}),
        Finish(),
    ]))
    with pytest.raises(MaxRoundsExceeded):
        await run_task(contract, backend, prompt="go", cwd=tmp_path,
                       policy=CorePolicy(max_rounds=2))


async def test_all_todos_plus_finish_satisfies(tmp_path):
    contract = CompletionContract(
        outputs=(), returns=(),
        todos=(TodoExpectation("a"), TodoExpectation("b")), require_finish=True)
    backend = FakeBackend(Script([
        CallTool("todo", {"name": "a"}),
        CallTool("todo", {"name": "b"}),
        Finish(),
    ]))
    result = await run_task(contract, backend, prompt="go", cwd=tmp_path,
                            policy=CorePolicy(max_rounds=2))
    assert result is None  # contract satisfied; zero declared returns -> None


async def test_finish_without_todos_is_insufficient(tmp_path):
    # finishing without completing the declared todo must not pass (finish != todo)
    contract = CompletionContract(
        outputs=(), returns=(), todos=(TodoExpectation("build"),), require_finish=True)
    backend = FakeBackend(Script([Finish()]))
    with pytest.raises(MaxRoundsExceeded):
        await run_task(contract, backend, prompt="go", cwd=tmp_path,
                       policy=CorePolicy(max_rounds=2))
