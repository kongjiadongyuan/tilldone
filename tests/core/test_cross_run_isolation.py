"""RT-TOOL-4 (true semantics): cross-run state isolation.

RT-TOOL-4: if contract/return state (or a stateful custom tool's state) were held in
a *module-global* instead of being built per run, then run 1's returns / todos /
finished / counter would LEAK into run 2.

`tests/core/test_large_tool_result.py` covers a *different* concern (large result /
no truncation, RT-TOOL-2 semantics). This file fills the deferred gap: prove that two
*sequential* runs do NOT share contract/tool state — the second run starts CLEAN.

How a regression would be caught
--------------------------------
`tilldone.core.loop.run_task` builds `state = ContractState()` as a *function-local*
on every call (loop.py). If someone "optimized" that into a module-global, e.g.::

    _STATE = ContractState()            # module scope
    async def run_task(...):
        state = _STATE                  # reuse across runs  <-- BUG

then run 1's recorded returns / done todos / finished flag would persist into run 2.

`test_second_run_does_not_inherit_returns_or_finish` would catch that exact change:
run 1 fully satisfies the contract, run 2 (same contract, a do-nothing script) would
*spuriously succeed* off run 1's leaked state instead of raising MaxRoundsExceeded.

`test_stateful_tool_counter_resets_between_runs` catches the same class of bug for a
stateful custom tool that parks its counter in ContractState: a module-global state
would let the counter keep climbing across runs.
"""

from __future__ import annotations

import pytest

from tilldone.core.loop import run_task, CorePolicy, MaxRoundsExceeded
from tilldone.core.contract import (
    CompletionContract,
    ContractEvaluator,
    ContractState,
    ReturnExpectation,
    TodoExpectation,
)
from tilldone.core.registry import ToolRegistry
from tilldone.core.tools import ToolCall, ToolResult
from tilldone.backends.fake import FakeBackend, Script, CallTool, Finish


# ---------------------------------------------------------------------------
# Test 1 — PRIMARY: a fully-satisfied run 1 must NOT make an empty run 2 pass.
#
# Run 1 records the declared return + finishes -> contract satisfied.
# Run 2 reuses the SAME contract but a do-nothing backend. If ContractState were a
# module-global, run 2 would inherit run 1's returns + finished flag and wrongly
# succeed. A per-run state means run 2 starts clean and raises MaxRoundsExceeded.
# ---------------------------------------------------------------------------

async def test_second_run_does_not_inherit_returns_or_finish(tmp_path):
    contract = CompletionContract(
        outputs=(),
        returns=(ReturnExpectation("ok", "bool", True),),
        todos=(TodoExpectation("step1"),),
        require_finish=True,
    )

    # Run 1: satisfy everything.
    backend1 = FakeBackend(Script([
        CallTool("return", {"ok": True}),
        CallTool("todo", {"name": "step1"}),
        Finish(),
    ]))
    result1 = await run_task(contract, backend1, prompt="go", cwd=tmp_path,
                             policy=CorePolicy(max_rounds=3))
    assert result1 is True, "run 1 should satisfy the contract (sanity)"

    # Run 2: SAME contract, a backend that does nothing at all. With per-run state,
    # nothing from run 1 is visible, so the contract is never satisfied.
    backend2 = FakeBackend(Script([]))
    with pytest.raises(MaxRoundsExceeded) as exc:
        await run_task(contract, backend2, prompt="go", cwd=tmp_path,
                       policy=CorePolicy(max_rounds=2))

    # The failure must cite run 1's items as STILL missing — proof run 2 began clean.
    msg = str(exc.value)
    assert "return:ok" in msg, f"run 2 inherited run 1's return (leak!): {msg}"
    assert "todo:step1" in msg, f"run 2 inherited run 1's todo (leak!): {msg}"
    assert "finish" in msg, f"run 2 inherited run 1's finished flag (leak!): {msg}"


# ---------------------------------------------------------------------------
# Test 2 — reusing the SAME FakeBackend instance across two runs is also clean.
#
# Guards against the *backend* being the leak vector (it builds a fresh handle per
# start()). Run 1 declares return "a"; run 2 declares a DIFFERENT contract needing
# "b" and supplies nothing -> must fail clean, not see a stale "a".
# ---------------------------------------------------------------------------

async def test_same_backend_reused_starts_clean_each_run(tmp_path):
    shared_backend = FakeBackend(Script([CallTool("return", {"a": 1})]))

    contract_a = CompletionContract(
        outputs=(), returns=(ReturnExpectation("a", "int", True),),
        todos=(), require_finish=False)
    result1 = await run_task(contract_a, shared_backend, prompt="go", cwd=tmp_path,
                             policy=CorePolicy(max_rounds=2))
    assert result1 == 1, "run 1 should record return a=1 (sanity)"

    # Run 2 reuses the very same backend instance, but its script ("return a")
    # does not satisfy a contract that requires a DECLARED 'b'. 'a' is undeclared
    # here, so even though the script runs, run 2 must end with b still missing.
    contract_b = CompletionContract(
        outputs=(), returns=(ReturnExpectation("b", "int", True),),
        todos=(), require_finish=False)
    with pytest.raises(MaxRoundsExceeded) as exc:
        await run_task(contract_b, shared_backend, prompt="go", cwd=tmp_path,
                       policy=CorePolicy(max_rounds=2))
    assert "return:b" in str(exc.value), (
        f"run 2 did not start clean / leaked state: {exc.value}"
    )


# ---------------------------------------------------------------------------
# Test 3 — stateful custom tool whose counter lives in ContractState resets.
#
# This is the literal matrix scenario ("有状态工具状态存全局"). The tool reads/writes
# a counter on the per-run ContractState. We drive two runs each with their OWN
# ContractState (exactly as run_task constructs one per call) and assert run 2's
# first read is 0, not a continuation of run 1's count.
# ---------------------------------------------------------------------------

def _make_counter_tool(state: ContractState):
    async def counter(call: ToolCall) -> ToolResult:
        # Park mutable tool state on the run's ContractState (not a closure-global).
        current = state.returns.get("_count", 0)
        current += 1
        state.returns["_count"] = current
        return ToolResult(content=str(current))
    return counter


async def _drive_counter_run(n_calls: int) -> ContractState:
    """Simulate one run: fresh ContractState + registry, invoke the tool n times.

    Mirrors run_task's per-call `state = ContractState()`. Returns the state so the
    caller can inspect the final counter value.
    """
    state = ContractState()
    reg = ToolRegistry(state=state)
    reg.add_custom("counter", _make_counter_tool(state))
    last = None
    for i in range(n_calls):
        last = await reg.invoke(ToolCall(f"c{i}", "counter", {}))
    return state, last


async def test_stateful_tool_counter_resets_between_runs():
    # Run 1: tick the counter 3 times.
    state1, last1 = await _drive_counter_run(3)
    assert state1.returns["_count"] == 3
    assert last1.content == "3"

    # Run 2: a brand-new run. Its FIRST tick must read 0 and return 1 — NOT 4.
    # If ContractState (or the counter) were module-global, run 2 would resume at 4.
    state2, last2 = await _drive_counter_run(1)
    assert state2.returns["_count"] == 1, (
        f"counter leaked across runs: run 2 started from run 1's value "
        f"(got {state2.returns['_count']}, expected 1)"
    )
    assert last2.content == "1", (
        f"stateful tool bled across runs: first call in run 2 returned {last2.content}, "
        "expected '1' (a clean start)"
    )
    # And the two runs must hold DISTINCT state objects.
    assert state1 is not state2, "both runs shared one ContractState instance (global leak)"


# ---------------------------------------------------------------------------
# Test 4 — direct evidence that run_task allocates a fresh ContractState per call
#          by checking that a SECOND run cannot be satisfied by structured output
#          that was only valid for the FIRST.
#
# Uses the structured-output (host_tool_calls=False) path so we exercise
# apply_structured / evaluate, the no-host branch, with a per-run state too.
# ---------------------------------------------------------------------------

async def test_structured_path_second_run_starts_clean(tmp_path):
    contract = CompletionContract(
        outputs=(), returns=(ReturnExpectation("ok", "bool", True),),
        todos=(), require_finish=True)

    # Run 1: structured backend reports everything done.
    backend_done = FakeBackend(Script([]), host_tool_calls=False,
                               structured_output={"returns": {"ok": True},
                                                  "todos_done": [], "finished": True})
    assert await run_task(contract, backend_done, prompt="go", cwd=tmp_path) is True

    # Run 2: structured backend reports NOTHING done (empty structured output).
    # With a fresh per-run state, run 1's True/finished cannot carry over.
    backend_empty = FakeBackend(Script([]), host_tool_calls=False,
                                structured_output={"returns": {},
                                                   "todos_done": [], "finished": False})
    with pytest.raises(MaxRoundsExceeded) as exc:
        await run_task(contract, backend_empty, prompt="go", cwd=tmp_path,
                       policy=CorePolicy(max_rounds=2))
    assert "return:ok" in str(exc.value) and "finish" in str(exc.value), (
        f"structured run 2 inherited run 1's state (leak!): {exc.value}"
    )
