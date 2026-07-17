"""AR-4: concurrent tool-call unit tests.

A single model turn can emit N tool_use blocks which the backend fires concurrently
via asyncio.gather.  These tests verify that:
  - ToolRegistry routes concurrent calls correctly (no routing loss)
  - A lock-guarded stateful custom tool stays consistent under concurrency
  - ContractState accumulates all todo/return writes with no lost updates
"""
from __future__ import annotations

import asyncio

from tilldone.core.contract import (
    CompletionContract,
    ContractEvaluator,
    ContractState,
    ReturnExpectation,
    TodoExpectation,
)
from tilldone.core.registry import ToolRegistry
from tilldone.core.tools import ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Test 1 — concurrent stateful custom tool with an internal asyncio.Lock
# ---------------------------------------------------------------------------

async def test_concurrent_stateful_custom_tool_no_lost_updates():
    """Register a custom 'tick' tool whose handler increments a shared counter.

    The read-modify-write is deliberately split across an await point to expose
    concurrency; an asyncio.Lock inside the handler keeps it consistent.

    N=25 concurrent invocations must produce counter==25 and 25 non-error results.
    """
    N = 25
    counter = 0
    lock = asyncio.Lock()

    async def tick_handler(call: ToolCall) -> ToolResult:
        nonlocal counter
        async with lock:
            current = counter          # read
            await asyncio.sleep(0)     # yield — forces interleaving without the lock this would race
            counter = current + 1      # write
        return ToolResult(content=f"tick {counter}")

    state = ContractState()
    reg = ToolRegistry(state=state, contract=None)
    reg.add_custom("tick", tick_handler)

    calls = [ToolCall(f"c{i}", "tick", {}) for i in range(N)]
    results = await asyncio.gather(*[reg.invoke(c) for c in calls])

    assert counter == N, f"expected counter=={N}, got {counter}"
    assert len(results) == N
    assert all(not r.is_error for r in results), "unexpected error results"


# ---------------------------------------------------------------------------
# Test 2 — concurrent todo invocations, all recorded in ContractState
# ---------------------------------------------------------------------------

async def test_concurrent_todo_calls_all_recorded():
    """Declare todos t0..t9; fire 10 concurrent todo-mark calls; verify all land."""
    names = [f"t{i}" for i in range(10)]
    contract = CompletionContract(
        outputs=(),
        returns=(),
        todos=tuple(TodoExpectation(n) for n in names),
        require_finish=False,
    )
    state = ContractState()
    reg = ToolRegistry(state=state, contract=contract)
    reg.add_contract_tools()

    calls = [ToolCall(f"c{i}", "todo", {"name": names[i]}) for i in range(len(names))]
    results = await asyncio.gather(*[reg.invoke(c) for c in calls])

    # No results should be errors
    assert all(not r.is_error for r in results), (
        f"unexpected error results: {[r for r in results if r.is_error]}"
    )

    # All todo names must appear in state
    missing_in_state = [n for n in names if n not in state.todos_done]
    assert not missing_in_state, f"lost todo writes: {missing_in_state}"

    # Evaluator must agree: no missing todos
    evaluator = ContractEvaluator(contract)
    missing = evaluator.evaluate(state, {})
    todo_missing = [m for m in missing if m.startswith("todo:")]
    assert not todo_missing, f"evaluator reports missing todos: {todo_missing}"


# ---------------------------------------------------------------------------
# Test 3 — concurrent return invocations, all recorded in ContractState
# ---------------------------------------------------------------------------

async def test_concurrent_returns_all_recorded():
    """Declare returns r0..r9 (type_hint='int'); fire 10 concurrent return calls."""
    names = [f"r{i}" for i in range(10)]
    contract = CompletionContract(
        outputs=(),
        returns=tuple(ReturnExpectation(n, "int", required=True) for n in names),
        todos=(),
        require_finish=False,
    )
    state = ContractState()
    reg = ToolRegistry(state=state, contract=contract)
    reg.add_contract_tools()

    # Each call sets exactly one declared key to an int value equal to its index
    calls = [ToolCall(f"c{i}", "return", {names[i]: i}) for i in range(len(names))]
    results = await asyncio.gather(*[reg.invoke(c) for c in calls])

    # No results should be errors
    assert all(not r.is_error for r in results), (
        f"unexpected error results: {[r for r in results if r.is_error]}"
    )

    # All return keys must appear in state with correct values
    missing_in_state = [n for n in names if n not in state.returns]
    assert not missing_in_state, f"lost return writes: {missing_in_state}"

    wrong_values = [n for n in names if state.returns.get(n) != int(n[1:])]
    assert not wrong_values, f"wrong values for: {wrong_values}"

    # Evaluator must agree: no missing returns
    evaluator = ContractEvaluator(contract)
    missing = evaluator.evaluate(state, {})
    ret_missing = [m for m in missing if m.startswith("return:")]
    assert not ret_missing, f"evaluator reports missing returns: {ret_missing}"
