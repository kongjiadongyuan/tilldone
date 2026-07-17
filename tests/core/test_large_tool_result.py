"""AR-8: Red-team test — large tool results must NOT be silently truncated.

Three tests:
1. A custom tool returning 200_000-char content round-trips with full length + hash.
2. A contract 'return' tool submitting a 100_000-char string is stored intact in
   ContractState and satisfies ContractEvaluator.
3. A tool returning a large is_error=True result preserves both flag and content.
"""

from __future__ import annotations

import hashlib

import pytest

from tilldone.core.contract import (
    CompletionContract,
    ContractEvaluator,
    ContractState,
    ReturnExpectation,
)
from tilldone.core.registry import ToolRegistry
from tilldone.core.tools import ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Test 1 — custom tool: 200 000 chars, no truncation, hash match
# ---------------------------------------------------------------------------

async def test_large_custom_tool_result_roundtrips():
    large_content = "X" * 200_000
    source_hash = hashlib.sha256(large_content.encode()).hexdigest()

    async def big_handler(call: ToolCall) -> ToolResult:
        return ToolResult(content=large_content)

    state = ContractState()
    reg = ToolRegistry(state=state)
    reg.add_custom("big", big_handler)

    res = await reg.invoke(ToolCall("c1", "big", {}))

    assert len(res.content) == 200_000, (
        f"Registry truncated content: got {len(res.content)} chars, expected 200_000"
    )
    result_hash = hashlib.sha256(res.content.encode()).hexdigest()
    assert result_hash == source_hash, (
        "Content hash mismatch — registry corrupted the large result"
    )
    assert res.is_error is False


# ---------------------------------------------------------------------------
# Test 2 — contract 'return' tool: 100 000-char str stored intact, evaluator satisfied
# ---------------------------------------------------------------------------

async def test_large_return_value_recorded_intact():
    large_blob = "A" * 100_000
    source_hash = hashlib.sha256(large_blob.encode()).hexdigest()

    contract = CompletionContract(
        outputs=(),
        returns=(ReturnExpectation("blob", "str", True),),
        todos=(),
        require_finish=False,
    )
    state = ContractState()
    reg = ToolRegistry(state=state, contract=contract)
    reg.add_contract_tools()

    # Submit the large string via the 'return' contract tool
    res = await reg.invoke(ToolCall("c1", "return", {"blob": large_blob}))

    assert res.is_error is False, f"return tool unexpectedly rejected large str: {res.content}"

    stored = state.returns.get("blob")
    assert stored is not None, "state.returns['blob'] was not recorded"
    assert len(stored) == 100_000, (
        f"Stored blob was truncated: got {len(stored)} chars, expected 100_000"
    )
    stored_hash = hashlib.sha256(stored.encode()).hexdigest()
    assert stored_hash == source_hash, (
        "Stored blob hash mismatch — content was corrupted"
    )

    # ContractEvaluator must report the contract as satisfied
    evaluator = ContractEvaluator(contract)
    missing = evaluator.evaluate(state, {})
    assert missing == [], f"ContractEvaluator still shows unsatisfied items: {missing}"


# ---------------------------------------------------------------------------
# Test 3 — is_error=True flag survives a large result intact
# ---------------------------------------------------------------------------

async def test_large_result_is_error_flag_preserved():
    large_error_content = "E" * 150_000
    source_hash = hashlib.sha256(large_error_content.encode()).hexdigest()

    async def error_handler(call: ToolCall) -> ToolResult:
        return ToolResult(content=large_error_content, is_error=True)

    state = ContractState()
    reg = ToolRegistry(state=state)
    reg.add_custom("errtool", error_handler)

    res = await reg.invoke(ToolCall("c1", "errtool", {}))

    assert res.is_error is True, "is_error flag was not preserved on large result"
    assert len(res.content) == 150_000, (
        f"Large error content was truncated: got {len(res.content)} chars, expected 150_000"
    )
    result_hash = hashlib.sha256(res.content.encode()).hexdigest()
    assert result_hash == source_hash, (
        "Large error content hash mismatch — registry corrupted the result"
    )
