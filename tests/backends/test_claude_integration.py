"""ClaudeBackend LIVE integration tests (env-gated).

Run with a real Claude credential:  TILLDONE_CLAUDE_E2E=1 uv run pytest -m integration -v

DoD: these are NOT skipped-as-passed. When the gate is on they must actually execute
(T1: a skip is not a pass). They make real (billed) Claude calls through the ambient
credential, so they are kept minimal and single-round.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tilldone.backends.claude import ClaudeBackend
from tilldone.core.contract import (
    CompletionContract,
    ContractState,
    OutputExpectation,
    ReturnExpectation,
)
from tilldone.core.loop import CorePolicy, run_task
from tilldone.core.registry import ToolRegistry
from tilldone.core.spec import AgentRunSpec
from tilldone.core.tools import ToolCall, ToolResult
from tilldone.core.validators import CallableValidator
from tilldone.testing.conformance import assert_protocol_invariants, collect_run

pytestmark = pytest.mark.integration

_GATE = os.environ.get("TILLDONE_CLAUDE_E2E") == "1"
_skip = pytest.mark.skipif(not _GATE, reason="set TILLDONE_CLAUDE_E2E=1 to run live Claude tests")


class _StubExecutor:
    async def invoke(self, call: ToolCall) -> ToolResult:
        return ToolResult(content="ok", is_error=False)


@_skip
async def test_claude_e2e_contract_completion(tmp_path: Path):
    """Full stack: core loop + ClaudeBackend + in-process MCP tools + real Claude.

    Proves the contract loop completes against live Claude AND that the MCP bridge
    actually fired under bypassPermissions (the returned value can ONLY appear if the
    `return` MCP tool was really invoked and recorded into ContractState).
    """
    report = tmp_path / "report.txt"
    contract = CompletionContract(
        outputs=(OutputExpectation("report", report, "created",
                                   validator=CallableValidator(lambda b: b"DONE" in b)),),
        returns=(ReturnExpectation("answer", "int", required=True),),
        todos=(),
        require_finish=True,
    )
    prompt = (
        "Complete this task in a single turn, performing ALL of the following:\n"
        f"1. Create a file at the absolute path {report} whose exact contents are: DONE\n"
        "2. Call the 'return' tool with argument answer set to the integer 4 (the value of 2+2).\n"
        "3. Call the 'finish' tool.\n"
        "Do all three steps; do not ask for confirmation."
    )
    events = []
    result = await run_task(
        contract, ClaudeBackend(), prompt=prompt, cwd=tmp_path,
        policy=CorePolicy(max_rounds=4), event_sink=events.append,
    )

    assert result == 4, f"expected scalar 4 from the single declared return, got {result!r}"
    assert report.exists() and b"DONE" in report.read_bytes()
    # MCP bridge really fired: at least one normalized tool_call surfaced.
    assert any(e.kind.value == "tool_call" for e in events), "no tool_call events — MCP bridge did not fire"
    assert any(e.kind.value == "run_completed" for e in events)


@_skip
async def test_claude_e2e_protocol_conformance(tmp_path: Path):
    """Connect the T2 conformance checker to a REAL Claude event stream (single turn)."""
    spec = AgentRunSpec(prompt="Reply with the single word ACK and nothing else.",
                        tools=(), cwd=tmp_path)
    events, outcome = await collect_run(ClaudeBackend(), spec, _StubExecutor())
    assert_protocol_invariants(events, outcome)
    assert outcome.status == "completed"
    assert outcome.resume is not None and outcome.resume.backend == "claude"
    assert outcome.resume.token.get("session_id")
