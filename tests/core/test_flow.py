"""Tests for tilldone.core.flow — RT-SDK-1 (AR-6)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tilldone.core.contract import (
    CompletionContract,
    OutputExpectation,
    ReturnExpectation,
)
from tilldone.core.enums import EventKind
from tilldone.core.flow import ActiveRunError, Flow
from tilldone.backends.fake import FakeBackend, Script, WriteFile, CallTool, Finish


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_contract(tmp_path: Path):
    """A contract with one output file and one return value."""
    out = tmp_path / "out.txt"
    contract = CompletionContract(
        outputs=(OutputExpectation("out", out, "created"),),
        returns=(ReturnExpectation("status", "str", True),),
        todos=(),
        require_finish=True,
    )
    backend = FakeBackend(Script([
        WriteFile(out, "hello"),
        CallTool("return", {"status": "ok"}),
        Finish(),
    ]))
    return contract, backend, out


def _return_only_contract():
    """A contract with only a return value (no file output)."""
    contract = CompletionContract(
        outputs=(),
        returns=(ReturnExpectation("answer", "int", True),),
        todos=(),
        require_finish=False,
    )
    backend = FakeBackend(Script([
        CallTool("return", {"answer": 42}),
    ]))
    return contract, backend


class _ClosingFakeBackend(FakeBackend):
    def __init__(self, script: Script, **kwargs) -> None:
        super().__init__(script, **kwargs)
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1


# ---------------------------------------------------------------------------
# Test 1: run() returns the expected result
# ---------------------------------------------------------------------------

async def test_run_returns_expected_result(tmp_path):
    """Flow.run() returns the same scalar result as run_task would."""
    contract, backend, _out = _simple_contract(tmp_path)
    flow = Flow(backend)
    result = await flow.run(contract, prompt="do it", cwd=tmp_path)
    assert result == "ok"


async def test_run_returns_scalar_for_single_return(tmp_path):
    """Single declared return -> scalar (not dict), matching run_task behaviour."""
    contract, backend = _return_only_contract()
    flow = Flow(backend)
    result = await flow.run(contract, prompt="compute", cwd=tmp_path)
    assert result == 42


async def test_run_returns_none_for_zero_returns(tmp_path):
    """Zero declared returns -> None."""
    out = tmp_path / "f.txt"
    contract = CompletionContract(
        outputs=(OutputExpectation("f", out, "created"),),
        returns=(),
        todos=(),
        require_finish=False,
    )
    backend = FakeBackend(Script([WriteFile(out, "content")]))
    flow = Flow(backend)
    result = await flow.run(contract, prompt="write", cwd=tmp_path)
    assert result is None


async def test_flow_async_context_manager_closes_backend(tmp_path):
    """Flow lifecycle cleanup is backend-agnostic: __aexit__ calls backend.aclose()."""

    contract = CompletionContract(
        outputs=(),
        returns=(ReturnExpectation("answer", "int", True),),
        todos=(),
        require_finish=False,
    )
    backend = _ClosingFakeBackend(Script([CallTool("return", {"answer": 42})]))

    async with Flow(backend) as flow:
        result = await flow.run(contract, prompt="compute", cwd=tmp_path)

    assert result == 42
    assert backend.close_count == 1

    # Explicit aclose remains idempotent for callers that close twice.
    await flow.aclose()
    assert backend.close_count == 1


# ---------------------------------------------------------------------------
# Test 2: run_iter() yields events and .result() gives the run() result
# ---------------------------------------------------------------------------

async def test_run_iter_yields_events_and_result_matches_run(tmp_path):
    """run_iter yields AgentEvents including run_completed, and result() == run()."""
    contract, backend_iter, _out = _simple_contract(tmp_path)

    # Build a fresh backend for the iter run (FakeBackend is stateful — same script).
    out2 = tmp_path / "out2.txt"
    contract2 = CompletionContract(
        outputs=(OutputExpectation("out2", out2, "created"),),
        returns=(ReturnExpectation("status", "str", True),),
        todos=(),
        require_finish=True,
    )
    backend2 = FakeBackend(Script([
        WriteFile(out2, "world"),
        CallTool("return", {"status": "ok"}),
        Finish(),
    ]))

    flow = Flow(backend2)
    it = flow.run_iter(contract2, prompt="go", cwd=tmp_path)

    events = []
    async for ev in it:
        events.append(ev)

    # At least one run_completed event must be present.
    assert any(ev.kind == EventKind.RUN_COMPLETED for ev in events), (
        f"No RUN_COMPLETED event in: {[ev.kind for ev in events]}"
    )

    # result() must resolve to the same value run() would return.
    result = await it.result()
    assert result == "ok"


async def test_run_iter_result_matches_run_for_int_return(tmp_path):
    """Confirm run_iter().result() == run() for an integer return."""
    contract, _b = _return_only_contract()

    # run() result via Flow
    backend_a = FakeBackend(Script([CallTool("return", {"answer": 42})]))
    flow_a = Flow(backend_a)
    run_result = await flow_a.run(contract, prompt="compute", cwd=tmp_path)

    # run_iter result
    backend_b = FakeBackend(Script([CallTool("return", {"answer": 42})]))
    flow_b = Flow(backend_b)
    it = flow_b.run_iter(contract, prompt="compute", cwd=tmp_path)
    async for _ in it:
        pass
    iter_result = await it.result()

    assert run_result == iter_result == 42


# ---------------------------------------------------------------------------
# Test 3: active-run guard — second concurrent run raises ActiveRunError
# ---------------------------------------------------------------------------

async def test_active_run_guard_run_iter_then_run(tmp_path):
    """Starting a second run while run_iter is active raises ActiveRunError."""
    out = tmp_path / "g.txt"
    contract = CompletionContract(
        outputs=(OutputExpectation("g", out, "created"),),
        returns=(),
        todos=(),
        require_finish=False,
    )
    backend = FakeBackend(Script([WriteFile(out, "data")]))
    flow = Flow(backend)

    # Obtain the run_iter iterator — this marks the flow as busy.
    it = flow.run_iter(contract, prompt="go", cwd=tmp_path)

    # Attempting a second run() before the iter is drained must raise immediately.
    with pytest.raises(ActiveRunError):
        await flow.run(contract, prompt="go again", cwd=tmp_path)

    # Drain the iterator to clean up.
    async for _ in it:
        pass
    await it.result()


async def test_active_run_guard_two_concurrent_runs(tmp_path):
    """Two concurrent run() calls on the same Flow: the second raises ActiveRunError.

    We use run_iter() to hold the lock open, then attempt run() — this is a
    deterministic version of the concurrent-run guard test that doesn't rely on
    asyncio scheduling ordering.
    """
    c = CompletionContract(
        outputs=(),
        returns=(ReturnExpectation("v", "int", True),),
        todos=(),
        require_finish=False,
    )
    backend = FakeBackend(Script([CallTool("return", {"v": 99})]))
    flow = Flow(backend)

    # run_iter acquires the guard synchronously (before any await).
    it = flow.run_iter(c, prompt="first", cwd=tmp_path)

    # The flow is now busy; run() must raise ActiveRunError immediately.
    with pytest.raises(ActiveRunError):
        await flow.run(c, prompt="second", cwd=tmp_path)

    # Clean up the running iter.
    async for _ in it:
        pass
    await it.result()


# ---------------------------------------------------------------------------
# Test 4: after a run completes the guard is released
# ---------------------------------------------------------------------------

async def test_guard_released_after_run_completes(tmp_path):
    """After run() finishes, a subsequent run() on the same Flow succeeds."""
    out1 = tmp_path / "seq1.txt"
    out2 = tmp_path / "seq2.txt"

    def make(out_path: Path):
        return (
            CompletionContract(
                outputs=(OutputExpectation("o", out_path, "created"),),
                returns=(),
                todos=(),
                require_finish=False,
            ),
            FakeBackend(Script([WriteFile(out_path, "done")])),
        )

    c1, b1 = make(out1)
    c2, b2 = make(out2)

    # Use b1 for the flow; swap backend for second run via a fresh Flow that
    # shares the guard state — actually, same flow must be reused. We can only
    # test sequential reuse since the backend is fixed per flow. Use b2 directly.
    flow1 = Flow(b1)
    await flow1.run(c1, prompt="first", cwd=tmp_path)

    # First run finished — flow should be idle. Second run on same Flow must work.
    # b1's script was consumed; create a new flow wrapping b2 to prove the
    # guard pattern, OR re-assign the backend. Since Flow.__init__ takes backend,
    # we create a second flow.
    # BUT the spec says "same Flow" — so let's use a flow that runs the same backend
    # with a re-creatable script. We'll use the fact that the contract has no outputs
    # to avoid fs issues.
    c_ret = CompletionContract(
        outputs=(),
        returns=(ReturnExpectation("v", "int", True),),
        todos=(),
        require_finish=False,
    )
    b_ret1 = FakeBackend(Script([CallTool("return", {"v": 1})]))
    b_ret2 = FakeBackend(Script([CallTool("return", {"v": 2})]))

    # We test with two separate flows to confirm the guard resets on completion.
    # For a stronger test, we build a flow that can run twice by relying on the
    # fact that the guard is an instance-level asyncio.Lock (released after each run).
    # Use the same flow object with a fresh backend each time is not possible —
    # the backend is baked in at construction. Instead we mock: use a flow whose
    # backend is b_ret1, run once; the lock must be released so a SECOND run on
    # the SAME flow object with a re-built contract also works.
    # Since b_ret1 is "consumed" after one run (all script steps executed),
    # we just confirm no exception is raised on the second attempt.
    # We create a backend that can handle two invocations by using two separate
    # script instances in a wrapper — or we just confirm that the second call
    # doesn't raise ActiveRunError (even if it may raise MaxRoundsExceeded).
    flow = Flow(b_ret1)
    r1 = await flow.run(c_ret, prompt="first", cwd=tmp_path)
    assert r1 == 1

    # Guard must be released; calling run() again should NOT raise ActiveRunError.
    # It may raise MaxRoundsExceeded since b_ret1 has no more script steps.
    try:
        await flow.run(c_ret, prompt="second", cwd=tmp_path)
    except ActiveRunError:
        pytest.fail("ActiveRunError raised after a completed run — guard was NOT released!")
    except Exception:
        pass  # MaxRoundsExceeded or similar is acceptable; guard WAS released.


async def test_guard_released_after_run_iter_completes(tmp_path):
    """After run_iter() is fully consumed, a subsequent run() on the same Flow succeeds."""
    c = CompletionContract(
        outputs=(),
        returns=(ReturnExpectation("n", "int", True),),
        todos=(),
        require_finish=False,
    )
    b1 = FakeBackend(Script([CallTool("return", {"n": 7})]))
    flow = Flow(b1)

    it = flow.run_iter(c, prompt="go", cwd=tmp_path)
    async for _ in it:
        pass
    await it.result()

    # Guard must be released now.
    b2 = FakeBackend(Script([CallTool("return", {"n": 8})]))
    # We can't swap the backend in-place, so we verify no ActiveRunError via a
    # second Flow instance sharing the same guard concept isn't possible —
    # what we CAN test is that calling run_iter again on the SAME flow doesn't
    # immediately raise ActiveRunError.
    try:
        it2 = flow.run_iter(c, prompt="go again", cwd=tmp_path)
        # Just starting iter should not raise; drain it.
        async for _ in it2:
            pass
    except ActiveRunError:
        pytest.fail("ActiveRunError after completed run_iter — guard was NOT released!")
    except Exception:
        pass  # backend exhausted — acceptable
