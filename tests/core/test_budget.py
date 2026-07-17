"""Tests for tilldone.core.budget — backend-agnostic token-budget enforcement (M7 Lane B).

Pins the G5 token-budget contract.
Scope of THIS lane (this module's scope):
  * TOKEN-count upper-limit enforcement only — dollar/cost budgets are DEFERRED.
  * It lives ABOVE ``run_task``'s backend seam (in the contract loop's view of the
    generic ``outcome.usage``), is BACKEND-AGNOSTIC (pure ``outcome.usage`` arithmetic,
    NO ``if codex_app_server`` / backend-name branch), and reads the HETEROGENEOUS usage
    shapes supported backends may emit via a small defensive accessor.
  * Missing usage (``available: False`` / usage absent / empty) is a FAIL-FAST, never a
    silent zero (treating "unknown cost" as "zero cost" would let an unbudgetable backend
    run unbounded).

The enforcement is wired into ``run_task`` via a GENERIC ``CorePolicy.max_total_tokens``
field; the 0-core-diff guard (``tests/test_zero_core_diff.py``) still passes because the
check references NO backend name — it is arithmetic over ``outcome.usage`` alone.

``asyncio_mode="auto"`` (pyproject) → async tests need no decorator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tilldone.core.budget import (
    BudgetExceeded,
    BudgetUnavailable,
    enforce_token_budget,
    total_tokens_from_usage,
)
from tilldone.core.contract import (
    CompletionContract,
    OutputExpectation,
    ReturnExpectation,
)
from tilldone.core.loop import CorePolicy, run_task
from tilldone.core.flow import Flow
from tilldone.backends.fake import FakeBackend, Script, CallTool, WriteFile, Finish


# --------------------------------------------------------------------------- #
# A FakeBackend variant whose outcome carries a (configurable) generic usage.   #
# The stock FakeBackend reports usage=None; budgeting needs a usage shape, so   #
# this thin subclass injects one WITHOUT touching the real backends.            #
# --------------------------------------------------------------------------- #

class _UsageFakeBackend(FakeBackend):
    """FakeBackend that stamps a fixed ``usage`` mapping on every outcome.

    Used to exercise the budget layer against arbitrary generic usage shapes
    (it is NOT a real backend; it just lets a test pin what ``outcome.usage`` is).
    """

    def __init__(self, *args, usage=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._usage_value = usage

    async def start(self, spec, executor):
        handle = await super().start(spec, executor)
        usage_value = self._usage_value

        class _UsageHandle:
            # Delegate events/cancel to the real handle; override outcome to attach usage.
            def __init__(self, inner):
                self._inner = inner

            @property
            def events(self):
                return self._inner.events

            async def outcome(self):
                base = await self._inner.outcome()
                # Re-shape the frozen RunOutcome with the injected usage.
                from dataclasses import replace
                return replace(base, usage=usage_value)

            async def cancel(self):
                await self._inner.cancel()

        return _UsageHandle(handle)


def _ret_contract():
    """A trivial single-return contract satisfied in one round."""
    return CompletionContract(
        outputs=(),
        returns=(ReturnExpectation("v", "int", True),),
        todos=(),
        require_finish=False,
    )


# A schema-complete structured breakdown with an explicit aggregate total.
def _structured_usage(total_tokens, *, available=True):
    if not available:
        return {"available": False, "reason": "no usage reported for this turn"}
    return {
        "available": True,
        "total": {
            "cachedInputTokens": 0,
            "inputTokens": total_tokens,
            "outputTokens": 0,
            "reasoningOutputTokens": 0,
            "totalTokens": total_tokens,
        },
        "turns_counted": 1,
    }


# A claude/codex-exec-style usage (NO ``available`` key, NO ``totalTokens`` — just the
# raw token counts the SDK / `exec --json` expose).
def _claude_usage(*, input_tokens, output_tokens):
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


# =========================================================================== #
# Part 1 — the defensive generic accessor `total_tokens_from_usage`.            #
# =========================================================================== #

class TestTotalTokensAccessor:
    """The accessor reads total tokens out of HETEROGENEOUS backend usage shapes."""

    def test_structured_total_shape_uses_total_totaltokens(self):
        usage = _structured_usage(420)
        assert total_tokens_from_usage(usage) == 420

    def test_claude_codex_shape_sums_input_output(self):
        # No totalTokens present → sum the component token counts.
        usage = _claude_usage(input_tokens=300, output_tokens=120)
        assert total_tokens_from_usage(usage) == 420

    def test_codex_camel_and_cache_fields_are_summed(self):
        # A mixed shape with cache fields but no totalTokens: sum all token-ish counts.
        usage = {"inputTokens": 100, "outputTokens": 30, "cachedInputTokens": 0}
        assert total_tokens_from_usage(usage) == 130

    def test_attribute_style_usage_object_is_supported(self):
        # claude's SDK ``msg.usage`` may be an OBJECT (attrs), not a Mapping. The accessor
        # must read it defensively rather than assume dict.
        @dataclass
        class _SdkUsage:
            input_tokens: int
            output_tokens: int

        assert total_tokens_from_usage(_SdkUsage(200, 220)) == 420

    def test_none_usage_raises_unavailable(self):
        with pytest.raises(BudgetUnavailable):
            total_tokens_from_usage(None)

    def test_available_false_raises_unavailable(self):
        with pytest.raises(BudgetUnavailable):
            total_tokens_from_usage(_structured_usage(0, available=False))

    def test_empty_usage_raises_unavailable(self):
        # codex_exec emits ``{}`` when the turn reported no token counts → fail-fast,
        # NOT a silent zero.
        with pytest.raises(BudgetUnavailable):
            total_tokens_from_usage({})

    def test_zero_tokens_is_a_real_value_not_unavailable(self):
        # An explicit, present breakdown that genuinely totals 0 is a REAL reading
        # (available), distinct from "no usage" — it must NOT raise unavailable.
        usage = _structured_usage(0)
        assert total_tokens_from_usage(usage) == 0


# =========================================================================== #
# Part 2 — `enforce_token_budget` (the pure policy primitive).                  #
# =========================================================================== #

class TestEnforcePrimitive:
    """The pure check used by the loop: under-limit passes, over-limit raises, missing
    usage fails-fast — all backend-agnostic."""

    def test_disabled_when_limit_is_none(self):
        # No budget configured → no-op, even with absent usage (don't fail-fast a run
        # that never asked to be budgeted).
        enforce_token_budget(None, None)
        enforce_token_budget({}, None)

    def test_under_limit_passes(self):
        enforce_token_budget(_structured_usage(100), 1000)

    def test_at_limit_passes(self):
        # Exactly at the limit is allowed; only EXCEEDING trips.
        enforce_token_budget(_structured_usage(1000), 1000)

    def test_over_limit_raises_with_detail(self):
        with pytest.raises(BudgetExceeded) as ei:
            enforce_token_budget(_structured_usage(1500), 1000)
        exc = ei.value
        assert exc.total_tokens == 1500
        assert exc.max_total_tokens == 1000
        # The usage detail is attached for the operator.
        assert exc.usage is not None

    def test_missing_usage_when_budgeted_fails_fast(self):
        # A budget is configured but the backend reported no usage → fail-fast.
        with pytest.raises(BudgetUnavailable):
            enforce_token_budget(None, 1000)
        with pytest.raises(BudgetUnavailable):
            enforce_token_budget(_structured_usage(0, available=False), 1000)

    def test_backend_agnostic_claude_shape(self):
        # The SAME primitive works on the claude/codex shape (no totalTokens key).
        enforce_token_budget(_claude_usage(input_tokens=400, output_tokens=400), 1000)
        with pytest.raises(BudgetExceeded):
            enforce_token_budget(_claude_usage(input_tokens=900, output_tokens=400), 1000)


# =========================================================================== #
# Part 3 — backend-agnostic: same enforcement, multiple usage shapes (param).   #
# =========================================================================== #

@pytest.mark.parametrize(
    "usage, total",
    [
        (_structured_usage(800), 800),                             # structured total shape
        (_claude_usage(input_tokens=500, output_tokens=300), 800),  # claude/codex shape
        ({"inputTokens": 600, "outputTokens": 200}, 800),          # camelCase no-total
    ],
)
def test_enforcement_is_shape_agnostic(usage, total):
    """The accessor + check yield the same answer regardless of which backend shape
    produced the usage — proving the enforcement is backend-AGNOSTIC."""
    assert total_tokens_from_usage(usage) == total
    # Under a generous limit it passes; under a tight one it trips — for ALL shapes.
    enforce_token_budget(usage, total + 1)
    with pytest.raises(BudgetExceeded):
        enforce_token_budget(usage, total - 1)


# =========================================================================== #
# Part 4 — wired through run_task / Flow via CorePolicy.max_total_tokens.        #
# =========================================================================== #

async def test_under_limit_run_completes_normally(tmp_path):
    """A run whose usage is under the budget completes and returns its result."""
    backend = _UsageFakeBackend(
        Script([CallTool("return", {"v": 7})]),
        usage=_structured_usage(100),
    )
    result = await run_task(
        _ret_contract(), backend, prompt="go", cwd=tmp_path,
        policy=CorePolicy(max_total_tokens=1000),
    )
    assert result == 7


async def test_over_limit_raises_budget_exceeded(tmp_path):
    """A run whose usage exceeds the budget raises BudgetExceeded (clean terminal)."""
    backend = _UsageFakeBackend(
        Script([CallTool("return", {"v": 7})]),
        usage=_structured_usage(5000),
    )
    with pytest.raises(BudgetExceeded) as ei:
        await run_task(
            _ret_contract(), backend, prompt="go", cwd=tmp_path,
            policy=CorePolicy(max_total_tokens=1000),
        )
    assert ei.value.total_tokens == 5000
    assert ei.value.max_total_tokens == 1000


async def test_over_limit_stops_before_further_rounds(tmp_path):
    """Over-budget trips on the FIRST round's outcome — no further contract round runs.

    The contract is NEVER satisfied (the script does nothing), so without a budget it
    would spin to max_rounds and raise MaxRoundsExceeded. WITH an over-budget usage on
    round 1, BudgetExceeded must fire first — proving the budget short-circuits the loop
    rather than letting it continue."""
    backend = _UsageFakeBackend(
        Script([]),  # never satisfies the contract
        usage=_structured_usage(5000),
    )
    with pytest.raises(BudgetExceeded):
        await run_task(
            _ret_contract(), backend, prompt="go", cwd=tmp_path,
            policy=CorePolicy(max_rounds=6, max_total_tokens=1000),
        )


async def test_missing_usage_when_budgeted_fails_fast_in_run_task(tmp_path):
    """When a budget is set but the backend reports NO usage, run_task fails fast
    (BudgetUnavailable) — it does NOT silently treat absent usage as zero cost."""
    backend = _UsageFakeBackend(
        Script([CallTool("return", {"v": 7})]),
        usage=None,  # backend reported no usage
    )
    with pytest.raises(BudgetUnavailable):
        await run_task(
            _ret_contract(), backend, prompt="go", cwd=tmp_path,
            policy=CorePolicy(max_total_tokens=1000),
        )


async def test_available_false_when_budgeted_fails_fast_in_run_task(tmp_path):
    """An explicit ``{available: False}`` marker under a budget fails fast."""
    backend = _UsageFakeBackend(
        Script([CallTool("return", {"v": 7})]),
        usage=_structured_usage(0, available=False),
    )
    with pytest.raises(BudgetUnavailable):
        await run_task(
            _ret_contract(), backend, prompt="go", cwd=tmp_path,
            policy=CorePolicy(max_total_tokens=1000),
        )


async def test_no_budget_configured_tolerates_missing_usage(tmp_path):
    """With NO budget (the default), a run with absent usage completes normally — the
    fail-fast is scoped to runs that actually asked to be budgeted."""
    backend = _UsageFakeBackend(
        Script([CallTool("return", {"v": 7})]),
        usage=None,
    )
    result = await run_task(
        _ret_contract(), backend, prompt="go", cwd=tmp_path,
        policy=CorePolicy(),  # max_total_tokens defaults to None
    )
    assert result == 7


async def test_flow_enforces_budget(tmp_path):
    """The enforcement rides through Flow (it owns CorePolicy + wraps run_task)."""
    backend = _UsageFakeBackend(
        Script([CallTool("return", {"v": 7})]),
        usage=_structured_usage(5000),
    )
    flow = Flow(backend, policy=CorePolicy(max_total_tokens=1000))
    with pytest.raises(BudgetExceeded):
        await flow.run(_ret_contract(), prompt="go", cwd=tmp_path)


async def test_flow_under_limit_completes(tmp_path):
    """Flow under-budget run returns its result normally."""
    backend = _UsageFakeBackend(
        Script([CallTool("return", {"v": 7})]),
        usage=_structured_usage(100),
    )
    flow = Flow(backend, policy=CorePolicy(max_total_tokens=1000))
    result = await flow.run(_ret_contract(), prompt="go", cwd=tmp_path)
    assert result == 7


# =========================================================================== #
# Part 5 — 0-core-diff: the generic CorePolicy.max_total_tokens check adds NO   #
# backend name to loop.py (a sibling assertion to test_zero_core_diff.py).      #
# =========================================================================== #

def test_loop_has_no_backend_name_after_budget():
    """The budget enforcement must not have introduced a backend NAME into loop.py.

    (test_zero_core_diff.test_run_task_is_backend_agnostic is the canonical guard; this
    is a local restatement so a regression in THIS lane is caught beside its own tests.)"""
    import pathlib

    loop_src = (
        pathlib.Path(__file__).resolve().parents[2]
        / "src" / "tilldone" / "core" / "loop.py"
    ).read_text(encoding="utf-8")
    for marker in (
        "app_server", "codex_app_server", "CodexAppServer", "codex_exec", "ClaudeBackend",
    ):
        assert marker not in loop_src, (
            f"loop.py references {marker!r} — the budget check must stay backend-agnostic"
        )
