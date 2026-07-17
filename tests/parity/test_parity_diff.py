"""Parity failure must be OBSERVABLE (structured, field-level, backend-labelled diff).

A multi-backend parity assertion that fails with a bare ``assert a == b == c`` tells
you almost nothing — which backend diverged? on which field? expected vs actual?
We require a field-level, backend-labelled structured diff so a parity failure is
debuggable.

These are pure unit tests (no live, no backend spawn): they feed a hand-built
divergent set of per-backend parity verdicts to the structured-diff helper and
assert the diff names the diverging backend + field + expected/actual. The helper
itself is the product artifact (``tilldone.testing.parity``); the live
``tests/parity/test_swap.py`` adopts it so a real multi-backend mismatch surfaces
the structured diff instead of a bare assert.
"""

from __future__ import annotations

import pytest

from tilldone.testing.parity import (
    ParityMismatch,
    ParityVerdict,
    assert_parity,
    diff_parity,
)


# --------------------------------------------------------------------------- #
# A ParityVerdict captures one backend's MUST-MATCH facts: #
# returns (deep-equal value), contract_satisfied, todos_done (name set),        #
# finished, terminal status, error code, output presence/validation.           #
# --------------------------------------------------------------------------- #

def _verdict(backend: str, **kw) -> ParityVerdict:
    base = dict(
        backend=backend,
        returns={"answer": 4},
        contract_satisfied=True,
        todos_done=frozenset(),
        finished=True,
        status="completed",
        error_code=None,
        output_ok=True,
    )
    base.update(kw)
    return ParityVerdict(**base)


# --------------------------------------------------------------------------- #
# diff_parity — the structured, field-level, backend-labelled diff             #
# --------------------------------------------------------------------------- #

def test_all_agree_no_diff():
    """When all compared backends agree on every MUST-MATCH field, the diff is empty."""
    verdicts = [_verdict("claude"), _verdict("codex"), _verdict("codex_candidate")]
    assert diff_parity(verdicts) == []


def test_diverging_return_value_is_field_and_backend_labelled():
    """One backend returns a different value → the diff names the FIELD (returns), the
    diverging BACKEND, the EXPECTED (majority) value and the ACTUAL value."""
    verdicts = [
        _verdict("claude", returns={"answer": 4}),
        _verdict("codex", returns={"answer": 4}),
        _verdict("codex_candidate", returns={"answer": 5}),  # the odd one out
    ]
    diffs = diff_parity(verdicts)
    assert len(diffs) == 1
    d = diffs[0]
    assert d.field == "returns"
    assert d.backend == "codex_candidate"
    assert d.expected == {"answer": 4}
    assert d.actual == {"answer": 5}


def test_divergence_reported_per_field():
    """A backend diverging on TWO fields produces TWO field-level diff entries (so the
    operator sees every divergence, not just the first)."""
    verdicts = [
        _verdict("claude"),
        _verdict("codex"),
        _verdict("codex_candidate", finished=False, contract_satisfied=False),
    ]
    diffs = diff_parity(verdicts)
    fields = {d.field for d in diffs}
    assert fields == {"finished", "contract_satisfied"}
    assert all(d.backend == "codex_candidate" for d in diffs)


def test_todos_done_set_divergence():
    """The completed-todo NAME SET is a MUST-MATCH field; a backend missing a todo name
    diverges (order-independent: it is compared as a set)."""
    verdicts = [
        _verdict("claude", todos_done=frozenset({"alpha", "beta"})),
        _verdict("codex", todos_done=frozenset({"beta", "alpha"})),  # same set, diff order
        _verdict("codex_candidate", todos_done=frozenset({"alpha"})),  # missing beta
    ]
    diffs = diff_parity(verdicts)
    assert len(diffs) == 1
    assert diffs[0].field == "todos_done"
    assert diffs[0].backend == "codex_candidate"


def test_no_majority_two_two_split_still_reports_every_minority():
    """With only two backends that disagree (no majority), BOTH are reported relative to
    each other so the diff is never silently empty on a genuine mismatch."""
    verdicts = [
        _verdict("claude", returns={"answer": 4}),
        _verdict("codex", returns={"answer": 5}),
    ]
    diffs = diff_parity(verdicts)
    # Two backends, two different values → at least one divergence is reported (never []).
    assert diffs, "a genuine 2-backend mismatch must not produce an empty diff"
    assert {d.backend for d in diffs} <= {"claude", "codex"}
    assert any(d.field == "returns" for d in diffs)


# --------------------------------------------------------------------------- #
# assert_parity — raises ParityMismatch carrying the structured diff           #
# --------------------------------------------------------------------------- #

def test_assert_parity_passes_when_all_agree():
    """assert_parity is a no-op when every backend agrees (returns None, no raise)."""
    verdicts = [_verdict("claude"), _verdict("codex"), _verdict("codex_candidate")]
    assert assert_parity(verdicts) is None


def test_assert_parity_raises_structured_mismatch_not_bare_assert():
    """On divergence assert_parity raises ParityMismatch — NOT a bare AssertionError — and
    the exception carries the structured ``.diffs`` list AND a human message naming the
    backend + field + expected/actual (the T4 observability requirement)."""
    verdicts = [
        _verdict("claude", returns={"answer": 4}),
        _verdict("codex", returns={"answer": 4}),
        _verdict("codex_candidate", returns={"answer": 5}),
    ]
    with pytest.raises(ParityMismatch) as ei:
        assert_parity(verdicts)
    exc = ei.value
    # Structured payload is attached (field-level, backend-labelled).
    assert isinstance(exc.diffs, list) and len(exc.diffs) == 1
    assert exc.diffs[0].field == "returns"
    assert exc.diffs[0].backend == "codex_candidate"
    # And the rendered message is human-debuggable.
    msg = str(exc)
    assert "codex_candidate" in msg
    assert "returns" in msg
    assert "4" in msg and "5" in msg


def test_assert_parity_message_lists_every_divergence():
    """The rendered message enumerates EACH field-level divergence (multi-field case)."""
    verdicts = [
        _verdict("claude"),
        _verdict("codex"),
        _verdict("codex_candidate", finished=False, status="failed",
                 error_code="protocol"),
    ]
    with pytest.raises(ParityMismatch) as ei:
        assert_parity(verdicts)
    msg = str(ei.value)
    assert "finished" in msg
    assert "status" in msg
    assert "error_code" in msg


def test_parity_verdict_requires_a_backend_label():
    """A ParityVerdict MUST carry a backend label (the diff is meaningless without it)."""
    with pytest.raises((TypeError, ValueError)):
        ParityVerdict(  # type: ignore[call-arg]
            returns={"answer": 4}, contract_satisfied=True, todos_done=frozenset(),
            finished=True, status="completed", error_code=None, output_ok=True)
