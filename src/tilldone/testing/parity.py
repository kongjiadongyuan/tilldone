"""Structured, field-level, backend-labelled parity diff.

A multi-backend parity assertion that fails with a bare
``assert claude == codex`` is not observable — it never says WHICH backend
diverged, on WHICH MUST-MATCH field, or what the expected/actual values were.
This module turns a set of per-backend parity verdicts into a STRUCTURED diff so
a parity failure is debuggable (and the live ``tests/parity/test_swap.py`` raises
it instead of a bare assert).

MUST-MATCH fields only:
    returns (deep-equal value), contract_satisfied, todos_done (NAME SET — order
    independent), finished, terminal status, normalized error code, and output
    presence/validation. MAY-DIFFER axes (assistant NL, raw event shape, tokens,
    model id, session/thread id, …) are deliberately absent — comparing them would
    make parity flaky.

Pure + dependency-light: a :class:`ParityVerdict` is a plain value object a caller
builds from a :class:`~tilldone.core.spec.RunOutcome` + the post-run contract
state; ``diff_parity`` / ``assert_parity`` are pure functions over a list of them.
No I/O, no backend import — fully unit-testable in isolation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

#: The MUST-MATCH fields compared across backends, in a STABLE report order
#:. Each name is an attribute of :class:`ParityVerdict` other than the
#: ``backend`` label itself. Kept as an explicit tuple (not derived) so a new
#: ParityVerdict field is a DELIBERATE choice to compare, never an accident.
_COMPARED_FIELDS: tuple[str, ...] = (
    "returns",
    "contract_satisfied",
    "todos_done",
    "finished",
    "status",
    "error_code",
    "output_ok",
)


@dataclass(frozen=True)
class ParityVerdict:
    """One backend's MUST-MATCH facts from a parity run.

    ``backend`` is the REQUIRED label (a diff is meaningless without it). The
    remaining fields are exactly the MUST-MATCH judgement classes:

    * ``returns`` — the type-normalized return mapping (deep-equal compared).
    * ``contract_satisfied`` — did core judge the contract satisfied.
    * ``todos_done`` — the completed-todo NAME SET (a ``frozenset``; compared as a
      set so todo COMPLETION ORDER is a MAY-DIFFER, not a divergence).
    * ``finished`` — was the finish gate reached.
    * ``status`` — the deterministic terminal status (``"completed"`` / ``"failed"``).
    * ``error_code`` — the normalized :class:`~tilldone.core.spec.BackendError`
      ``code`` (``None`` on success).
    * ``output_ok`` — declared output_file exists AND passes its validator.
    """

    backend: str
    returns: Mapping[str, Any]
    contract_satisfied: bool
    todos_done: frozenset[str]
    finished: bool
    status: str
    error_code: str | None
    output_ok: bool

    def __post_init__(self) -> None:
        # A verdict with no backend label cannot be attributed in a diff — reject it
        # up-front (the test feeds backend="" / missing to prove this guard).
        if not self.backend:
            raise ValueError("ParityVerdict requires a non-empty backend label")


@dataclass(frozen=True)
class FieldDivergence:
    """One backend diverging from the consensus on one MUST-MATCH field.

    ``expected`` is the consensus (majority) value the rest of the backends agreed
    on; ``actual`` is THIS backend's value. In a no-majority split (e.g. a 2-backend
    1-vs-1 disagreement) ``expected`` is the value of the OTHER side it is being
    contrasted against, so the divergence is still concrete and never silently empty.
    """

    field: str
    backend: str
    expected: Any
    actual: Any

    def render(self) -> str:
        return (f"  - {self.backend}.{self.field}: "
                f"expected={self.expected!r} actual={self.actual!r}")


class ParityMismatch(AssertionError):
    """Raised by :func:`assert_parity` when backends disagree on a MUST-MATCH field.

    Subclasses :class:`AssertionError` so it reads as a test failure, but carries the
    STRUCTURED :attr:`diffs` (a list of :class:`FieldDivergence`) so a caller can
    inspect the divergence programmatically — not just a bare assert string.
    """

    def __init__(self, diffs: Sequence[FieldDivergence]) -> None:
        self.diffs: list[FieldDivergence] = list(diffs)
        body = "\n".join(d.render() for d in self.diffs)
        super().__init__(
            f"backend parity mismatch on {len(self.diffs)} field(s) "
            f"(MUST-MATCH):\n{body}")


def _norm(value: Any) -> Any:
    """Normalize a field value to something hashable + order-independent for the
    consensus vote. A ``todos_done`` frozenset is already order-independent; a
    ``returns`` mapping is rendered to a stable, hashable key (sorted items, recursing
    into nested mappings) so two equal-but-differently-ordered dicts vote together."""
    if isinstance(value, frozenset):
        return frozenset(value)
    if isinstance(value, Mapping):
        return tuple(sorted((k, _norm(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_norm(v) for v in value)
    return value


def _consensus(values: list[Any]) -> tuple[Any, Any]:
    """Return ``(consensus_value, consensus_key)`` — the strict-majority value (and its
    normalized key) across ``values``. With no strict majority (e.g. a 1-vs-1 split) the
    FIRST value is taken as the reference so divergences are still reported concretely."""
    keys = [_norm(v) for v in values]
    counts = Counter(keys)
    best_key, best_n = counts.most_common(1)[0]
    if best_n * 2 > len(values):  # strict majority
        idx = keys.index(best_key)
        return values[idx], best_key
    # No majority: use the first value as the reference point.
    return values[0], keys[0]


def diff_parity(verdicts: Sequence[ParityVerdict]) -> list[FieldDivergence]:
    """Compute the structured, field-level, backend-labelled parity diff.

    For each MUST-MATCH field a consensus value is chosen (strict majority, else the
    first backend as reference); every backend whose value differs from the consensus
    is reported as a :class:`FieldDivergence`. The result is EMPTY iff all backends
    agree on every field. Field order in the result follows :data:`_COMPARED_FIELDS`,
    then verdict order — deterministic so the rendered message is stable.

    Requires ≥2 verdicts (a single backend has nothing to be in parity WITH).
    """
    if len(verdicts) < 2:
        raise ValueError("parity needs at least two backend verdicts to compare")

    diffs: list[FieldDivergence] = []
    for field in _COMPARED_FIELDS:
        values = [getattr(v, field) for v in verdicts]
        consensus_value, consensus_key = _consensus(values)
        for v in verdicts:
            actual = getattr(v, field)
            if _norm(actual) != consensus_key:
                diffs.append(FieldDivergence(
                    field=field, backend=v.backend,
                    expected=consensus_value, actual=actual))
    return diffs


def assert_parity(verdicts: Sequence[ParityVerdict]) -> None:
    """Assert all backends reached a contract-equivalent verdict.

    A no-op (returns ``None``) when :func:`diff_parity` is empty; otherwise raises
    :class:`ParityMismatch` carrying the full structured diff — NEVER a bare
    ``assert a == b`` whose failure says nothing about WHICH backend/field diverged
    (the T4 observability requirement).
    """
    diffs = diff_parity(verdicts)
    if diffs:
        raise ParityMismatch(diffs)


def verdict_from_outcome(
    backend: str,
    outcome: Any,
    *,
    contract_satisfied: bool,
    todos_done: "frozenset[str] | set[str] | Sequence[str]" = frozenset(),
    finished: bool,
    output_ok: bool,
    returns: "Mapping[str, Any] | None" = None,
) -> ParityVerdict:
    """Build a :class:`ParityVerdict` from a :class:`~tilldone.core.spec.RunOutcome`
    plus the post-run contract facts the OUTCOME alone does not carry.

    ``outcome`` supplies the terminal ``status`` and (via ``outcome.error``) the
    normalized ``error_code``; the caller supplies the contract-state facts
    (``contract_satisfied`` / ``todos_done`` / ``finished`` / ``output_ok``) and,
    optionally, the type-normalized ``returns`` mapping (else ``{}``). This keeps the
    diff helper decoupled from how a caller derives those facts (run_task result,
    ContractState snapshot, …) while giving the live parity test a one-liner.
    """
    error = getattr(outcome, "error", None)
    return ParityVerdict(
        backend=backend,
        returns=dict(returns) if returns is not None else {},
        contract_satisfied=contract_satisfied,
        todos_done=frozenset(todos_done),
        finished=finished,
        status=getattr(outcome, "status", "failed"),
        error_code=getattr(error, "code", None) if error is not None else None,
        output_ok=output_ok,
    )


# Re-export the compared-field tuple length as a sanity surface for tests that want to
# assert the dataclass and the compare-list stay in lock-step (a new ParityVerdict field
# that should be compared must be added to _COMPARED_FIELDS deliberately).
def _verdict_field_names() -> tuple[str, ...]:
    return tuple(f.name for f in fields(ParityVerdict) if f.name != "backend")
