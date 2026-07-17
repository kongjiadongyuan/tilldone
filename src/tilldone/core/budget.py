"""tilldone.core.budget — backend-agnostic token-budget enforcement.

This is the Flow/policy-layer budget primitive: it enforces a TOKEN-count upper limit
by reading the GENERIC ``RunOutcome.usage`` — pure arithmetic over the usage mapping,
with NO backend-name branch. It sits ABOVE ``run_task``'s ``backend.start`` seam (the
contract loop applies it to each round's outcome) so it works identically for every
backend that populates ``outcome.usage``.

Scope
-----
* **Token-count limits ONLY.** Precise dollar/cost budgets are not implemented here;
  this module never reasons about price — only token totals.
* **Backend-agnostic.** Heterogeneous usage shapes (app-server's
  ``{available, total:{totalTokens,...}}`` vs claude/codex-exec's
  ``{input_tokens, output_tokens, ...}``) are read via one defensive accessor,
  :func:`total_tokens_from_usage`. There is NO backend-name branch anywhere.
* **Missing usage fails fast.** A run that is being budgeted but reports no usage
  (``available: False``, ``usage is None``, or an empty/no-token-count mapping) must NOT
  be treated as zero cost — that would let an unbudgetable backend run unbounded. The
  accessor raises :class:`BudgetUnavailable`.

Terminal states
---------------
* :class:`BudgetExceeded` — cumulative/total tokens exceeded the configured limit.
  Carries ``total_tokens``, ``max_total_tokens`` and the raw ``usage`` detail; raised
  from the loop it stops the run with no further round.
* :class:`BudgetUnavailable` — a budget was configured but the backend reported no
  usable usage to enforce against.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The token-count field names a usage breakdown may carry. We read total tokens
#: defensively across the heterogeneous backend shapes:
#:   * app-server breakdown → ``totalTokens`` is the authoritative per-breakdown total;
#:   * claude SDK usage / codex-exec ``exec --json`` usage → NO ``totalTokens``, only the
#:     component counts (snake_case ``input_tokens``/``output_tokens``[/cache], and the
#:     camelCase variants for safety) which we SUM.
#: Listing both an explicit-total field and the component fields lets the accessor prefer
#: a backend's own total when present and otherwise sum the parts — without ever naming a
#: backend.
_EXPLICIT_TOTAL_FIELDS = ("totalTokens", "total_tokens")
_COMPONENT_TOKEN_FIELDS = (
    "inputTokens", "input_tokens",
    "outputTokens", "output_tokens",
    "cachedInputTokens", "cache_read_input_tokens", "cache_creation_input_tokens",
    "reasoningOutputTokens", "reasoning_output_tokens",
)


class BudgetExceeded(RuntimeError):
    """Raised when a run's cumulative/total token usage exceeds the configured limit.

    A CLEAN terminal state: it stops the run (no further contract round) and carries the
    detail an operator needs. ``usage`` is the raw generic ``outcome.usage`` mapping that
    tripped the limit.
    """

    def __init__(self, *, total_tokens: int, max_total_tokens: int,
                 usage: Any = None) -> None:
        self.total_tokens = total_tokens
        self.max_total_tokens = max_total_tokens
        self.usage = usage
        super().__init__(
            f"token budget exceeded: {total_tokens} total tokens > limit "
            f"{max_total_tokens} (usage={usage!r})"
        )


class BudgetUnavailable(RuntimeError):
    """Raised when a budget is configured but the backend reported NO usage to enforce
    against (``available: False`` / ``usage is None`` / an empty / no-token-count
    mapping).

    G5: a missing usage must NOT be silently treated as zero cost — an unbudgetable
    backend that reports no tokens would otherwise run UNBOUNDED. This is the explicit
    fail-fast for that case.
    """

    def __init__(self, reason: str, *, usage: Any = None) -> None:
        self.usage = usage
        super().__init__(f"token budget cannot be enforced: {reason} (usage={usage!r})")


def _maybe_int(value: Any) -> int | None:
    """Coerce a usage field to a non-negative-ish int, or ``None`` if it isn't a number.

    Booleans are rejected (``available: True`` must never be read as the integer 1).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _get(usage: Any, key: str) -> Any:
    """Read ``key`` from a usage value that may be a Mapping OR an attribute object.

    The app-server / codex-exec usages are plain dicts, but claude's SDK ``msg.usage``
    can be an OBJECT (a dataclass / pydantic model) — read it defensively rather than
    assume a Mapping, so the accessor is genuinely backend-agnostic.
    """
    if isinstance(usage, Mapping):
        return usage.get(key)
    return getattr(usage, key, None)


def total_tokens_from_usage(usage: Any) -> int:
    """Return the total token count from a GENERIC ``outcome.usage``, or fail fast.

    Backend-AGNOSTIC: it understands the heterogeneous shapes defensively, WITHOUT any
    backend-name branch:

    * **app-server** ``{available: bool, total: {totalTokens, ...}, ...}`` — when
      ``available`` is explicitly ``False`` this is the G5 unavailable marker
      (:class:`BudgetUnavailable`); otherwise the total is read from the ``total``
      breakdown's ``totalTokens`` (the dedup-safe accumulated total the backend already
      computed).
    * **claude / codex-exec** ``{input_tokens, output_tokens, ...}`` (or the camelCase /
      attribute-object variants) — no ``totalTokens``, so the component token counts are
      SUMMED.
    * a top-level explicit total (``totalTokens`` / ``total_tokens``) is honored if the
      usage carries one directly.

    Fails fast with :class:`BudgetUnavailable` when there is nothing real to count:
    ``usage is None``, ``available: False``, or a mapping/object with NO token-count
    field at all (e.g. codex-exec's ``{}`` for a turn that reported no usage). A present
    breakdown that genuinely totals 0 is a REAL reading (returns 0), distinct from
    "no usage".
    """
    if usage is None:
        raise BudgetUnavailable("no usage reported (usage is None)", usage=usage)

    # app-server's explicit availability marker — when present and False, this is the G5
    # "usage unavailable" signal the backend deliberately emitted (NOT a silent absence).
    available = _get(usage, "available")
    if available is False:
        reason = _get(usage, "reason") or "backend reported usage unavailable"
        raise BudgetUnavailable(str(reason), usage=usage)

    # Prefer an explicit total breakdown if one is exposed (app-server's accumulated
    # ``total`` is the authoritative dedup-safe figure); fall back to a top-level total,
    # then to summing the component counts. The search is generic over field NAMES, never
    # over a backend identity.
    total = _get(usage, "total")
    candidates: list[Any] = []
    if total is not None:
        candidates.append(total)
    # Also consider the usage object itself for top-level totals / component counts
    # (claude/codex put the counts at the top level; app-server nests them under total).
    candidates.append(usage)

    for source in candidates:
        explicit = _first_int(source, _EXPLICIT_TOTAL_FIELDS)
        if explicit is not None:
            return explicit

    for source in candidates:
        summed = _sum_components(source)
        if summed is not None:
            return summed

    # Nothing countable anywhere → the backend gave us no token figures to budget against.
    raise BudgetUnavailable(
        "usage carries no token-count fields to enforce against", usage=usage)


def _first_int(source: Any, fields: tuple[str, ...]) -> int | None:
    """First field in ``fields`` present on ``source`` as an int, else ``None``."""
    for f in fields:
        v = _maybe_int(_get(source, f))
        if v is not None:
            return v
    return None


def _sum_components(source: Any) -> int | None:
    """Sum the component token fields on ``source``; ``None`` if none are present.

    Returning ``None`` (not 0) when NO component field exists is what lets the caller
    distinguish "a real breakdown that totals 0" from "no token info at all" (the latter
    must fail fast, the former is a valid 0 reading).
    """
    found = False
    total = 0
    for f in _COMPONENT_TOKEN_FIELDS:
        v = _maybe_int(_get(source, f))
        if v is not None:
            found = True
            total += v
    return total if found else None


def enforce_token_budget(usage: Any, max_total_tokens: int | None) -> None:
    """Enforce a token upper-limit against a GENERIC ``outcome.usage``.

    * ``max_total_tokens is None`` → budgeting is DISABLED: no-op (and crucially NO
      fail-fast on missing usage — a run that never asked to be budgeted is never failed
      for lacking usage).
    * Otherwise read the total via :func:`total_tokens_from_usage` (which fails fast with
      :class:`BudgetUnavailable` if the backend reported no usage — G5), and raise
      :class:`BudgetExceeded` when the total EXCEEDS the limit (being exactly AT the limit
      is allowed).

    Pure and backend-agnostic — the loop calls this on each round's outcome; it has no
    knowledge of which backend produced the usage.
    """
    if max_total_tokens is None:
        return
    total = total_tokens_from_usage(usage)  # may raise BudgetUnavailable (G5 fail-fast)
    if total > max_total_tokens:
        raise BudgetExceeded(
            total_tokens=total, max_total_tokens=max_total_tokens, usage=usage)
