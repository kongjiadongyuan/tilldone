from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from tilldone.core.backend import Backend
from tilldone.core.enums import EventKind
from tilldone.core.events import AgentEvent
from tilldone.core.spec import AgentRunSpec, BackendCapabilities, RunOutcome
from tilldone.core.tools import ToolExecutor

_TERMINAL = (EventKind.RUN_COMPLETED, EventKind.RUN_FAILED)


async def collect_run(backend: Backend, spec: AgentRunSpec,
                      executor: ToolExecutor) -> tuple[list[AgentEvent], RunOutcome]:
    """Drive one backend run to completion, collecting all events + the outcome."""
    handle = await backend.start(spec, executor)
    events = [ev async for ev in handle.events]
    outcome = await handle.outcome()
    return events, outcome


def assert_protocol_invariants(events: Sequence[AgentEvent], outcome: RunOutcome) -> None:
    """Assert a normalized event stream + outcome obey the cross-backend protocol
    contract. Raises AssertionError on any violation. EVERY backend (Fake, Claude,
    CodexExec, third-party) must satisfy this — it is the generalization gate.
    """
    assert events, "no events emitted"

    # 1. seq strictly increasing
    seqs = [e.seq for e in events]
    assert all(b > a for a, b in zip(seqs, seqs[1:])), f"seq not strictly increasing: {seqs}"

    # 2. exactly one terminal event, and it is the last event
    terminals = [i for i, e in enumerate(events) if e.kind in _TERMINAL]
    assert len(terminals) == 1, f"expected exactly 1 terminal event, got {len(terminals)}"
    assert terminals[0] == len(events) - 1, "terminal event must be the last event"

    # 3. outcome status agrees with the terminal event kind
    term = events[terminals[0]]
    if outcome.status == "completed":
        assert term.kind == EventKind.RUN_COMPLETED, "completed outcome without RUN_COMPLETED terminal"
    elif outcome.status == "failed":
        assert term.kind == EventKind.RUN_FAILED, "failed outcome without RUN_FAILED terminal"
    else:
        raise AssertionError(f"unknown outcome status: {outcome.status!r}")

    # 4. tool results carry a call_id; every tool_call has a matching later result
    result_ids = {e.call_id for e in events if e.kind == EventKind.TOOL_RESULT}
    for e in events:
        if e.kind == EventKind.TOOL_RESULT:
            assert e.call_id is not None, "TOOL_RESULT without call_id"
        if e.kind == EventKind.TOOL_CALL:
            assert e.call_id in result_ids, f"TOOL_CALL {e.call_id!r} without matching TOOL_RESULT"

    # 5. resume token round-trips (opaque but present)
    assert outcome.resume is not None, "outcome missing resume handle"
    assert outcome.resume.backend, "resume handle missing backend id"
    assert outcome.resume.token, "resume handle missing token"


class ScriptedRunHandle:
    """A RunHandle that replays a fixed list of pre-built events + a fixed outcome.
    Used to craft adversarial/edge streams (including protocol-violating ones)."""

    def __init__(self, events: Sequence[AgentEvent], outcome: RunOutcome):
        self._events = list(events)
        self._outcome = outcome

    @property
    async def events(self) -> AsyncIterator[AgentEvent]:
        for e in self._events:
            yield e

    async def outcome(self) -> RunOutcome:
        return self._outcome

    async def cancel(self) -> None:
        ...


class ScriptedEventsBackend:
    """A Backend that emits a caller-supplied event list + outcome verbatim.
    Handy for adversarial conformance testing and viewer fixtures."""

    name = "scripted"

    def __init__(self, events: Sequence[AgentEvent], outcome: RunOutcome,
                 *, host_tool_calls: bool = True):
        self._events = events
        self._outcome = outcome
        self._host = host_tool_calls

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(host_tool_calls=self._host)

    async def start(self, spec: AgentRunSpec, executor: ToolExecutor) -> ScriptedRunHandle:
        return ScriptedRunHandle(self._events, self._outcome)

    async def aclose(self) -> None:
        ...
