from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from tilldone.core.events import AgentEvent
from tilldone.core.spec import AgentRunSpec, BackendCapabilities, RunOutcome
from tilldone.core.tools import ToolExecutor


@runtime_checkable
class RunHandle(Protocol):
    @property
    def events(self) -> AsyncIterator[AgentEvent]: ...
    async def outcome(self) -> RunOutcome: ...
    async def cancel(self) -> None: ...


@runtime_checkable
class Backend(Protocol):
    name: str
    def capabilities(self) -> BackendCapabilities: ...
    async def start(self, spec: AgentRunSpec, executor: ToolExecutor) -> RunHandle: ...
    async def aclose(self) -> None: ...


@runtime_checkable
class SessionCapable(Protocol):
    async def open_session(self, spec: AgentRunSpec, executor: ToolExecutor) -> "AgentSession": ...


@runtime_checkable
class AgentSession(Protocol):
    async def turn(self, prompt: str) -> RunHandle: ...
    async def close(self) -> None: ...


# --- Rich-control optional protocols (capability-scoped CONTROL PLANE) --------
#
# steer / fork / rollback are a backend-agnostic, capability-scoped control plane.
# They are declared here as runtime_checkable optional Protocols — the same pattern
# as ``SessionCapable`` — so a control-plane caller discovers the capability via
# ``isinstance(backend, SteerCapable)`` WITHOUT core ever importing a backend.
#
# HARD BOUNDARY: ``run_task`` / ``ContractEvaluator`` / ``ToolRegistry`` MUST NOT
# reference these. The contract loop's only seam is ``backend.start`` + ``capabilities``;
# a rollback (or steer/fork) branch leaking into ``run_task`` is an architecture failure
# (``test_run_task_is_ignorant_of_rich_control`` enforces this on every unit pass).


@runtime_checkable
class SteerCapable(Protocol):
    """A backend that can STEER an in-flight turn (inject input mid-turn).

    ``expected_turn_id`` is a REQUIRED active-turn precondition (the source is the
    turn id minted when the turn started); the steer fails fast if it does not match
    the currently-active turn (stale id / already-completed turn / consecutive steer
    over a now-rotated turn id). Returns the new active turn id.

    Pure interface — NOT wired into ``run_task``; reachable only by a control-plane
    caller that has the live thread id.
    """

    async def steer(self, thread_id: str, input: object, *,
                    expected_turn_id: str) -> str: ...


@runtime_checkable
class BranchingCapable(Protocol):
    """A backend that can branch a thread: ``fork`` (copy a thread into a new one) and
    ``rollback`` (drop the last N turns from a thread's HISTORY).

    ``rollback`` rolls back THREAD HISTORY ONLY — it does **NOT** revert the
    filesystem (no artifact/file is restored or deleted). File-level rollback, if ever
    wanted, is a separate artifact-snapshot concern and must not be conflated with this.
    ``num_turns`` must be ``>= 1``.

    Pure interface — NOT wired into ``run_task``; reachable only by a control-plane
    caller that has the live thread id.
    """

    async def fork(self, thread_id: str) -> str: ...
    async def rollback(self, thread_id: str, *, num_turns: int) -> str: ...


@runtime_checkable
class SkillsCapable(Protocol):
    """A backend that can DISCOVER skills and register skill ROOT directories.

    ``skills_list`` discovers the skills available to a thread (returning the raw
    per-cwd skill metadata the backend reports). ``set_skills_extra_roots`` registers
    extra skill-root directories; ``clear_skills_extra_roots`` resets that registration.

    GLOBAL-STATE HYGIENE (RT-APP-SKILLS-ROOT): extra roots are a GLOBAL, NOT a
    thread-scoped, setting on backends that implement this (e.g. ``codex app-server``'s
    ``skills/extraRoots/set``). A caller that registers extra roots MUST reset them
    (``clear_skills_extra_roots``) when done, or the registration leaks into a shared
    home and pollutes a later list / resume / skills view. Prefer a temporary
    CODEX_HOME and always clear extra roots when done.

    Pure interface — the SAME capability-scoped pattern as ``SteerCapable`` /
    ``BranchingCapable``: NOT wired into ``run_task`` (skills discovery is not part of
    the contract loop), reachable only by a caller that holds the live thread id.
    """

    async def skills_list(self, thread_id: str, *, cwds: object = ...,
                          force_reload: bool = ...) -> object: ...
    async def set_skills_extra_roots(self, thread_id: str,
                                     roots: object) -> None: ...
    async def clear_skills_extra_roots(self, thread_id: str) -> None: ...
