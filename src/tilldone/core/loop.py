from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tilldone.core.backend import Backend
from tilldone.core.budget import enforce_token_budget
from tilldone.core.contract import CompletionContract, ContractEvaluator, ContractState
from tilldone.core.refs import expand_refs
from tilldone.core.registry import ToolRegistry
from tilldone.core.snapshot import snapshot_paths
from tilldone.core.spec import AgentRunSpec, ContextDirView, ResumeHandle


@dataclass
class CorePolicy:
    max_rounds: int = 6
    timeout_s: float | None = None
    api_retry_max: int = 2
    # G5 token budget (backend-agnostic). When set, run_task enforces a TOKEN-count
    # upper limit by reading the GENERIC outcome.usage (pure arithmetic, NO backend
    # branch — see tilldone.core.budget). None (default) disables budgeting. Dollar/cost
    # budgets are DEFERRED.
    max_total_tokens: int | None = None


class MaxRoundsExceeded(RuntimeError):
    ...


class BackendFailure(RuntimeError):
    ...


def _result(contract: CompletionContract, state: ContractState):
    """Shape the return value: 0 declared -> None, 1 -> scalar, >1 -> dict."""
    if len(contract.returns) == 0:
        return None
    if len(contract.returns) == 1:
        return state.returns.get(contract.returns[0].name)
    return dict(state.returns)


async def run_task(contract: CompletionContract, backend: Backend, *,
                   prompt: str, cwd: Path, policy: CorePolicy = CorePolicy(),
                   custom_tools: dict | None = None,
                   context_dirs: "Sequence[ContextDirView]" = (), event_sink=None):
    state = ContractState()
    evaluator = ContractEvaluator(contract)
    caps = backend.capabilities()
    registry = ToolRegistry(state, contract)
    if caps.host_tool_calls:
        registry.add_contract_tools()
    for name, handler in (custom_tools or {}).items():
        registry.add_custom(name, handler)

    # Expand @ref:<id>/<relpath> once up front (fail-fast on unknown id / traversal).
    prompt = expand_refs(prompt, context_dirs)
    snaps = snapshot_paths([o.abs_path for o in contract.outputs])
    contract_defs = evaluator.tool_defs(host_tool_calls=caps.host_tool_calls)
    resume: ResumeHandle | None = None
    missing: list[str] = ["<unstarted>"]

    for attempt in range(1, policy.max_rounds + 1):
        # Inner loop: transient-error retries do NOT consume a contract round (RT-CFG-3).
        retries = 0
        while True:
            spec = AgentRunSpec(
                prompt=prompt, tools=registry.tool_defs(contract_defs), cwd=cwd,
                context_dirs=context_dirs,
                output_schema=None if caps.host_tool_calls else evaluator.output_schema(),
                resume=resume, timeout_s=policy.timeout_s,
                metadata={"attempt": attempt, "retry": retries},
            )
            handle = await backend.start(spec, registry)
            async for ev in handle.events:
                if event_sink is not None:
                    event_sink(ev)
            outcome = await handle.outcome()
            resume = outcome.resume
            if outcome.status == "failed":
                err = outcome.error
                if err is not None and err.code == "transient" and retries < policy.api_retry_max:
                    retries += 1
                    continue
                raise BackendFailure(err.message if err else "unknown backend error")
            break

        # G5 token budget — backend-AGNOSTIC enforcement on the GENERIC outcome.usage.
        # This is pure arithmetic over the usage mapping (tilldone.core.budget) with NO
        # backend-name branch, so it does not touch the backend.start seam and the
        # 0-core-diff guard (test_run_task_is_backend_agnostic) stays green. A no-op when
        # policy.max_total_tokens is None. Over-limit raises BudgetExceeded (clean
        # terminal — stops here, before any further round); a budgeted run that reported
        # no usage raises BudgetUnavailable (fail-fast, never a silent zero cost).
        enforce_token_budget(outcome.usage, policy.max_total_tokens)

        if not caps.host_tool_calls:
            evaluator.apply_structured(state, outcome.structured_output)
        missing = evaluator.evaluate(state, snaps)
        if not missing:
            return _result(contract, state)
        prompt = _continue_prompt(prompt, missing)

    raise MaxRoundsExceeded(f"unsatisfied after {policy.max_rounds} rounds: {missing}")


def _continue_prompt(original: str, missing: list[str]) -> str:
    return (f"{original}\n\n[continue] Still missing: {', '.join(missing)}. "
            "Complete only the missing items.")
