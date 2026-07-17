"""LIVE contract-equivalence round-trip for ``CodexAppServerBackend`` (M2 LaneC).

Run (via the live runner, which sets the gate + sources the gateway key)::

    bash scripts/run-live.sh tests/backends/app_server/test_backend_contract_live.py -v

Or manually::

    TILLDONE_APPSERVER_E2E=1 LLM_API_KEY=... \
        uv run pytest tests/backends/app_server/test_backend_contract_live.py -v

This is the proof that the app-server backend is contract-equivalent to core's
``run_task`` loop end to end, against a REAL ``codex app-server --stdio`` talking
to the live gateway — the **structured-output path** satisfying a real
:class:`CompletionContract`:

    run_task(contract, CodexAppServerBackend, ...)
      -> caps.host_tool_calls is False  (the backend declares this)
      -> spec.output_schema = ContractEvaluator.output_schema()  (sent on turn/start)
      -> model emits the final agentMessage as schema-valid JSON
      -> backend strict-parses it into RunOutcome.structured_output
      -> evaluator.apply_structured(state, structured_output)   <-- the contract JUDGE
      -> evaluator.evaluate(state, snaps) finds nothing missing
      -> _result(contract, state) returns the single declared scalar

The assertion (``result == "ZETA-4417-QX"``) therefore proves the WHOLE chain,
not just that a turn ran: the high-entropy string the prompt asked for must have
travelled outputSchema -> model JSON -> apply_structured -> ContractEvaluator ->
core's scalar return. The contract done-ness judge (apply_structured + evaluate)
is squarely in the path; nothing is bypassed.

Why ``run_task`` (not driving the backend by hand)
---------------------------------------------------
``run_task`` builds the :class:`AgentRunSpec` internally and does NOT expose an
``env`` parameter, so ``spec.env`` is empty. The test passes the isolated
``CODEX_HOME`` through the backend's explicit ``codex_home=`` constructor
argument, while only ``LLM_API_KEY`` is inherited from ``os.environ`` so the
isolated config can resolve its ``env_key``. This keeps the real ``run_task``
contract loop fully in the path (the whole point of LaneC) instead of
re-implementing it at the backend level.

SKIP != PASS (T1)
-----------------
``pytest.mark.integration``. The repo-root ``conftest.py`` counts integration
items collected vs actually-run and, under any live gate env, **fails the
session** if every integration test merely skipped (run-count==0).
``scripts/run-live.sh`` turns that gate on, so green-from-100%-skips is
impossible. With no gate env the test skips cleanly via a real ``skipif`` (never
a silent pass), and a body-executed sentinel (``_BODY_RAN``) makes a
falsely-"passed" empty body detectable.

No secret is ever printed or written to disk: ``LLM_API_KEY`` is read from the
ambient environment and flows only into the child process env; the
``config.toml`` under the throwaway ``CODEX_HOME`` carries only the env-var
*name* (``env_key``), never the value.

Teardown / orphan reaping — now via ``backend.aclose()`` (LaneD)
---------------------------------------------------------------
``run_task``'s **happy path** (``evaluate`` clean -> ``return _result(...)``) does
NOT close the backend's run handle. The ``codex app-server`` child is *persistent*
(it stays alive for further turns / resume across a multi-round ``run_task``), so
unlike the one-shot ``codex exec`` child — which self-exits and is reaped when
``run_task`` drains its event stream — the app-server child SURVIVES a successful
``run_task`` and would be an orphan within a long-lived live pytest session.

LaneD made the backend keep the transport **resident** (so multi-round reuse works
without the ephemeral ``thread/resume`` ``-32600`` footgun) and made
``CodexAppServerBackend.aclose()`` truly close every resident transport — which is
exactly what reaps the persistent child. So this test now does the *correct*
thing the Flow/harness layer will do in production: ``await backend.aclose()`` in
a ``finally``. No manual PID SIGKILL is needed any more. We still diff live
``codex app-server --stdio`` pids against a pre-run baseline and assert NONE
survive teardown — but the reap is now driven by ``aclose()``, proving the leak
fix end-to-end against a real child rather than papering over it with ``kill``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from tilldone.backends.app_server.backend import CodexAppServerBackend
from tilldone.core.contract import CompletionContract, ReturnExpectation
from tilldone.core.loop import CorePolicy, run_task

pytestmark = pytest.mark.integration

# Gate: BOTH the opt-in env flag AND a present gateway key are required. A
# missing key is a real skip (not a pass) — and under a gated live run the T1
# theater check in conftest turns an all-skip session into a failure.
_GATE_ENV = os.environ.get("TILLDONE_APPSERVER_E2E") == "1"
_HAS_KEY = bool(os.environ.get("LLM_API_KEY"))
_skip = pytest.mark.skipif(
    not (_GATE_ENV and _HAS_KEY),
    reason="set TILLDONE_APPSERVER_E2E=1 and LLM_API_KEY to run the live app-server contract round-trip",
)

# Body-executed sentinel: flipped True only when the live body runs to
# completion. A test that "passes" without flipping this would be an empty body
# falsely reported green — the trailing assertion guards against that.
_BODY_RAN = {"done": False}

# The high-entropy answer the model is instructed to return. It is NOT a value
# the backend/core could synthesize — the only way ``result`` equals it is if the
# model put it into the structured channel and the contract judge ingested it.
_EXPECTED_ANSWER = "ZETA-4417-QX"

# Non-secret gateway fixture config for the
# isolated CODEX_HOME. Carries the provider name + base_url + the env-var NAME
# (env_key), never the key value itself.
_CONFIG_TOML = """model = "gw/model-a"
model_provider = "gateway"
[model_providers.gateway]
name = "gateway"
base_url = "https://gateway.example.invalid/v1"
env_key = "LLM_API_KEY"
wire_api = "responses"
"""


def _codex_appserver_pids() -> set[int]:
    """PIDs of any live ``codex app-server --stdio`` processes (for baseline diff).

    Matches the full ``--stdio`` invocation so we only ever count the app-server
    children this run spawns, never an unrelated ``codex exec``.
    """
    try:
        out = subprocess.run(
            ["pgrep", "-f", "codex app-server --stdio"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except Exception:
        return set()
    return {int(tok) for tok in out.split() if tok.strip().isdigit()}


def _orphans_since(baseline: set[int], *, timeout_s: float = 3.0) -> set[int]:
    """Return any ``codex app-server --stdio`` pid spawned since ``baseline`` that is
    still alive, giving the OS a bounded window to reap children that ``aclose()``
    has already SIGKILLed (``proc.kill()`` + ``await proc.wait()`` inside the
    transport returns once *this* process reaps its child, but ``pgrep`` may still
    observe a transient zombie for a tick).

    Unlike the old ``_reap`` this NEVER sends a signal — it only OBSERVES. The
    reaping is owned by ``backend.aclose()`` (the LaneD leak fix); a non-empty
    return here is a genuine teardown failure (aclose did not close the child).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        survivors = _codex_appserver_pids() - baseline
        if not survivors:
            return set()
        time.sleep(0.1)
    return _codex_appserver_pids() - baseline


def _make_isolated_codex_home() -> str:
    """Throwaway CODEX_HOME under /tmp containing the non-secret gateway config."""
    home = tempfile.mkdtemp(prefix="cfv2_appserver_contract_codexhome_")
    os.chmod(home, 0o700)
    with open(os.path.join(home, "config.toml"), "w", encoding="utf-8") as fh:
        fh.write(_CONFIG_TOML)
    return home


@_skip
async def test_appserver_contract_roundtrip_live():
    """Real ``codex app-server`` satisfies a minimal CompletionContract via run_task.

    Minimal contract: a single required ``answer:str`` return, no outputs, no
    todos, ``require_finish=False``. The prompt instructs the model to return
    ``answer`` set to the high-entropy string ``ZETA-4417-QX``. Because the
    backend declares ``host_tool_calls=False``, ``run_task`` sends
    ``output_schema=evaluator.output_schema()`` on the turn and ingests the
    model's final JSON via ``apply_structured`` before ``evaluate`` — so a green
    assertion proves the full contract-equivalence chain.
    """
    # Sanity: the gate genuinely admitted us with a key present (defense in depth
    # vs the skipif). We read the key only to assert presence and to seed the
    # child env below — its value is never logged or asserted on.
    api_key = os.environ.get("LLM_API_KEY")
    assert api_key, "live test entered without LLM_API_KEY — gate misconfigured"

    codex_home = _make_isolated_codex_home()
    workdir = tempfile.mkdtemp(prefix="cfv2_appserver_contract_workdir_")
    # Baseline BEFORE spawn so the orphan check diffs only processes we created.
    baseline_pids = _codex_appserver_pids()

    # CODEX_HOME is explicit on the backend; only LLM_API_KEY is inherited by the
    # child so the isolated config can resolve its env_key without logging the key.
    saved_env = {k: os.environ.get(k) for k in ("LLM_API_KEY",)}
    os.environ["LLM_API_KEY"] = api_key

    backend = CodexAppServerBackend(mode="structured", codex_home=codex_home)

    # Minimal completion contract: one required str return, nothing else.
    contract = CompletionContract(
        outputs=(),
        returns=(ReturnExpectation("answer", "str", required=True),),
        todos=(),
        require_finish=False,
    )
    # The prompt MUST describe the structured envelope, not just `answer`. The
    # The live gateway/model combo does NOT strictly enforce the nested outputSchema
    # envelope (it will otherwise emit a flat {"answer": ...}, which the strict
    # apply_structured judge correctly drops because it has no top-level
    # `returns` key — leaving the contract unsatisfied and forcing a useless
    # round 2 that fails on ephemeral thread/resume). Describing the exact three
    # top-level keys makes the model emit the schema-shaped object the contract
    # judge ingests. The high-entropy ANSWER value is still the load-bearing
    # proof — it can only reach `result` via the structured channel + judge.
    prompt = (
        "Return ONLY a JSON object that matches the provided output schema exactly. "
        "It MUST have these three top-level keys: `returns`, `todos_done`, `finished`. "
        f'Set `returns` to the object {{"answer": "{_EXPECTED_ANSWER}"}}, set `todos_done` '
        "to the empty array [], and set `finished` to true. "
        f"`returns.answer` must equal `{_EXPECTED_ANSWER}` verbatim — do not add words, "
        "punctuation, quotes, or explanation around it."
    )

    try:
        # THE contract loop: structured path -> outputSchema -> model JSON ->
        # apply_structured -> ContractEvaluator.evaluate -> scalar return.
        result = await run_task(
            contract,
            backend,
            prompt=prompt,
            cwd=Path(workdir),
            policy=CorePolicy(max_rounds=2, timeout_s=120.0),
        )

        # Single declared return -> run_task returns the scalar (not a dict).
        # Equality with the high-entropy string is the M2 contract-equivalence
        # proof: it could ONLY arrive via the structured channel + contract judge.
        assert result == _EXPECTED_ANSWER, (
            f"contract round-trip returned {result!r}, expected {_EXPECTED_ANSWER!r} "
            "(structured channel -> apply_structured -> evaluate -> scalar return)"
        )

        # Body ran to completion — flip the sentinel (proves a real execution,
        # not a falsely-green empty body).
        _BODY_RAN["done"] = True
    finally:
        # THE leak fix, exercised against a real child: backend.aclose() closes the
        # resident transport run_task's happy path leaves open (LaneD). This is what
        # the production Flow/harness layer calls in its own finally — no manual
        # SIGKILL. We then OBSERVE (never signal) that no orphan survived, so a
        # non-empty `leaked` is a genuine aclose() teardown failure.
        await backend.aclose()
        leaked = _orphans_since(baseline_pids)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(codex_home, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)

    # --- post-teardown assertions (outside the finally so failures surface) ---
    assert not os.path.exists(codex_home), "throwaway CODEX_HOME not removed"
    assert not os.path.exists(workdir), "throwaway workdir not removed"

    # No orphaned app-server from THIS test: after backend.aclose() closed the
    # resident transport, no codex app-server --stdio pid spawned since the pre-run
    # baseline may remain. A survivor here means aclose() failed to reap the child.
    assert not leaked, (
        "orphan codex app-server --stdio process(es) survived backend.aclose() "
        f"teardown: {sorted(leaked)}")

    # SKIP != PASS sentinel: a green result here is only legitimate if the live
    # body actually executed end to end.
    assert _BODY_RAN["done"], "live body did not run to completion (SKIP != PASS guard)"
