"""LIVE tools-mode contract proof for ``CodexAppServerBackend`` (M3 LaneB).

Run (via the live runner, which sets the gate + sources the gateway key)::

    bash scripts/run-live.sh tests/backends/app_server/test_backend_tools_live.py -v

Or manually::

    TILLDONE_APPSERVER_E2E=1 LLM_API_KEY=... \
        uv run pytest tests/backends/app_server/test_backend_tools_live.py -v

This is the **tools-mode** companion to ``test_backend_contract_live.py`` (which
proved the structured/outputSchema path). It proves that
``CodexAppServerBackend(mode="tools")`` drives a REAL ``codex app-server --stdio``
through core's ``run_task`` over the **tool channel** (``host_tool_calls=True``),
and that the tool channel is genuinely wired all the way to the host executor —
not the model merely *claiming* it called tools.

The proof is two-pronged (RT-TOOL-1 nonce-side-effect + contract-equivalence):

    run_task(contract, CodexAppServerBackend(mode="tools"), custom_tools={...})
      -> caps.host_tool_calls is True
      -> registry.add_contract_tools()  (return/todo/finish handlers)
      -> registry.add_custom("write_marker", handler)   (the host tool)
      -> backend wires ``item/tool/call`` -> executor.invoke (ToolRegistry)
      -> thread/start carries dynamicTools = [return, todo, finish, write_marker]
      -> the model calls write_marker -> item/tool/call -> OUR handler runs HOST-SIDE
           -> writes the high-entropy nonce file on THIS box (the side-effect)
      -> the model calls return done="ok" -> item/tool/call -> ToolRegistry._return
           -> ContractState.returns["done"] = "ok"   (the contract JUDGE channel)
      -> the model calls finish -> ContractState.finished = True (require_finish)
      -> evaluate(state, snaps) finds nothing missing
      -> _result(contract, state) returns the single declared scalar "ok"

**Why the nonce file is the load-bearing proof.** A model could *narrate* "I
called write_marker" in its agentMessage without the tool channel ever firing.
But the high-entropy marker (``NOVA-8841-QX``) can only appear in a file under the
tmp workdir if OUR Python handler actually executed host-side — which only happens
when ``codex app-server`` issues a real ``item/tool/call`` server→client request
that the backend dispatched to the core ToolRegistry. The model cannot write that
file itself (sandbox aside, the handler — not the model — chooses what bytes land
on disk). So ``marker_file contains NOVA-8841-QX`` proves the host executor REALLY
fired (RT-TOOL-1 semantics), and ``result == "ok"`` proves the contract ``return``
+ ``finish`` tools drove ContractState through the SAME tool channel and core's
evaluator was satisfied. Both run through ``run_task`` + the tool channel; nothing
is bypassed and the agentMessage narration is NOT consulted (tools mode →
``structured_output is None``; the tool channel is the sole ContractState
authority).

Why ``run_task`` (not driving the backend by hand)
---------------------------------------------------
``run_task`` builds the :class:`AgentRunSpec` internally and does NOT expose an
``env`` parameter, so ``spec.env`` is empty. The test passes the isolated
``CODEX_HOME`` through the backend's explicit ``codex_home=`` constructor
argument, while only ``LLM_API_KEY`` is inherited from ``os.environ`` so the
isolated config can resolve its ``env_key``. This keeps the real ``run_task``
contract loop fully in the path (the whole point of LaneB): core builds the
ToolRegistry, calls ``add_contract_tools()`` + ``add_custom()``, and passes it
to ``backend.start`` as the executor.

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
*name* (``env_key``), never the value. The nonce is a fixed non-secret test
string.

Teardown / orphan reaping — via ``backend.aclose()`` (LaneD)
------------------------------------------------------------
``run_task``'s happy path (``evaluate`` clean -> ``return _result(...)``) does NOT
close the backend's run handle. The ``codex app-server`` child is *persistent*, so
it SURVIVES a successful ``run_task`` and would be an orphan within a long-lived
live pytest session. ``CodexAppServerBackend.aclose()`` closes every resident
transport, which reaps the persistent child. So this test does the correct thing
the Flow/harness layer will do in production: ``await backend.aclose()`` in a
``finally``. We then diff live ``codex app-server --stdio`` pids against a pre-run
baseline and assert NONE spawned by this test survive teardown — observing (never
signalling), so a survivor is a genuine ``aclose()`` failure.
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
from tilldone.core.tools import ToolCall, ToolResult

pytestmark = pytest.mark.integration

# Gate: BOTH the opt-in env flag AND a present gateway key are required. A
# missing key is a real skip (not a pass) — and under a gated live run the T1
# theater check in conftest turns an all-skip session into a failure.
_GATE_ENV = os.environ.get("TILLDONE_APPSERVER_E2E") == "1"
_HAS_KEY = bool(os.environ.get("LLM_API_KEY"))
_skip = pytest.mark.skipif(
    not (_GATE_ENV and _HAS_KEY),
    reason="set TILLDONE_APPSERVER_E2E=1 and LLM_API_KEY to run the live app-server tools-mode contract proof",
)

# Body-executed sentinel: flipped True only when the live body runs to
# completion. A test that "passes" without flipping this would be an empty body
# falsely reported green — the trailing assertion guards against that.
_BODY_RAN = {"done": False}

# The high-entropy marker the host tool writes to disk. It is the nonce-side-effect
# proof (RT-TOOL-1): the only way this string lands in a file under the tmp workdir
# is if OUR handler ran host-side via a real item/tool/call dispatch. It is a fixed
# (non-secret) test string — the entropy is to make an accidental/pre-existing match
# implausible, not to hide anything.
_MARKER_NONCE = "NOVA-8841-QX"

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
    children this run spawns, never an unrelated ``codex exec`` or a long-lived
    ``codex app-server --listen``/``--enable goals`` daemon. (The diff against a
    pre-run baseline also neutralizes any incidental ``pgrep`` self-match, since it
    is present in both snapshots.)
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
    has already SIGKILLed.

    This NEVER sends a signal — it only OBSERVES. The reaping is owned by
    ``backend.aclose()`` (the LaneD leak fix); a non-empty return here is a genuine
    teardown failure (aclose did not close the child).
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
    home = tempfile.mkdtemp(prefix="cfv2_appserver_tools_codexhome_")
    os.chmod(home, 0o700)
    with open(os.path.join(home, "config.toml"), "w", encoding="utf-8") as fh:
        fh.write(_CONFIG_TOML)
    return home


@_skip
async def test_appserver_tools_nonce_and_contract_live():
    """Real ``codex app-server`` (tools mode) fires the host tool AND satisfies the
    contract via the tool channel, end to end through ``run_task``.

    Contract: a single required ``done:str`` return, no outputs, no todos,
    ``require_finish=True`` (so the finish tool is exercised too). A custom host
    tool ``write_marker`` writes the high-entropy nonce to a file under the tmp
    workdir. The prompt instructs the model to: call ``write_marker``, then submit
    ``return`` with ``done="ok"``, then call ``finish``.

    Because the backend declares ``host_tool_calls=True``, ``run_task`` builds a
    ToolRegistry (contract + custom handlers) and passes it as the executor; the
    backend wires ``item/tool/call`` -> ``executor.invoke``. So a green run proves
    the full tool-channel chain — the nonce file (host executor REALLY fired) and
    ``result == "ok"`` (contract return + finish drove ContractState).
    """
    # Sanity: the gate genuinely admitted us with a key present (defense in depth
    # vs the skipif). We read the key only to assert presence and to seed the child
    # env below — its value is never logged or asserted on.
    api_key = os.environ.get("LLM_API_KEY")
    assert api_key, "live test entered without LLM_API_KEY — gate misconfigured"

    codex_home = _make_isolated_codex_home()
    workdir = tempfile.mkdtemp(prefix="cfv2_appserver_tools_workdir_")
    marker_file = Path(workdir) / "marker.txt"
    # Baseline BEFORE spawn so the orphan check diffs only processes we created.
    baseline_pids = _codex_appserver_pids()

    # Did OUR host handler actually run? Closure-captured flag flipped inside the
    # custom tool — the in-process witness that complements the on-disk side-effect.
    handler_ran = {"count": 0}

    async def write_marker(call: ToolCall) -> ToolResult:
        # This runs HOST-SIDE (in THIS pytest process) only when the backend
        # dispatched a real item/tool/call to the core ToolRegistry. The side-effect
        # under test: write the high-entropy nonce to disk.
        handler_ran["count"] += 1
        marker_file.write_text(_MARKER_NONCE, encoding="utf-8")
        return ToolResult(content="written", is_error=False)

    # CODEX_HOME is explicit on the backend; only LLM_API_KEY is inherited by the
    # child so the isolated config can resolve its env_key without logging the key.
    saved_env = {k: os.environ.get(k) for k in ("LLM_API_KEY",)}
    os.environ["LLM_API_KEY"] = api_key

    backend = CodexAppServerBackend(
        mode="tools", codex_home=codex_home)  # the MAIN path (default)

    # Minimal completion contract: one required str return + require_finish (so the
    # finish tool is also exercised). No outputs, no todos.
    contract = CompletionContract(
        outputs=(),
        returns=(ReturnExpectation("done", "str", required=True),),
        todos=(),
        require_finish=True,
    )

    # Firm, ordered, tool-centric instructions. In tools mode the model MUST call
    # the dynamicTools (item/tool/call) — narration alone never satisfies the
    # contract (tool channel is the sole authority). We spell out the exact tool
    # names + arguments so gpt-5.5 reliably drives all three (custom + return +
    # finish). The marker text is named verbatim so the host tool's nonce on disk is
    # unambiguous.
    prompt = (
        "You have host tools available. Do ALL of the following IN ORDER, using the "
        "tools (do not just describe them):\n"
        f"1. Call the `write_marker` tool with no arguments. It writes the marker "
        f"text {_MARKER_NONCE} to a file. Call it exactly once.\n"
        '2. Call the `return` tool with the argument `done` set to the string "ok" '
        "(exactly the two letters o, k).\n"
        "3. Call the `finish` tool to declare the task complete.\n"
        "Do not stop until you have called all three tools in that order."
    )

    try:
        # THE contract loop: tools path -> dynamicTools -> item/tool/call ->
        # ToolRegistry (custom + contract handlers) -> ContractState -> evaluate ->
        # scalar return. The host tool's disk write happens inside this call.
        result = await run_task(
            contract,
            backend,
            prompt=prompt,
            cwd=Path(workdir),
            custom_tools={"write_marker": write_marker},
            policy=CorePolicy(max_rounds=3, timeout_s=180.0),
        )

        # (1) Nonce-side-effect proof (RT-TOOL-1): the marker file exists and holds
        # the high-entropy string. This can ONLY be true if OUR handler ran host-side
        # via a real item/tool/call dispatch — the model could not write it itself.
        assert marker_file.exists(), (
            "marker file was never written — the write_marker host tool did not fire "
            "via item/tool/call (the model may have only narrated calling it)"
        )
        contents = marker_file.read_text(encoding="utf-8")
        assert _MARKER_NONCE in contents, (
            f"marker file does not contain {_MARKER_NONCE!r} (got {contents!r}) — "
            "the host executor side-effect did not occur as expected"
        )

        # (2) In-process witness: the handler closure actually executed.
        assert handler_ran["count"] >= 1, (
            "write_marker handler closure never ran — no item/tool/call reached the "
            "core ToolRegistry executor"
        )

        # (3) Contract-equivalence proof: the single declared return arrived via the
        # `return` tool (tool channel -> ContractState.returns), and require_finish
        # means the `finish` tool also fired — both judged by core's evaluator, which
        # was satisfied (run_task returned instead of raising MaxRoundsExceeded). A
        # single declared return -> run_task returns the scalar, not a dict.
        assert result == "ok", (
            f"contract returned {result!r}, expected 'ok' (return tool -> ContractState "
            "-> evaluate -> scalar return; require_finish also needed the finish tool)"
        )

        # Body ran to completion — flip the sentinel (proves a real execution, not a
        # falsely-green empty body).
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

    # SKIP != PASS sentinel: a green result here is only legitimate if the live body
    # actually executed end to end.
    assert _BODY_RAN["done"], "live body did not run to completion (SKIP != PASS guard)"
