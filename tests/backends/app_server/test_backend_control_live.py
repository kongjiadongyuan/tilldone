"""LIVE rich-control proof for ``CodexAppServerBackend`` (M5 LaneB) — fork / rollback / steer.

Run (via the live runner, which sets the gate + sources the gateway key)::

    bash scripts/run-live.sh tests/backends/app_server/test_backend_control_live.py -v

Or manually::

    TILLDONE_APPSERVER_E2E=1 LLM_API_KEY=... \
        uv run pytest tests/backends/app_server/test_backend_control_live.py -v

These are the LIVE counterparts to the deterministic unit tests in
``test_backend_control.py`` (which drive the in-memory ``FakePeer``). Here the
control-plane methods (:meth:`CodexAppServerBackend.rollback` /
:meth:`~CodexAppServerBackend.fork` / :meth:`~CodexAppServerBackend.steer`) are
exercised against a REAL ``codex app-server --stdio`` child, on a RESIDENT thread
the backend instance owns (the control plane acts on the SAME ``_live`` residency
the multi-round contract loop reuses).

THE headline — rollback rolls back HISTORY, NOT the filesystem (Test 1)
-----------------------------------------------------------------------
This is the strongest possible live proof of the capability-scoped control-plane boundary
and the live counterpart to the unit ``test_rollback_touches_no_filesystem`` /
the 0-core-diff ``test_rollback_is_history_only_keeps_files``:

  * A ``CodexAppServerBackend(mode="tools")`` with a custom host tool
    ``write_artifact`` that writes a high-entropy string (``ROLLBACK-KEEP-8841``)
    to ``artifact.txt`` in cwd. ONE turn is driven where the model calls that tool
    → the file lands on disk (proven by reading it back BEFORE the rollback).
  * Then ``await backend.rollback(thread_id, num_turns=1)`` is issued on the SAME
    backend instance (the thread is resident). The ``thread/rollback`` succeeds.
  * **Assert**: the ``thread/rollback`` returned the thread id (history rolled back)
    AND ``artifact.txt`` STILL EXISTS with its content intact. The rollback rolled
    back the conversation HISTORY only; reverting local file changes is the
    caller's responsibility (per the ``ThreadRollbackParams.numTurns`` schema), so
    the side-effect file SURVIVES untouched.

How the thread id is obtained for the control call
---------------------------------------------------
We drive each turn via the backend's OWN ``start()``/``events``/``outcome`` (NOT
``run_task``) so we keep the backend instance alive (thread stays resident) AND
read the raw ``outcome.resume`` — the ResumeHandle whose ``token["thread_id"]`` is
the resident-thread key the control methods target. Driving the backend by hand
(rather than ``run_task``) is also what lets us catch a turn IN-FLIGHT for the
steer race (Test 3).

SKIP != PASS (T1) / no-secret / no-orphan teardown
--------------------------------------------------
``pytest.mark.integration`` + a real ``skipif`` gate. The repo-root conftest fails
a gated live session in which every integration test merely skipped, and the
``_BODY_RAN`` sentinel makes a falsely-green empty body detectable. No secret is
printed or written: ``LLM_API_KEY`` flows only into the child env; ``config.toml``
under the throwaway ``CODEX_HOME`` carries only the env-var NAME. Each test does
``await backend.aclose()`` in a ``finally`` (the leak fix — reaps the persistent
child) and then OBSERVES (never signals) that no ``codex app-server --stdio`` pid
spawned by the test survived teardown. Each test is cheap (ONE live turn).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from tilldone.backends.app_server.backend import CodexAppServerBackend
from tilldone.core.spec import AgentRunSpec
from tilldone.core.tools import ToolCall, ToolDef, ToolResult

pytestmark = pytest.mark.integration

# Gate: BOTH the opt-in env flag AND a present gateway key are required. A missing
# key is a real skip (not a pass); under a gated live run the T1 theater check in
# conftest turns an all-skip session into a failure.
_GATE_ENV = os.environ.get("TILLDONE_APPSERVER_E2E") == "1"
_HAS_KEY = bool(os.environ.get("LLM_API_KEY"))
_skip = pytest.mark.skipif(
    not (_GATE_ENV and _HAS_KEY),
    reason="set TILLDONE_APPSERVER_E2E=1 and LLM_API_KEY to run the live app-server rich-control proof",
)

# Body-executed sentinels (one per test): flipped True only when that live body runs
# to completion. A test that "passes" without flipping its sentinel would be a falsely
# green empty body — the trailing assertion guards against that.
_BODY_RAN = {"rollback": False, "fork": False, "steer": False}

# The high-entropy string the rollback test's host tool writes to disk. The only way
# it lands in a file under the tmp workdir is if OUR handler ran host-side via a real
# item/tool/call dispatch. Fixed (non-secret) test string; the entropy just makes an
# accidental/pre-existing match implausible. It is THE survivor the rollback must keep.
_ARTIFACT_NONCE = "ROLLBACK-KEEP-8841"
_ARTIFACT_NAME = "artifact.txt"

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
    ``codex app-server --listen`` daemon. (The diff against a pre-run baseline also
    neutralizes any incidental ``pgrep`` self-match, present in both snapshots.)
    """
    try:
        out = subprocess.run(
            ["pgrep", "-f", "codex app-server --stdio"],
            capture_output=True, text=True, check=False,
        ).stdout
    except Exception:
        return set()
    return {int(tok) for tok in out.split() if tok.strip().isdigit()}


def _orphans_since(baseline: set[int], *, timeout_s: float = 3.0) -> set[int]:
    """``codex app-server --stdio`` pids spawned since ``baseline`` still alive after a
    bounded reap window. NEVER signals — a non-empty return is a real aclose() failure
    (the reaping is owned by ``backend.aclose()``)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        survivors = _codex_appserver_pids() - baseline
        if not survivors:
            return set()
        time.sleep(0.1)
    return _codex_appserver_pids() - baseline


def _make_isolated_codex_home() -> str:
    """Throwaway CODEX_HOME under /tmp containing the non-secret gateway config."""
    home = tempfile.mkdtemp(prefix="cfv2_appserver_control_codexhome_")
    os.chmod(home, 0o700)
    with open(os.path.join(home, "config.toml"), "w", encoding="utf-8") as fh:
        fh.write(_CONFIG_TOML)
    return home


def _no_tools_spec(prompt: str, cwd: Path, *, resume=None) -> AgentRunSpec:
    """A spec for a pure free-text turn (no host tools, no output_schema). Driven in
    TOOLS mode (the MAIN path) so the model's free-text reply lands in
    ``outcome.final_text`` WITHOUT being strict-JSON parsed — the fork/steer smokes
    only need a thread that reaches turn/start (and, for fork, a completed turn)."""
    return AgentRunSpec(prompt=prompt, tools=(), cwd=cwd, output_schema=None,
                        resume=resume, timeout_s=180.0)


class _NoToolExecutor:
    """No tools declared → no ``item/tool/call`` can arrive → invoke must never run."""

    async def invoke(self, call):  # pragma: no cover - must never run
        raise AssertionError("this control smoke declares no tools; invoke must not run")


async def _consume(handle) -> None:
    """Drain a run handle's event stream to its single terminal event."""
    async for _ev in handle.events:
        pass


# --------------------------------------------------------------------------- #
# Test 1 — THE headline: rollback rolls back HISTORY, not the FILESYSTEM        #
# --------------------------------------------------------------------------- #

@_skip
async def test_appserver_rollback_keeps_files_live():
    """Real ``codex app-server`` (tools mode): a turn writes a high-entropy artifact via
    a host tool; ``backend.rollback(thread_id, num_turns=1)`` rolls back the conversation
    HISTORY and the artifact file SURVIVES on disk untouched.

    This is the live counterpart to the unit boundary tests — the strongest possible
    proof that ``thread/rollback`` is history-only and the backend performs ZERO
    filesystem operations (reverting local file changes is the caller's responsibility,
    per the schema). The thread id targeted by ``rollback`` comes from the turn's
    ``outcome.resume.token["thread_id"]`` (the resident-thread key).

    LIVE FACT (drove the test design): ``thread/rollback`` loads the thread from its
    on-disk ROLLOUT, so it requires a NON-ephemeral thread — against the default
    ephemeral thread the live app-server rejects it with
    ``-32600 "no rollout found for thread id ..."``. Hence ``persist_thread=True``
    (a rollout materializes); ``aclose()`` archives the persisted thread on teardown.
    """
    api_key = os.environ.get("LLM_API_KEY")
    assert api_key, "live test entered without LLM_API_KEY — gate misconfigured"

    codex_home = _make_isolated_codex_home()
    workdir = tempfile.mkdtemp(prefix="cfv2_appserver_control_rb_workdir_")
    artifact = Path(workdir) / _ARTIFACT_NAME
    baseline_pids = _codex_appserver_pids()

    # CODEX_HOME is explicit on the backend; only LLM_API_KEY is inherited by the
    # child so the isolated config can resolve its env_key. Never logged.
    saved_env = {k: os.environ.get(k) for k in ("LLM_API_KEY",)}
    os.environ["LLM_API_KEY"] = api_key

    # In-process witness that complements the on-disk side-effect: the closure flips
    # this only when a real item/tool/call reached the host executor.
    handler_ran = {"count": 0}

    async def write_artifact(call: ToolCall) -> ToolResult:
        # Runs HOST-SIDE (in THIS pytest process) only when the backend dispatched a
        # real item/tool/call. The side-effect under test: write the nonce to disk.
        handler_ran["count"] += 1
        artifact.write_text(_ARTIFACT_NONCE, encoding="utf-8")
        return ToolResult(content="written", is_error=False)

    # persist_thread=True so the thread is NON-ephemeral and a rollout materializes on
    # disk. thread/rollback loads the thread from that rollout — against an EPHEMERAL
    # thread (the default) the live app-server rejects it with
    # ``-32600 "no rollout found for thread id ..."`` (the same footgun thread/resume
    # has). aclose() best-effort thread/archive's the persisted thread on teardown
    # (RT-APP-HOME-LEAK), so this stays orphan-safe.
    backend = CodexAppServerBackend(
        mode="tools", persist_thread=True, codex_home=codex_home)  # MAIN path

    # A single dynamicTool the model is told to call. (No contract here — we drive the
    # turn directly via backend.start, so this is just the host tool + a finish nudge.)
    tools = (
        ToolDef(
            name="write_artifact",
            description=f"Write the marker text {_ARTIFACT_NONCE} to {_ARTIFACT_NAME}.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            kind="custom",
        ),
    )
    prompt = (
        "You have a host tool available. Call the `write_artifact` tool exactly once "
        f"(it writes the marker text {_ARTIFACT_NONCE} to a file), then reply 'done'. "
        "Use the tool — do not just describe it."
    )

    try:
        # --- drive ONE turn: the host tool writes the artifact on disk -------------
        spec = AgentRunSpec(prompt=prompt, tools=tools, cwd=Path(workdir),
                            output_schema=None, timeout_s=180.0)
        handle = await backend.start(spec, _Executor({"write_artifact": write_artifact}))
        await _consume(handle)
        outcome = await handle.outcome()
        assert outcome.status == "completed", f"turn did not complete: {outcome.error!r}"

        # The host tool fired (in-process witness) AND the artifact is on disk BEFORE
        # the rollback — establishing the precondition the rollback must not disturb.
        assert handler_ran["count"] >= 1, (
            "write_artifact handler never ran — no item/tool/call reached the executor")
        assert artifact.exists(), "artifact.txt was not written by the host tool"
        before_text = artifact.read_text(encoding="utf-8")
        assert _ARTIFACT_NONCE in before_text, (
            f"artifact.txt does not contain {_ARTIFACT_NONCE!r} before rollback (got {before_text!r})")
        before_listing = sorted(os.listdir(workdir))

        # The thread id the control call targets — from the turn's resume handle.
        resume = outcome.resume
        assert resume is not None, "turn produced no resume handle"
        assert resume.backend == "codex_app_server"
        thread_id = resume.token.get("thread_id")
        assert thread_id, "resume token missing thread_id (cannot target rollback)"
        thread_id = str(thread_id)
        # The thread is RESIDENT in this backend instance (control plane operates on it).
        assert thread_id in backend._live, (
            "thread must be resident for the control plane to act on it")

        # --- THE control call: rollback the last turn's HISTORY --------------------
        rolled = await backend.rollback(thread_id, num_turns=1)
        # thread/rollback succeeded and returned the thread id (history rolled back).
        assert rolled == thread_id, (
            f"thread/rollback returned {rolled!r}, expected the rolled thread id {thread_id!r}")

        # --- THE headline assertion: the artifact SURVIVED the rollback ------------
        # Rollback rolled back the CONVERSATION HISTORY only; it performed ZERO
        # filesystem ops, so the host tool's side-effect file is untouched.
        assert artifact.exists(), (
            "artifact.txt was REMOVED by rollback — rollback must roll back HISTORY ONLY, "
            "NOT the filesystem (the side-effect file must survive)")
        after_text = artifact.read_text(encoding="utf-8")
        assert after_text == before_text, (
            f"artifact.txt content changed across rollback ({before_text!r} -> {after_text!r}) "
            "— rollback must not touch the filesystem")
        assert _ARTIFACT_NONCE in after_text, (
            f"artifact.txt no longer contains {_ARTIFACT_NONCE!r} after rollback (got {after_text!r})")
        # The whole cwd listing is unchanged: rollback created/removed no file.
        after_listing = sorted(os.listdir(workdir))
        assert after_listing == before_listing, (
            f"cwd listing changed across rollback ({before_listing} -> {after_listing}) "
            "— rollback must perform zero filesystem operations")

        _BODY_RAN["rollback"] = True
    finally:
        # THE leak fix, against a real child: aclose() closes the resident transport
        # (reaping the persistent child). Then OBSERVE (never signal) no orphan survived.
        await backend.aclose()
        leaked = _orphans_since(baseline_pids)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(codex_home, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)

    assert not os.path.exists(codex_home), "throwaway CODEX_HOME not removed"
    assert not os.path.exists(workdir), "throwaway workdir not removed"
    assert not leaked, (
        "orphan codex app-server --stdio process(es) survived backend.aclose() teardown: "
        f"{sorted(leaked)}")
    assert _BODY_RAN["rollback"], "live body did not run to completion (SKIP != PASS guard)"


class _Executor:
    """Dispatch ``item/tool/call`` to a name->handler map (the rollback test's single
    host tool). Mirrors what the core ToolRegistry would do, kept tiny because this
    test drives the backend directly rather than through ``run_task``."""

    def __init__(self, handlers: dict) -> None:
        self._handlers = handlers

    async def invoke(self, call: ToolCall) -> ToolResult:
        handler = self._handlers.get(call.name)
        if handler is None:  # pragma: no cover - prompt names only the declared tool
            return ToolResult(content=f"no such tool {call.name!r}", is_error=True)
        return await handler(call)


# --------------------------------------------------------------------------- #
# Test 2 — fork live smoke: a new, distinct thread id comes back                #
# --------------------------------------------------------------------------- #

@_skip
async def test_appserver_fork_live_smoke():
    """Real ``codex app-server``: drive ONE turn to a resident thread, then
    ``backend.fork(thread_id)`` returns a NEW, distinct thread id (ThreadForkResponse
    .thread.id). Cheap smoke — proves ``thread/fork`` is wired live and branches."""
    api_key = os.environ.get("LLM_API_KEY")
    assert api_key, "live test entered without LLM_API_KEY — gate misconfigured"

    codex_home = _make_isolated_codex_home()
    workdir = tempfile.mkdtemp(prefix="cfv2_appserver_control_fork_workdir_")
    baseline_pids = _codex_appserver_pids()

    saved_env = {k: os.environ.get(k) for k in ("LLM_API_KEY",)}
    os.environ["LLM_API_KEY"] = api_key

    # persist_thread=True so a rollout materializes on disk — thread/fork loads the
    # source thread from its rollout, so an EPHEMERAL thread (the default) would be
    # rejected with ``-32600 "no rollout found"`` (same footgun as rollback/resume).
    # aclose() archives the persisted thread on teardown (RT-APP-HOME-LEAK).
    backend = CodexAppServerBackend(
        mode="tools", persist_thread=True, codex_home=codex_home)
    try:
        # ONE cheap turn → a resident thread to fork from.
        handle = await backend.start(
            _no_tools_spec("Reply with exactly: ok", Path(workdir)), _NoToolExecutor())
        await _consume(handle)
        outcome = await handle.outcome()
        assert outcome.status == "completed", f"turn did not complete: {outcome.error!r}"
        resume = outcome.resume
        assert resume is not None and resume.token.get("thread_id"), (
            "turn produced no resume handle with a thread_id")
        thread_id = str(resume.token["thread_id"])
        assert thread_id in backend._live, "thread must be resident to fork it"

        # --- THE control call: fork the resident thread ---------------------------
        new_thread = await backend.fork(thread_id)
        assert new_thread, "thread/fork returned no new thread id"
        assert new_thread != thread_id, (
            f"fork must return a NEW, distinct thread id (got {new_thread!r} == source)")

        _BODY_RAN["fork"] = True
    finally:
        await backend.aclose()
        leaked = _orphans_since(baseline_pids)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(codex_home, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)

    assert not os.path.exists(codex_home), "throwaway CODEX_HOME not removed"
    assert not os.path.exists(workdir), "throwaway workdir not removed"
    assert not leaked, (
        "orphan codex app-server --stdio process(es) survived backend.aclose() teardown: "
        f"{sorted(leaked)}")
    assert _BODY_RAN["fork"], "live body did not run to completion (SKIP != PASS guard)"


# --------------------------------------------------------------------------- #
# Test 3 — steer live smoke (best-effort; race fail-fast is an ACCEPTABLE path) #
# --------------------------------------------------------------------------- #

@_skip
async def test_appserver_steer_live_best_effort():
    """Real ``codex app-server``: attempt to STEER an in-flight turn.

    Steering is timing-sensitive live: ``turn/steer`` requires the turn to STILL be the
    steerable active turn (``expectedTurnId`` precondition). A live turn may finish before
    we can steer. BOTH outcomes are valid and asserted:

    * **Steered in time** → ``steer`` returns a new (non-empty) active turn id.
    * **Race lost** (turn already completed / not steerable) → ``steer`` raises a CLEAN,
      NAMED fail-fast (a ``RuntimeError`` mentioning "steer"), NOT a hang or crash.

    Either way the steer is well-behaved; this test never flaky-FAILS on the race (the
    deterministic race cases are unit-covered in ``test_backend_control.py``). We record
    which path hit so the run output documents the live timing reality.
    """
    api_key = os.environ.get("LLM_API_KEY")
    assert api_key, "live test entered without LLM_API_KEY — gate misconfigured"

    codex_home = _make_isolated_codex_home()
    workdir = tempfile.mkdtemp(prefix="cfv2_appserver_control_steer_workdir_")
    baseline_pids = _codex_appserver_pids()

    saved_env = {k: os.environ.get(k) for k in ("LLM_API_KEY",)}
    os.environ["LLM_API_KEY"] = api_key

    backend = CodexAppServerBackend(mode="tools", codex_home=codex_home)
    # Recorded for the run report: "steered" | "race-fail-fast".
    path_hit = {"which": None, "detail": None}
    try:
        # Start a turn with a prompt that buys a little wall-clock (so there is a window
        # to steer the still-active turn). We do NOT consume the stream first — start()
        # returns right after turn/start responds, while the turn is still IN-FLIGHT.
        spec = _no_tools_spec(
            "Think step by step and then write a short paragraph (3-4 sentences) "
            "about why deterministic tests beat flaky ones.",
            Path(workdir))
        handle = await backend.start(spec, _NoToolExecutor())

        resume = handle._resume_handle()  # the resident-thread key (thread_id)
        assert resume is not None and resume.token.get("thread_id"), (
            "in-flight turn produced no resume handle with a thread_id")
        thread_id = str(resume.token["thread_id"])
        assert thread_id in backend._live, "thread must be resident to steer it"
        # The active turn id is captured by the handle from the turn/start response — it
        # is the REQUIRED expectedTurnId precondition for steer.
        expected_turn_id = handle._turn_id
        assert expected_turn_id, "turn/start did not yield a turn id to steer against"

        # Drain the turn's events concurrently (mirrors a real caller, who keeps consuming
        # the run while a control-plane caller may steer out-of-band on the same thread).
        consumer = asyncio.ensure_future(_consume(handle))

        # --- THE control call (best-effort): steer the in-flight turn --------------
        try:
            new_turn = await backend.steer(
                thread_id, "Actually, make it exactly two sentences.",
                expected_turn_id=expected_turn_id)
            # Steered in time → a new active turn id came back.
            assert new_turn, "steer was accepted but returned no new turn id"
            path_hit["which"] = "steered"
            path_hit["detail"] = new_turn
        except RuntimeError as exc:
            # Race lost: the turn finished / rotated before we could steer. This is the
            # CLEAN, NAMED fail-fast (not a hang/crash) — an ACCEPTABLE outcome. Assert
            # the error names the steer race rather than leaking an opaque code.
            msg = str(exc)
            assert "steer" in msg.lower(), (
                f"steer race fail-fast must name the steer race, got: {msg!r}")
            path_hit["which"] = "race-fail-fast"
            path_hit["detail"] = msg

        # Let the (possibly steered) turn settle; bounded so a hang surfaces as a test
        # timeout rather than wedging the suite.
        await asyncio.wait_for(consumer, timeout=180.0)
        out = await handle.outcome()
        # Whichever path hit, the turn reached a terminal cleanly (no transport crash).
        assert out.status in ("completed", "failed"), f"turn did not settle: {out!r}"

        assert path_hit["which"] in ("steered", "race-fail-fast"), "steer path not recorded"
        # Surface which live path hit (visible with -s / on failure) — documents the
        # real steer timing for the report.
        print(f"\n[steer-live] path={path_hit['which']!r} detail={path_hit['detail']!r} "
              f"turn_status={out.status!r}")

        _BODY_RAN["steer"] = True
    finally:
        await backend.aclose()
        leaked = _orphans_since(baseline_pids)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(codex_home, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)

    assert not os.path.exists(codex_home), "throwaway CODEX_HOME not removed"
    assert not os.path.exists(workdir), "throwaway workdir not removed"
    assert not leaked, (
        "orphan codex app-server --stdio process(es) survived backend.aclose() teardown: "
        f"{sorted(leaked)}")
    assert _BODY_RAN["steer"], "live body did not run to completion (SKIP != PASS guard)"
