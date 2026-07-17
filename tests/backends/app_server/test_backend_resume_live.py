"""LIVE same-backend cross-process RESUME proof for ``CodexAppServerBackend`` (M4 LaneB).

Run (via the live runner, which sets the gate + sources the gateway key)::

    bash scripts/run-live.sh tests/backends/app_server/test_backend_resume_live.py -v

Or manually::

    TILLDONE_APPSERVER_E2E=1 LLM_API_KEY=... \
        uv run pytest tests/backends/app_server/test_backend_resume_live.py -v

Live proof that ``CodexAppServerBackend(persist_thread=True)`` gives a working
same-backend resume across SEPARATE backend instances + SEPARATE app-server child
PROCESSES, with full context continuity (RT-SESS-1) AND a PRESERVED trust-full posture.

The proof shape (two backend INSTANCES here)
------------------------------------------------------------------------------------
* **Instance 1** (``persist_thread=True``) runs turn 1: the prompt makes the model
  STATE + remember a high-entropy token. Because the thread is non-ephemeral a rollout
  materializes on disk under the shared CODEX_HOME. From the turn-1 outcome we mint a
  :class:`ResumeHandle` whose token carries ``{thread_id, sandbox, approvalPolicy}``.
  Instance 1's transport is then closed WITHOUT archiving the thread (we close the
  transport directly, NOT via ``backend.aclose()``, so the rollout SURVIVES on disk —
  archiving here would defeat the resume).
* **Instance 2** (a SECOND ``CodexAppServerBackend``, separate process) is fed that
  ResumeHandle. The thread is NOT resident in it → it spawns a fresh ``codex
  app-server``, handshakes, and issues a native ``thread/resume`` RE-SUPPLYING
  ``sandbox`` + ``approvalPolicy`` (the probe2 silent-downgrade fix). It asserts the
  resume result is STILL NO GATING (else fail-fast). Turn 2 asks for the token back.

**The load-bearing proof.** The high-entropy token (``RESUME-PROBE-LIVE-7731``) was
stated ONLY in turn 1, in instance 1's now-dead process. The only way it can appear in
turn 2's answer (run by instance 2's fresh process) is if the persisted rollout was
genuinely rehydrated by ``thread/resume`` — i.e. real cross-process context continuity
(RT-SESS-1). And the run staying NO-GATING (instance 2 did not raise the downgrade
fail-fast, and no unexpected approval server-request landed) proves the re-supplied
policy preserved the posture, exactly as the probe predicted.

Teardown (RT-APP-HOME-LEAK) — via ``backend.aclose()``
------------------------------------------------------
Instance 2 owns the resumed (persisted) thread, so ``await be2.aclose()`` in the
``finally`` best-effort ``thread/archive``'s it before closing its transport (reaping
the rollout/daemon thread), then we OBSERVE (never signal) that no ``codex app-server
--stdio`` pid spawned by this test survived. Instance 1's transport was already closed.

SKIP != PASS (T1)
-----------------
``pytest.mark.integration`` + a real ``skipif`` gate. The repo-root conftest fails a
gated live session in which every integration test merely skipped, and the body-ran
sentinel makes a falsely-green empty body detectable. No secret is printed or written:
``LLM_API_KEY`` flows only into the child env; ``config.toml`` carries the env-var NAME.
The token is a fixed non-secret test string.
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

pytestmark = pytest.mark.integration

_GATE_ENV = os.environ.get("TILLDONE_APPSERVER_E2E") == "1"
_HAS_KEY = bool(os.environ.get("LLM_API_KEY"))
_skip = pytest.mark.skipif(
    not (_GATE_ENV and _HAS_KEY),
    reason="set TILLDONE_APPSERVER_E2E=1 and LLM_API_KEY to run the live app-server resume proof",
)

_BODY_RAN = {"done": False}

# The high-entropy token stated ONLY in turn 1 (instance 1's process). Its reappearance
# in turn 2 (instance 2's fresh process) is the cross-process continuity proof. Fixed
# non-secret string; the entropy makes an accidental match implausible.
_RESUME_TOKEN = "RESUME-PROBE-LIVE-7731"

# Non-secret gateway fixture config. Carries the
# provider name + base_url + the env-var NAME (env_key), never the key value.
_CONFIG_TOML = """model = "gw/model-a"
model_provider = "gateway"
[model_providers.gateway]
name = "gateway"
base_url = "https://gateway.example.invalid/v1"
env_key = "LLM_API_KEY"
wire_api = "responses"
"""


def _codex_appserver_pids() -> set[int]:
    """PIDs of live ``codex app-server --stdio`` processes (for baseline diff)."""
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
    bounded reap window. NEVER signals — a non-empty return is a real aclose() failure."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        survivors = _codex_appserver_pids() - baseline
        if not survivors:
            return set()
        time.sleep(0.1)
    return _codex_appserver_pids() - baseline


def _make_persistent_codex_home() -> str:
    """A throwaway-but-PERSISTENT (shared across both instances) CODEX_HOME under /tmp."""
    home = tempfile.mkdtemp(prefix="cfv2_appserver_resume_codexhome_")
    os.chmod(home, 0o700)
    with open(os.path.join(home, "config.toml"), "w", encoding="utf-8") as fh:
        fh.write(_CONFIG_TOML)
    return home


def _spec(prompt: str, cwd: Path, *, resume=None) -> AgentRunSpec:
    """A spec for a pure continuity turn: no tools, no output_schema. Driven in TOOLS
    mode (the MAIN path) so the model's free-text reply lands in ``outcome.final_text``
    WITHOUT being strict-JSON parsed — exactly the probe's raw-turn continuity shape (a
    plain ``ok`` / the recalled token, not a JSON contract payload)."""
    return AgentRunSpec(prompt=prompt, tools=(), cwd=cwd, output_schema=None,
                        resume=resume, timeout_s=180.0)


async def _run_one_turn(backend: CodexAppServerBackend, spec: AgentRunSpec):
    """Drive ONE turn to completion via the backend's own start()/events/outcome (NOT
    run_task — we need the raw outcome.resume handle + control over teardown timing).
    Returns the RunOutcome."""
    handle = await backend.start(spec, _NoToolExecutor())
    async for _ev in handle.events:
        pass
    return await handle.outcome()


class _NoToolExecutor:
    """No tools are declared in this continuity probe, so no ``item/tool/call`` can
    arrive → invoke must never be called."""

    async def invoke(self, call):  # pragma: no cover - must never run
        raise AssertionError("resume continuity probe declares no tools; invoke must not run")


@_skip
async def test_appserver_cross_instance_resume_continuity_and_no_gating_live():
    """Real ``codex app-server``: persist-mode turn 1 (instance 1) states a token; a
    SECOND backend instance resumes (fresh process, native thread/resume re-supplying
    policy) and the model RECALLS the token, with the run staying NO-GATING."""
    api_key = os.environ.get("LLM_API_KEY")
    assert api_key, "live test entered without LLM_API_KEY — gate misconfigured"

    codex_home = _make_persistent_codex_home()
    workdir = tempfile.mkdtemp(prefix="cfv2_appserver_resume_workdir_")
    baseline_pids = _codex_appserver_pids()

    # CODEX_HOME is explicit on the backend; only LLM_API_KEY is inherited by the
    # child so the isolated config can resolve its env_key. Never logged.
    saved_env = {k: os.environ.get(k) for k in ("LLM_API_KEY",)}
    os.environ["LLM_API_KEY"] = api_key

    # Instance 1: persist_thread=True so a rollout lands on disk and the resume token
    # carries the posture. Instance 2: a SEPARATE instance (separate process) that will
    # resume cross-process. TOOLS mode (the MAIN path) so a free-text reply completes
    # the turn without strict-JSON parsing (this is a raw continuity probe, not a
    # contract run — no tools are declared, so the model just answers in prose).
    be1 = CodexAppServerBackend(
        mode="tools", persist_thread=True, codex_home=codex_home)
    be2 = CodexAppServerBackend(
        mode="tools", persist_thread=True, codex_home=codex_home)

    try:
        # --- turn 1 (instance 1): state + remember the token --------------------
        out1 = await _run_one_turn(
            be1,
            _spec(f"Remember this token for later: {_RESUME_TOKEN}. Reply 'ok'.",
                  Path(workdir)),
        )
        assert out1.status == "completed", f"turn 1 did not complete: {out1.error!r}"
        resume = out1.resume
        assert resume is not None, "turn 1 produced no resume handle"
        # The persisted token MUST carry the posture for the cross-process re-supply.
        assert resume.backend == "codex_app_server"
        assert resume.token.get("thread_id"), "resume token missing thread_id"
        assert resume.token.get("sandbox") == "danger-full-access", (
            "persisted resume token must carry the sandbox posture to re-supply")
        assert resume.token.get("approvalPolicy") == "never", (
            "persisted resume token must carry the approvalPolicy posture to re-supply")

        # Close instance 1's transport WITHOUT archiving (archiving would delete the
        # rollout instance 2 needs). We close the resident transport directly, leaving
        # the persisted rollout on disk. (be1.aclose() would archive → defeat resume.)
        thread_id = str(resume.token["thread_id"])
        t1 = be1._live.get(thread_id)
        assert t1 is not None, "turn 1's thread should be resident in instance 1"
        await t1.aclose()
        be1._live.clear()
        be1._persisted.clear()  # forget it WITHOUT archiving — keep the rollout alive
        await asyncio.sleep(0.5)

        # --- turn 2 (instance 2): cross-process resume + recall the token -------
        out2 = await _run_one_turn(
            be2,
            _spec("What is the token I asked you to remember? "
                  "Reply with ONLY the token, nothing else.",
                  Path(workdir), resume=resume),
        )
        # If the posture had silently downgraded, _native_resume would have raised
        # (RuntimeError "downgraded the NO-GATING posture") — reaching here proves the
        # re-supplied policy preserved NO GATING, exactly as the probe predicted.
        assert out2.status == "completed", (
            f"resumed turn 2 did not complete: {out2.error!r}")

        # THE continuity proof (RT-SESS-1): the token stated only in turn 1 (instance
        # 1's dead process) is recalled in turn 2 (instance 2's fresh process) → the
        # persisted rollout was genuinely rehydrated by thread/resume cross-process.
        assert _RESUME_TOKEN in (out2.final_text or ""), (
            f"resumed model did not recall the turn-1 token {_RESUME_TOKEN!r} "
            f"(got {out2.final_text!r}) — cross-process context continuity failed")

        # NO GATING preserved: no unexpected approval/elicitation server-request landed
        # on instance 2's resumed connection (the outcome would be a protocol error if
        # one had — and it completed instead).
        assert out2.error is None

        _BODY_RAN["done"] = True
    finally:
        # Instance 2 owns the resumed persisted thread → aclose archives it
        # (RT-APP-HOME-LEAK) then closes the transport, reaping the child.
        await be2.aclose()
        # Instance 1 was already closed above; aclose() now is a harmless no-op.
        await be1.aclose()
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
        "orphan codex app-server --stdio process(es) survived teardown: "
        f"{sorted(leaked)}")
    assert _BODY_RAN["done"], "live body did not run to completion (SKIP != PASS guard)"
