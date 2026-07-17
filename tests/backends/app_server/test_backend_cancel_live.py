"""LIVE interrupt smoke for ``CodexAppServerBackend.cancel`` (M4 LaneC).

Run (via the live runner, which sets the gate + sources the gateway key)::

    bash scripts/run-live.sh tests/backends/app_server/test_backend_cancel_live.py -v

Or manually::

    TILLDONE_APPSERVER_E2E=1 LLM_API_KEY=... \
        uv run pytest tests/backends/app_server/test_backend_cancel_live.py -v

This is the live companion to ``test_backend_cancel.py``. It proves cancel against
a REAL ``codex app-server --stdio``: start a real turn, fire ``cancel()`` mid-flight
(the ``turn/interrupt`` REQUEST + aclose), and assert (a) cancel returns cleanly and
quickly, and (b) NO orphaned ``codex app-server --stdio`` child survives teardown.

We do NOT assert the turn reached ``status=interrupted``: a cheap turn can finish
before the interrupt lands, which is an ACCEPTABLE state-3 path (interrupt is then a
harmless no-op). Either way the load-bearing properties hold — cancel does not raise,
does not hang, and reaps the child. So this smoke is robust to timing without being a
theater test: it drives a real child + real interrupt + real teardown.

SKIP != PASS (T1): ``pytest.mark.integration`` + a real ``skipif`` gate, and the
repo-root conftest fails a gated session whose integration tests all merely skipped.
A body-executed sentinel (``_BODY_RAN``) makes a falsely-green empty body detectable.

No secret is printed or written: ``LLM_API_KEY`` flows only into the child env; the
throwaway ``CODEX_HOME`` config carries only the env-var NAME (``env_key``).
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
from tilldone.core.tools import ToolResult

pytestmark = pytest.mark.integration

_GATE_ENV = os.environ.get("TILLDONE_APPSERVER_E2E") == "1"
_HAS_KEY = bool(os.environ.get("LLM_API_KEY"))
_skip = pytest.mark.skipif(
    not (_GATE_ENV and _HAS_KEY),
    reason="set TILLDONE_APPSERVER_E2E=1 and LLM_API_KEY to run the live app-server cancel smoke",
)

_BODY_RAN = {"done": False}

# Non-secret gateway fixture config for the
# isolated CODEX_HOME (provider name + base_url + env-var NAME, never the key value).
_CONFIG_TOML = """model = "gw/model-a"
model_provider = "gateway"
[model_providers.gateway]
name = "gateway"
base_url = "https://gateway.example.invalid/v1"
env_key = "LLM_API_KEY"
wire_api = "responses"
"""


class _NullExecutor:
    """No host tools — the structured-mode smoke needs no executor dispatch."""

    async def invoke(self, call) -> ToolResult:  # pragma: no cover - never called
        raise AssertionError("executor must not be invoked in the cancel smoke")


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
    """``codex app-server --stdio`` pid(s) spawned since ``baseline`` still alive,
    giving the OS a bounded window to reap children cancel/aclose SIGKILLed.
    OBSERVES only — never signals; a survivor is a genuine teardown failure."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        survivors = _codex_appserver_pids() - baseline
        if not survivors:
            return set()
        time.sleep(0.1)
    return _codex_appserver_pids() - baseline


def _make_isolated_codex_home() -> str:
    home = tempfile.mkdtemp(prefix="cfv2_appserver_cancel_codexhome_")
    os.chmod(home, 0o700)
    with open(os.path.join(home, "config.toml"), "w", encoding="utf-8") as fh:
        fh.write(_CONFIG_TOML)
    return home


@_skip
async def test_appserver_cancel_midflight_live():
    """Start a real turn, cancel mid-flight, assert clean + quick teardown + no orphan.

    The prompt asks for a long, slow answer so there is something in flight to
    interrupt. We do NOT consume the event stream to completion — we fire cancel
    shortly after the turn starts. cancel() issues the ``turn/interrupt`` REQUEST
    (short timeout + suppress) then aclose()s the resident transport (reaping the
    child). We assert cancel does not raise, returns well under a real request
    timeout, and leaves no orphaned ``codex app-server --stdio`` process.
    """
    api_key = os.environ.get("LLM_API_KEY")
    assert api_key, "live test entered without LLM_API_KEY — gate misconfigured"

    codex_home = _make_isolated_codex_home()
    workdir = tempfile.mkdtemp(prefix="cfv2_appserver_cancel_workdir_")
    baseline_pids = _codex_appserver_pids()

    saved_env = {k: os.environ.get(k) for k in ("LLM_API_KEY",)}
    os.environ["LLM_API_KEY"] = api_key

    # Structured mode keeps the smoke minimal (no host tools to wire) — cancel's
    # turn/interrupt + aclose path is identical across modes (it lives on the handle).
    be = CodexAppServerBackend(mode="structured", codex_home=codex_home)
    spec = AgentRunSpec(
        prompt=(
            "Write a very long, detailed essay (at least 800 words) about the history "
            "of computing, taking your time and going into depth on every decade."
        ),
        tools=(),
        cwd=Path(workdir),
        output_schema=None,
        timeout_s=120.0,
    )

    leaked: set[int] = set()
    try:
        handle = await be.start(spec, _NullExecutor())

        # Start consuming events in the background so the turn genuinely runs, then
        # give it a moment to get in-flight before interrupting.
        async def _drain():
            try:
                async for _ in handle.events:
                    pass
            except Exception:
                pass

        drain_task = asyncio.ensure_future(_drain())
        await asyncio.sleep(1.5)  # let the turn start producing

        # cancel mid-flight: turn/interrupt REQUEST (short timeout + suppress) + aclose.
        # Must return cleanly and quickly even if the model is mid-generation.
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await asyncio.wait_for(handle.cancel(), timeout=10.0)
        elapsed = loop.time() - t0
        assert elapsed < 10.0, f"cancel hung {elapsed:.2f}s against a live peer"

        # The drain task ends once aclose terminates the notifications iterator.
        await asyncio.wait_for(drain_task, timeout=5.0)

        # cancel is idempotent against a real (now-closed) transport.
        await asyncio.wait_for(handle.cancel(), timeout=5.0)

        _BODY_RAN["done"] = True
    finally:
        # backend.aclose() is idempotent; cancel already closed the resident
        # transport, but the production Flow/harness layer calls aclose in finally.
        await be.aclose()
        leaked = _orphans_since(baseline_pids)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(codex_home, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)

    assert not leaked, (
        "orphan codex app-server --stdio process(es) survived cancel + aclose "
        f"teardown: {sorted(leaked)}")
    assert _BODY_RAN["done"], "live body did not run to completion (SKIP != PASS guard)"
