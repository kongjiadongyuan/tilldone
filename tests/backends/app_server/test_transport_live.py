"""LIVE handshake smoke test for the productionized ``AppServerTransport``.

Run (via the live runner, which sets the gate + sources the gateway key)::

    bash scripts/run-live.sh tests/backends/app_server/test_transport_live.py -v

Or manually::

    TILLDONE_APPSERVER_E2E=1 LLM_API_KEY=... \
        uv run pytest tests/backends/app_server/test_transport_live.py -v

This is the only test that proves the *productionized* transport really talks to
a real ``codex app-server --stdio``: spawn + child-stdio wiring + newline framing
+ request/response multiplexing + the handshake, end to end, against the live
gateway. It drives the REAL :meth:`AppServerTransport.start` path (NOT the
``_attach`` in-memory test seam used by the unit tests).

Scope is **protocol-only** (M1): we go as far as ``thread/start`` and assert a
thread id comes back, then tear down. We do NOT run a turn — turns burn the
gateway and add model-quality flakiness; that is M2/M3's job. Spawn + framing +
multiplexing + handshake working live is exactly what M1 needs to prove.

SKIP != PASS (T1)
-----------------
This test is ``pytest.mark.integration``. The repo-root ``conftest.py`` counts
integration items collected vs actually-run and, when any live gate env is set,
**fails the session** if every integration test merely skipped (run-count==0).
``scripts/run-live.sh`` turns that gate on, so a green CI from 100%-skips is
impossible. Locally, with no gate env, the test skips cleanly via a real
``skipif`` (never a silent pass), and a body-executed sentinel
(``_BODY_RAN``) makes a falsely-"passed" empty body detectable.

No secret is ever printed or written to disk: ``LLM_API_KEY`` is read from the
ambient environment and passed straight into the child env; the ``config.toml``
written under the throwaway ``CODEX_HOME`` carries only the env-var *name*.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest

from tilldone.backends.app_server.transport import AppServerTransport

pytestmark = pytest.mark.integration

# Gate: BOTH the opt-in env flag AND a present gateway key are required. A
# missing key is a real skip (not a pass) — and under a gated live run the T1
# theater check in conftest turns an all-skip session into a failure.
_GATE_ENV = os.environ.get("TILLDONE_APPSERVER_E2E") == "1"
_HAS_KEY = bool(os.environ.get("LLM_API_KEY"))
_skip = pytest.mark.skipif(
    not (_GATE_ENV and _HAS_KEY),
    reason="set TILLDONE_APPSERVER_E2E=1 and LLM_API_KEY to run the live app-server handshake",
)

# Body-executed sentinel: flipped to True only when the live body actually runs
# to completion. A test that "passes" without flipping this would be an empty
# body falsely reported green — the trailing assertion guards against that.
_BODY_RAN = {"done": False}

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
    children this transport spawns, never an unrelated ``codex exec``.
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


def _make_isolated_codex_home() -> str:
    """Throwaway CODEX_HOME under /tmp containing the non-secret gateway config."""
    home = tempfile.mkdtemp(prefix="cfv2_appserver_live_codexhome_")
    os.chmod(home, 0o700)
    with open(os.path.join(home, "config.toml"), "w", encoding="utf-8") as fh:
        fh.write(_CONFIG_TOML)
    return home


@_skip
async def test_appserver_live_handshake():
    """Spawn a real ``codex app-server``, run initialize → initialized → thread/start.

    Proves, end to end against the live gateway, that the productionized
    transport: spawns the child, wires its stdio, frames newline-delimited JSON,
    multiplexes id-correlated responses, and completes the handshake. Asserts a
    thread id is returned at ``result.thread.id``.
    No turn is run (protocol-only handshake check).
    """
    # Sanity: the gate genuinely admitted us with a key present (defense in depth
    # vs the skipif). We read the key only to assert presence and to hand it to
    # the child env below — its value is never logged or asserted on.
    api_key = os.environ.get("LLM_API_KEY")
    assert api_key, "live test entered without LLM_API_KEY — gate misconfigured"

    codex_home = _make_isolated_codex_home()
    workdir = tempfile.mkdtemp(prefix="cfv2_appserver_live_workdir_")
    # Baseline BEFORE spawn so the orphan check diffs only processes we created.
    baseline_pids = _codex_appserver_pids()

    # Child env = inherit + isolated CODEX_HOME + the gateway key. Nothing here is
    # logged; the key flows straight into the subprocess environment.
    child_env = dict(os.environ)
    child_env["CODEX_HOME"] = codex_home
    child_env["LLM_API_KEY"] = api_key

    transport = AppServerTransport(env=child_env, cwd=workdir, request_timeout_s=60.0)

    try:
        # REAL production path: spawn codex app-server --stdio + wire its stdio.
        await transport.start()

        # 1) initialize — MUST carry experimentalApi (protocol observations) though we run
        #    no dynamicTools here; this mirrors the proven live handshake.
        init_result = await transport.request(
            "initialize",
            {
                "clientInfo": {"name": "tilldone-appserver-live", "version": "0.0.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        assert isinstance(init_result, dict), (
            f"initialize returned non-dict result: {init_result!r}\n"
            f"stderr tail:\n{transport.stderr_tail()}"
        )

        # 2) initialized notification (no response expected).
        await transport.notify("initialized", {})

        # 3) thread/start — string-form sandbox; ephemeral throwaway thread.
        ts_result = await transport.request(
            "thread/start",
            {
                "cwd": workdir,
                "sandbox": "danger-full-access",
                "approvalPolicy": "never",
                "ephemeral": True,
            },
        )
        assert isinstance(ts_result, dict), (
            f"thread/start returned non-dict result: {ts_result!r}\n"
            f"stderr tail:\n{transport.stderr_tail()}"
        )
        # Thread id lands at result.thread.id .
        thread = ts_result.get("thread")
        assert isinstance(thread, dict), (
            f"thread/start result missing a 'thread' object: keys={sorted(ts_result)}"
        )
        thread_id = thread.get("id")
        assert isinstance(thread_id, str) and thread_id, (
            f"thread/start result.thread.id absent/empty: thread={thread!r}"
        )

        # Body ran to completion — flip the sentinel (proves a real execution, not
        # a falsely-green empty body).
        _BODY_RAN["done"] = True
    finally:
        await transport.aclose()
        shutil.rmtree(codex_home, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)

    # --- post-teardown assertions (outside the finally so failures surface) ---
    assert not os.path.exists(codex_home), "throwaway CODEX_HOME not removed"
    assert not os.path.exists(workdir), "throwaway workdir not removed"

    # No orphaned app-server from THIS test: any codex app-server --stdio pid that
    # was not in the pre-spawn baseline must be gone after aclose().
    leaked = _codex_appserver_pids() - baseline_pids
    assert not leaked, f"orphan codex app-server --stdio process(es) survived teardown: {sorted(leaked)}"

    # SKIP != PASS sentinel: a green result here is only legitimate if the live
    # body actually executed end to end.
    assert _BODY_RAN["done"], "live body did not run to completion (SKIP != PASS guard)"
