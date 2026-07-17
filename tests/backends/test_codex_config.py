"""Codex config edge behaviors: RT-CFG-1 (model_map fail-fast) + RT-CFG-2 (timeout cleanup)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from tilldone.backends.codex import CodexBackend
from tilldone.backends.codex_exec import CodexRunHandle
from tilldone.core.enums import EventKind
from tilldone.core.spec import AgentRunSpec
from tilldone.core.tools import ToolCall, ToolResult
from tilldone.testing.conformance import collect_run


class _StubExecutor:
    async def invoke(self, call: ToolCall) -> ToolResult:
        return ToolResult(content="ok")


# ---------------- RT-CFG-1: model_map must fail-fast on Codex (never silently ignored) ----------------

async def test_codex_model_map_fails_fast(tmp_path):
    spec = AgentRunSpec(prompt="go", tools=(), cwd=tmp_path, model_map={"smart": "gpt-5.5"})
    with pytest.raises(ValueError):
        await asyncio.wait_for(CodexBackend().start(spec, _StubExecutor()), timeout=5)


async def test_codex_no_model_map_is_fine(tmp_path, monkeypatch):
    # sanity: without model_map, start() proceeds past the guard (it will try to bind a bridge;
    # we only assert it does NOT raise the model_map ValueError before that).
    spec = AgentRunSpec(prompt="go", tools=(), cwd=tmp_path)
    assert spec.model_map is None


# ---------------- RT-CFG-1 (LIVE, positive proof): the model Codex ACTUALLY ran is gpt-5.5 ----------------
#
# `test_codex_model_map_fails_fast` only proves model_map is REJECTED. This adds the POSITIVE
# half the brief asked for: prove a user cannot be DECEIVED about which model executed.
#
# Empirical finding (codex-cli 0.139.0, verified against RAW `codex exec --json` stdout): codex's
# event stream does NOT carry the model name. `thread.started` exposes only {thread_id};
# `turn.started` is empty; `turn.completed` exposes only `usage` (token counts). The model name
# (the configured gateway model) is written by codex itself into its SESSION ROLLOUT file
# ($CODEX_HOME/sessions/.../rollout-*-<thread_id>.jsonl) under `turn_context.payload.model` +
# `session_meta.payload.model_provider`. So the positive proof correlates the LIVE event-stream
# `thread_id` (surfaced in normalized metadata: RUN_STARTED.data["thread_id"] / .session_id /
# outcome.resume.token["session_id"]) with codex's own session record for THAT run, and asserts
# the model codex recorded == the configured real model. Nothing here is faked: codex wrote that
# model string while really executing the billed turn.

pytestmark_int = pytest.mark.integration
_GATE = os.environ.get("TILLDONE_CODEX_E2E") == "1"


def _live_codex_home() -> Path | None:
    """The isolated CODEX_HOME run-live.sh exports; rollout sessions are written under it."""
    home = os.environ.get("CODEX_HOME")
    return Path(home) if home else None


def _rollout_for_thread(codex_home: Path, thread_id: str) -> Path | None:
    # codex names each rollout `rollout-<timestamp>-<thread_id>.jsonl`; the thread_id is also the
    # session id recorded inside session_meta.payload.id — match on the filename suffix.
    matches = sorted(codex_home.glob(f"sessions/**/rollout-*-{thread_id}.jsonl"))
    return matches[-1] if matches else None


def _recorded_model_and_provider(rollout: Path) -> tuple[str | None, str | None]:
    """Pull the executed model + provider codex persisted for this session (no secrets here)."""
    model = provider = None
    for line in rollout.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        if obj.get("type") == "turn_context" and payload.get("model"):
            model = payload["model"]
        if obj.get("type") == "session_meta" and payload.get("model_provider"):
            provider = payload["model_provider"]
    return model, provider


@pytest.mark.integration
@pytest.mark.skipif(not _GATE, reason="set TILLDONE_CODEX_E2E=1 to run live Codex tests")
async def test_codex_e2e_executed_model_is_configured_gpt_5_5(tmp_path):
    """RT-CFG-1 (positive): a REAL codex turn ran the configured model gpt-5.5 — proven by
    correlating the live event-stream thread_id with codex's own session record for that run.

    Guards against a user being deceived about which model executed: the assertion is on data
    codex wrote while actually running the billed turn, keyed by the thread_id the event stream
    emitted — not on the prompt, the config we sent, or anything tautological.
    """
    codex_home = _live_codex_home()
    if codex_home is None or not codex_home.exists():
        pytest.skip("CODEX_HOME not set to the isolated live home (run via scripts/run-live.sh)")

    spec = AgentRunSpec(prompt="Reply with exactly the word ACK and nothing else.",
                        tools=(), cwd=tmp_path)
    # force low effort so the live turn is seconds, not minutes (config default is xhigh)
    backend = CodexBackend(extra_config=['model_reasoning_effort="low"'])
    events, outcome = await collect_run(backend, spec, _StubExecutor())

    # the run actually completed (SKIP != PASS; a failed/empty turn must not silently "pass")
    assert outcome.status == "completed", f"live codex turn did not complete: {outcome.error}"

    # thread_id comes from NORMALIZED EVENT METADATA (three independent surfaces must agree):
    run_started = next(e for e in events if e.kind is EventKind.RUN_STARTED)
    thread_id = run_started.data.get("thread_id")
    assert thread_id, "RUN_STARTED carried no thread_id in normalized metadata"
    assert run_started.session_id == thread_id
    assert outcome.resume is not None and outcome.resume.token.get("session_id") == thread_id

    # correlate that live thread_id with codex's OWN session record and read the executed model:
    rollout = _rollout_for_thread(codex_home, thread_id)
    assert rollout is not None, (
        f"no codex rollout session file found for live thread_id {thread_id} under {codex_home}")
    executed_model, provider = _recorded_model_and_provider(rollout)

    # POSITIVE PROOF: the model codex really executed is the configured gateway model.
    assert executed_model is not None, f"codex recorded no model in {rollout.name}"
    configured_model = os.environ.get("TILLDONE_GATEWAY_CODEX_MODEL")
    assert configured_model, "TILLDONE_GATEWAY_CODEX_MODEL is required for the live test"
    assert executed_model == configured_model, (
        f"deception guard: live codex executed model {executed_model!r}, expected "
        f"{configured_model!r} (the configured model) — rollout {rollout.name}")
    assert provider == "gateway", (
        f"expected gateway provider 'gateway', codex recorded {provider!r}")


# ---------------- RT-CFG-2: timeout kills the child + stops the bridge (no orphan) ----------------

class _EmptyStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _HangingProc:
    """A subprocess whose stdout never ends until killed — simulates a hung codex."""

    def __init__(self):
        self.returncode = None
        self.killed = False
        self._killed = asyncio.Event()
        self.stdout = self
        self.stderr = _EmptyStream()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self._killed.wait()  # block until killed
        raise StopAsyncIteration

    def kill(self):
        self.killed = True
        self.returncode = -9
        self._killed.set()

    async def wait(self):
        await self._killed.wait()
        return self.returncode


class _FakeBridge:
    def __init__(self):
        self.stopped = False

    async def stop(self):
        self.stopped = True


async def test_codex_timeout_kills_proc_and_stops_bridge():
    proc = _HangingProc()
    bridge = _FakeBridge()
    handle = CodexRunHandle(proc, bridge, timeout_s=0.3)
    events = [ev async for ev in handle.events]
    outcome = await handle.outcome()
    # no orphan: the watchdog killed the child AND the MCP bridge was torn down
    assert proc.killed is True and bridge.stopped is True
    assert events[-1].kind.value == "run_failed"
    assert outcome.status == "failed" and outcome.error.code == "timeout"
