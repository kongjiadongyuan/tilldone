"""Unit + RT-APP-SKILLS-ROOT tests for the app-server skills surface (M6 LaneC).

Scope (the lead's scope-check ruling, see this lane's report): app-server skills
ONLY. Claude has NO skills wiring — its ``capabilities().skills=True`` is a bare
flag and ``claude-agent-sdk`` exposes only a per-session *enablement* list
(``ClaudeAgentOptions.skills``), not a ``skills/list`` discovery API nor an
extra-roots registration call. The two backends' skills models are structurally
asymmetric, so full Claude+AppServer skills *parity* is a separate Claude-side
lane (deferred exactly as ``dialog`` was in M5). These tests therefore exercise
the app-server side, which is well-defined by the v2 schema.

The skills surface is a **capability-scoped CONTROL PLANE** (like steer / fork /
rollback): reachable only via the :class:`~tilldone.core.backend.SkillsCapable`
optional protocol, NEVER from ``run_task``. It operates on a RESIDENT thread's
transport — the same ``_live`` registry the multi-round contract loop reuses —
because that is the only ``codex app-server`` connection the backend owns.

Wire shapes (live-exact, from the v2 schema):
* ``skills/list`` (``SkillsListParams`` {cwds?, forceReload?}) → ``SkillsListResponse``
  {data: [{cwd, errors, skills:[SkillMetadata]}]}.
* ``skills/extraRoots/set`` (``SkillsExtraRootsSetParams`` {extraRoots:[AbsolutePathBuf]})
  → ``SkillsExtraRootsSetResponse`` ({}). **GLOBAL, not thread-scoped** (RT-APP-SKILLS-ROOT).
* ``skills/changed`` (notification, empty params) → invalidation signal.

All tests drive a *real* :class:`AppServerTransport` over the in-memory ``FakePeer``
(no codex spawn, no gateway). They reuse the M2 handshake helpers from
``test_backend`` so a skills call rides a genuinely resident transport.

asyncio_mode="auto" (pyproject) → async tests need no decorator.
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
from tilldone.backends.app_server.events import map_notification
from tilldone.core.backend import SkillsCapable
from tilldone.core.enums import EventKind

from .fake_peer import FakePeer
from .test_backend import THREAD_ID, _NullExecutor, _serve_handshake, _spec

# A SkillsListResponse exactly as the v2 schema describes: one entry per cwd, each
# carrying its discovered skills (SkillMetadata) + any per-cwd discovery errors.
_SKILL_NAME = "summarize-pr"
_SKILLS_LIST_RESULT = {
    "data": [
        {
            "cwd": "/work/repo",
            "errors": [],
            "skills": [
                {
                    "name": _SKILL_NAME,
                    "description": "Summarize a pull request",
                    "enabled": True,
                    "path": "/work/repo/.codex/skills/summarize-pr",
                    "scope": "repo",
                },
                {
                    "name": "lint-fix",
                    "description": "Run the linter and fix",
                    "enabled": False,
                    "path": "/work/repo/.codex/skills/lint-fix",
                    "scope": "repo",
                },
            ],
        }
    ]
}


async def _resident_backend(tmp_path: Path) -> tuple[CodexAppServerBackend, FakePeer]:
    """Start a backend over the fake peer so it owns ONE resident transport+thread.

    Returns the live ``(backend, peer)``; the caller drives subsequent skills/*
    requests against the resident transport and is responsible for ``aclose()``.
    """
    peer = FakePeer()
    backend = CodexAppServerBackend(
        transport_factory=lambda **kw: peer.make_transport(**kw), mode="tools")
    driver = asyncio.ensure_future(_serve_handshake(peer))
    await backend.start(_spec(tmp_path), _NullExecutor())
    await asyncio.wait_for(driver, timeout=2.0)
    return backend, peer


# --------------------------------------------------------------------------- #
# SkillsCapable protocol — the capability-scoped seam                          #
# --------------------------------------------------------------------------- #

def test_backend_is_skills_capable():
    """The app-server backend satisfies the SkillsCapable optional protocol.

    This is the discovery seam (``isinstance(backend, SkillsCapable)``) — the SAME
    runtime_checkable pattern as SteerCapable/BranchingCapable — so a skills caller
    finds the capability WITHOUT core importing the backend.
    """
    backend = CodexAppServerBackend(mode="tools")
    assert isinstance(backend, SkillsCapable)


# --------------------------------------------------------------------------- #
# events.py — skills/changed mapping                                           #
# --------------------------------------------------------------------------- #

def test_map_skills_changed_to_backend_notice():
    """``skills/changed`` (empty params) surfaces as an observable BACKEND_NOTICE.

    Per the schema it is an invalidation signal ("re-run skills/list"); it must be
    observable (a notice), never silently dropped as telemetry noise.
    """
    ev = map_notification({"method": "skills/changed", "params": None}, seq=7)
    assert ev is not None
    assert ev.kind == EventKind.BACKEND_NOTICE
    assert ev.data["notice"] == "skills_changed"


# --------------------------------------------------------------------------- #
# skills/list — discovery + parsing                                            #
# --------------------------------------------------------------------------- #

async def test_skills_list_parses_discovered_skills(tmp_path):
    """``skills_list()`` issues ``skills/list`` and parses the discovered skills."""
    backend, peer = await _resident_backend(tmp_path)
    try:
        fut = asyncio.ensure_future(backend.skills_list(THREAD_ID))
        req = await peer.read_client_message()
        assert req["method"] == "skills/list"
        peer.respond(req["id"], _SKILLS_LIST_RESULT)
        entries = await asyncio.wait_for(fut, timeout=2.0)

        # One entry per cwd; the entry carries the discovered SkillMetadata verbatim.
        assert len(entries) == 1
        entry = entries[0]
        assert entry["cwd"] == "/work/repo"
        assert entry["errors"] == []
        names = [s["name"] for s in entry["skills"]]
        assert names == [_SKILL_NAME, "lint-fix"]
        # enabled/scope are preserved (the discovery + enablement view).
        assert entry["skills"][0]["enabled"] is True
        assert entry["skills"][1]["enabled"] is False
    finally:
        await backend.aclose()


async def test_skills_list_forwards_cwds_and_force_reload(tmp_path):
    """``skills_list(cwds=..., force_reload=True)`` forwards both wire params.

    ``cwds`` defaults to the session cwd when empty (so it is OMITTED when not
    supplied); ``forceReload`` bypasses the skills cache.
    """
    backend, peer = await _resident_backend(tmp_path)
    try:
        fut = asyncio.ensure_future(
            backend.skills_list(THREAD_ID, cwds=["/a", "/b"], force_reload=True))
        req = await peer.read_client_message()
        assert req["method"] == "skills/list"
        assert req["params"]["cwds"] == ["/a", "/b"]
        assert req["params"]["forceReload"] is True
        peer.respond(req["id"], {"data": []})
        await asyncio.wait_for(fut, timeout=2.0)
    finally:
        await backend.aclose()


async def test_skills_list_omits_empty_params(tmp_path):
    """With no cwds + no force_reload, ``skills/list`` sends NO (or empty) params.

    The schema makes both optional (empty cwds ⇒ "current session cwd"); we must
    not invent a ``cwds`` key or a falsey ``forceReload``.
    """
    backend, peer = await _resident_backend(tmp_path)
    try:
        fut = asyncio.ensure_future(backend.skills_list(THREAD_ID))
        req = await peer.read_client_message()
        assert req["method"] == "skills/list"
        params = req.get("params") or {}
        assert "cwds" not in params
        assert "forceReload" not in params
        peer.respond(req["id"], {"data": []})
        assert await asyncio.wait_for(fut, timeout=2.0) == []
    finally:
        await backend.aclose()


async def test_skills_list_not_resident_fails_fast(tmp_path):
    """``skills_list`` on a thread this backend does not own fails fast.

    Skills operate on a resident transport (the only connection the backend owns);
    an unknown thread has nothing to drive, so it raises rather than silently no-op.
    """
    backend = CodexAppServerBackend(mode="tools")
    with pytest.raises(RuntimeError, match="not resident"):
        await backend.skills_list("nonexistent-thread")


# --------------------------------------------------------------------------- #
# skills/extraRoots/set — registration + absolute-path enforcement             #
# --------------------------------------------------------------------------- #

async def test_set_skills_extra_roots_issues_wire_call(tmp_path):
    """``set_skills_extra_roots`` issues ``skills/extraRoots/set`` with the roots.

    The schema requires AbsolutePathBuf, so the roots are absolutized on the wire.
    """
    backend, peer = await _resident_backend(tmp_path)
    roots_dir = tmp_path / "skills_root"
    roots_dir.mkdir()
    try:
        fut = asyncio.ensure_future(
            backend.set_skills_extra_roots(THREAD_ID, [str(roots_dir)]))
        req = await peer.read_client_message()
        assert req["method"] == "skills/extraRoots/set"
        assert req["params"]["extraRoots"] == [os.path.abspath(str(roots_dir))]
        peer.respond(req["id"], {})
        await asyncio.wait_for(fut, timeout=2.0)
    finally:
        await backend.aclose()


async def test_set_skills_extra_roots_absolutizes_relative_path(tmp_path):
    """A relative root is made absolute before it hits the wire (schema requires it)."""
    backend, peer = await _resident_backend(tmp_path)
    try:
        fut = asyncio.ensure_future(
            backend.set_skills_extra_roots(THREAD_ID, ["rel/skills"]))
        req = await peer.read_client_message()
        assert req["method"] == "skills/extraRoots/set"
        sent = req["params"]["extraRoots"]
        assert len(sent) == 1
        assert os.path.isabs(sent[0]), f"extraRoot must be absolute: {sent[0]!r}"
        peer.respond(req["id"], {})
        await asyncio.wait_for(fut, timeout=2.0)
    finally:
        await backend.aclose()


async def test_clear_skills_extra_roots_sends_empty(tmp_path):
    """``clear_skills_extra_roots`` resets the GLOBAL extra roots to [].

    This is the RT-APP-SKILLS-ROOT teardown primitive: extraRoots is global (not
    thread-scoped), so a test/caller MUST reset it after use to avoid polluting a
    shared CODEX_HOME's resume/list/skills state.
    """
    backend, peer = await _resident_backend(tmp_path)
    try:
        fut = asyncio.ensure_future(backend.clear_skills_extra_roots(THREAD_ID))
        req = await peer.read_client_message()
        assert req["method"] == "skills/extraRoots/set"
        assert req["params"]["extraRoots"] == []
        peer.respond(req["id"], {})
        await asyncio.wait_for(fut, timeout=2.0)
    finally:
        await backend.aclose()


# --------------------------------------------------------------------------- #
# RT-APP-SKILLS-ROOT — global-state hygiene (set then teardown clears)         #
# --------------------------------------------------------------------------- #

async def test_rt_app_skills_root_global_state_hygiene(tmp_path):
    """RT-APP-SKILLS-ROOT: set extra roots, then teardown CLEARS them.

    ``skills/extraRoots/set`` is GLOBAL (not thread-scoped), so a run that registers
    a roots dir MUST reset extraRoots on teardown or it pollutes a shared CODEX_HOME
    (the next list/resume/skills sees stale roots). This test simulates that
    lifecycle at the unit level: register a temp roots dir, then assert the teardown
    issues a ``skills/extraRoots/set`` with ``extraRoots: []`` (the reset). We capture
    BOTH wire calls in order to prove the set was actually undone — not merely that a
    clear exists in isolation.
    """
    backend, peer = await _resident_backend(tmp_path)
    roots_dir = tmp_path / "ephemeral_skills_root"
    roots_dir.mkdir()
    wire_calls: list[tuple[str, object]] = []

    async def _capture_extra_roots_calls(n: int) -> None:
        for _ in range(n):
            req = await peer.read_client_message()
            assert req["method"] == "skills/extraRoots/set"
            wire_calls.append((req["method"], req["params"]["extraRoots"]))
            peer.respond(req["id"], {})

    try:
        cap = asyncio.ensure_future(_capture_extra_roots_calls(2))
        # 1) register the temp roots dir (the "use a temp roots dir" half).
        await backend.set_skills_extra_roots(THREAD_ID, [str(roots_dir)])
        # 2) teardown: reset extraRoots so the global state is left clean.
        await backend.clear_skills_extra_roots(THREAD_ID)
        await asyncio.wait_for(cap, timeout=2.0)

        # Both calls landed, IN ORDER: first the set (non-empty), then the clear ([]).
        assert len(wire_calls) == 2
        assert wire_calls[0][1] == [os.path.abspath(str(roots_dir))]
        assert wire_calls[1][1] == [], (
            "RT-APP-SKILLS-ROOT: teardown did not reset the GLOBAL extraRoots to [] — "
            "a registered skills root would leak into a shared CODEX_HOME")
    finally:
        await backend.aclose()


# =========================================================================== #
# LIVE deterministic skills fixture (env-gated; SKIP != PASS)                 #
#                                                                             #
# Proves the app-server skills surface against a REAL ``codex app-server``:    #
# register a TEMP roots dir holding a known SKILL.md, ``skills/list`` discovers #
# it, then teardown CLEARS extraRoots (RT-APP-SKILLS-ROOT global-state          #
# hygiene) and removes the temp CODEX_HOME. Run via:                          #
#                                                                             #
#   bash scripts/run-live.sh tests/backends/app_server/test_backend_skills.py -v
# =========================================================================== #

_GATE_ENV = os.environ.get("TILLDONE_APPSERVER_E2E") == "1"
_HAS_KEY = bool(os.environ.get("LLM_API_KEY"))
_skip_live = pytest.mark.skipif(
    not (_GATE_ENV and _HAS_KEY),
    reason="set TILLDONE_APPSERVER_E2E=1 and LLM_API_KEY to run the live app-server skills fixture",
)

# Body-executed sentinel: flipped True only when the live body runs to completion.
# A test that "passes" without flipping this would be a falsely-green empty body.
_LIVE_BODY_RAN = {"done": False}

# A known, deterministic skill the live fixture plants on disk. The high-entropy
# token makes an accidental/pre-existing match implausible. SKILL.md carries YAML
# frontmatter (name/description) — the codex skill-discovery layout.
_LIVE_SKILL_NAME = "cfv2-skill-zulu7731"
_LIVE_SKILL_DESC = "TillDone v2 deterministic skills-fixture probe (ZULU-7731)."
_SKILL_MD = f"""---
name: {_LIVE_SKILL_NAME}
description: {_LIVE_SKILL_DESC}
---

# {_LIVE_SKILL_NAME}

This is a throwaway test skill used to prove app-server skills/list discovery.
"""

# Non-secret gateway fixture config shared with the tools-live test.
# Carries the provider name + base_url + the env-var NAME (env_key), never the value.
_CONFIG_TOML = """model = "gw/model-a"
model_provider = "gateway"
[model_providers.gateway]
name = "gateway"
base_url = "https://gateway.example.invalid/v1"
env_key = "LLM_API_KEY"
wire_api = "responses"
"""


def _codex_appserver_pids() -> set[int]:
    """PIDs of any live ``codex app-server --stdio`` processes (for baseline diff)."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", "codex app-server --stdio"],
            capture_output=True, text=True, check=False,
        ).stdout
    except Exception:
        return set()
    return {int(tok) for tok in out.split() if tok.strip().isdigit()}


def _orphans_since(baseline: set[int], *, timeout_s: float = 3.0) -> set[int]:
    """Any ``codex app-server --stdio`` pid spawned since ``baseline`` still alive
    (observes only; reaping is owned by ``backend.aclose()``)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        survivors = _codex_appserver_pids() - baseline
        if not survivors:
            return set()
        time.sleep(0.1)
    return _codex_appserver_pids() - baseline


def _make_isolated_codex_home() -> str:
    """Throwaway CODEX_HOME under /tmp with the non-secret gateway config."""
    home = tempfile.mkdtemp(prefix="cfv2_appserver_skills_codexhome_")
    os.chmod(home, 0o700)
    with open(os.path.join(home, "config.toml"), "w", encoding="utf-8") as fh:
        fh.write(_CONFIG_TOML)
    return home


def _make_skill_root() -> str:
    """Throwaway extra-roots dir holding ``<skill>/SKILL.md`` (the planted skill)."""
    root = tempfile.mkdtemp(prefix="cfv2_appserver_skills_root_")
    skill_dir = Path(root) / _LIVE_SKILL_NAME
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
    return root


@_skip_live
@pytest.mark.integration
async def test_appserver_skills_list_discovers_planted_skill_live():
    """Real ``codex app-server`` discovers a planted skill via the extra-roots surface.

    Lifecycle (RT-APP-SKILLS-ROOT global-state hygiene, on a TEMP CODEX_HOME):

      backend.start()                         # resident thread + transport
      set_skills_extra_roots([temp_root])     # GLOBAL registration
      skills/list(force_reload=True)          # re-scan from disk → discover the skill
      assert the planted skill name is present
      finally:
        clear_skills_extra_roots()            # reset GLOBAL extraRoots (the hygiene)
        backend.aclose()                      # reap the persistent child
        rm temp CODEX_HOME + temp root        # no disk residue
        assert no orphan app-server survived

    The planted skill (``cfv2-skill-zulu7731``) can only appear in ``skills/list``
    output if the temp roots dir was REALLY registered and re-scanned by the live
    server — proving the extra-roots + list surface end to end (not a mock).
    """
    api_key = os.environ.get("LLM_API_KEY")
    assert api_key, "live test entered without LLM_API_KEY — gate misconfigured"

    codex_home = _make_isolated_codex_home()
    skill_root = _make_skill_root()
    workdir = tempfile.mkdtemp(prefix="cfv2_appserver_skills_workdir_")
    baseline_pids = _codex_appserver_pids()

    # CODEX_HOME is explicit on the backend; only LLM_API_KEY is inherited by the
    # child so the isolated config can resolve its env_key. The key value is never logged.
    saved_env = {k: os.environ.get(k) for k in ("LLM_API_KEY",)}
    os.environ["LLM_API_KEY"] = api_key

    backend = CodexAppServerBackend(mode="tools", codex_home=codex_home)
    thread_id: str | None = None
    leaked: set[int] = set()
    try:
        # Start a resident thread (a trivial turn is issued by start(); we never drive
        # it to completion — skills/list is independent of the in-flight turn). The
        # ResumeHandle.token carries the thread_id we then act on.
        spec = _live_spec(Path(workdir))
        handle = await backend.start(spec, _NullExecutor())
        outcome_resume = handle._resume_handle()  # thread id without awaiting the turn
        assert outcome_resume is not None, "start() yielded no thread id"
        thread_id = outcome_resume.token["thread_id"]

        # Register the temp roots dir (GLOBAL), then re-scan from disk.
        await backend.set_skills_extra_roots(thread_id, [skill_root])
        entries = await backend.skills_list(thread_id, force_reload=True)

        # The planted skill must be discoverable in SOME entry's skills list.
        discovered = {
            s.get("name")
            for entry in entries
            for s in (entry.get("skills") or [])
        }
        assert _LIVE_SKILL_NAME in discovered, (
            f"planted skill {_LIVE_SKILL_NAME!r} not in discovered skills {sorted(discovered)!r} "
            "— extra-roots registration + skills/list re-scan did not surface it"
        )
        _LIVE_BODY_RAN["done"] = True
    finally:
        # RT-APP-SKILLS-ROOT teardown: reset the GLOBAL extraRoots so the registration
        # does not leak (best-effort — the temp CODEX_HOME is also discarded, but the
        # explicit clear is the hygiene contract a real caller must honor).
        if thread_id is not None:
            try:
                await backend.clear_skills_extra_roots(thread_id)
            except Exception:
                pass
        await backend.aclose()
        leaked = _orphans_since(baseline_pids)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(codex_home, ignore_errors=True)
        shutil.rmtree(skill_root, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)

    # Post-teardown assertions (outside finally so failures surface).
    assert not os.path.exists(codex_home), "throwaway CODEX_HOME not removed"
    assert not os.path.exists(skill_root), "throwaway skill root not removed"
    assert not leaked, (
        "orphan codex app-server --stdio process(es) survived backend.aclose(): "
        f"{sorted(leaked)}")
    assert _LIVE_BODY_RAN["done"], "live body did not run to completion (SKIP != PASS guard)"


def _live_spec(cwd: Path):
    """A minimal AgentRunSpec for the live skills fixture (no output schema needed).

    Kept separate from ``_spec`` (which always sets an output schema for the
    structured-mode unit tests). Here the turn is never driven to completion — we only
    need ``thread/start`` to make a resident thread the skills surface can act on.
    """
    from tilldone.core.spec import AgentRunSpec
    return AgentRunSpec(
        prompt="Wait for further instructions.",
        tools=(),
        cwd=cwd,
    )
