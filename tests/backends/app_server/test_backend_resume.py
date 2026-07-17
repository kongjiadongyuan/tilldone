"""Same-backend RESUME tests for CodexAppServerBackend.

Cross-process resume: a NON-ephemeral thread + ``thread/resume`` RE-SUPPLYING
``sandbox`` + ``approvalPolicy`` gives, across a process restart: a working resume,
full context continuity, AND a PRESERVED trust-full posture (resume result top-level
``approvalPolicy:"never"`` + ``sandbox:{type:"dangerFullAccess"}``). OMITTING the
policy on resume silently downgrades, so this backend always re-supplies it and
FAILS FAST if the resume result reports a downgraded (gated) posture.

The design landed here (backend.py only):

* ``persist_thread=True`` → ``thread/start`` sends ``ephemeral:false`` so a rollout
  persists; the resume token then carries the NO-GATING posture.
* Within-run RESIDENT reuse stays FIRST (a resident thread_id → another turn/start,
  NO thread/resume) — unchanged from LaneD.
* Resident MISS + token HAS posture (persisted) → fresh transport + native
  ``thread/resume`` (re-supplying policy) + NO-GATING assertion + resident.
* Resident MISS + NO posture (ephemeral) → the M2 fail-fast (unchanged).
* ``context_dirs`` → ``runtimeWorkspaceRoots`` (absolute) on thread/start AND resume.
* ``aclose()`` best-effort ``thread/archive``'s persisted threads (RT-APP-HOME-LEAK).

All tests drive a *real* :class:`AppServerTransport` through the in-memory
``FakePeer`` (NO codex spawn, NO gateway). ``asyncio_mode="auto"`` (pyproject) →
async tests need no decorator.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tilldone.backends.app_server.backend import CodexAppServerBackend
from tilldone.backends.app_server.transport import JsonRpcError
from tilldone.core.spec import AgentRunSpec, ContextDirView, ResumeHandle

from .fake_peer import FakePeer
from .test_backend import (
    FINAL_JSON_TEXT,
    THREAD_ID,
    TURN_ID,
    _NullExecutor,
    _notify_final_agent_message,
    _notify_token_usage,
    _notify_turn_completed,
    _notify_turn_started,
    _respond_initialize,
)
from .test_backend_multiround import (
    FINAL_JSON_TEXT_2,
    TURN_ID_2,
    _CountingFactory,
    _collect,
)

# The NO-GATING posture as the resume RESULT reports it at the TOP LEVEL — exactly the
# shape probe_resume.jsonl seq7 carried (sandbox is an OBJECT tagged by `type`,
# approvalPolicy a STRING). Re-supplying policy on resume keeps this preserved.
_NO_GATING_RESULT = {
    "approvalPolicy": "never",
    "sandbox": {"type": "dangerFullAccess"},
    "activePermissionProfile": None,
}
# The probe2 silent-downgrade posture: resume reported these when policy was NOT
# re-supplied. A resume that reports THIS (despite our re-supply) is a fail-fast.
_DOWNGRADED_RESULT = {
    "approvalPolicy": "on-request",
    "sandbox": {"type": "workspaceWrite"},
    "activePermissionProfile": ":workspace",
}


def _resume_spec(tmp_path: Path, *, token: dict, **kw) -> AgentRunSpec:
    """An AgentRunSpec carrying a resume handle for THIS backend with ``token``."""
    return AgentRunSpec(
        prompt=kw.pop("prompt", "Continue the task."),
        tools=(),
        cwd=tmp_path,
        output_schema=kw.pop("output_schema", None),
        resume=ResumeHandle("codex_app_server", token),
        **kw,
    )


def _persisted_token(thread_id: str = THREAD_ID) -> dict:
    """The resume token a persisted run mints: thread_id + the NO-GATING posture."""
    return {"thread_id": thread_id, "sandbox": "danger-full-access",
            "approvalPolicy": "never"}


async def _serve_persist_handshake(peer: FakePeer, *, thread_id=THREAD_ID,
                                   expect_ephemeral=False, want_roots=None):
    """Answer initialize → initialized → thread/start → turn/start for a PERSIST run,
    asserting the thread/start carried ``ephemeral`` == ``expect_ephemeral`` and
    (optionally) ``runtimeWorkspaceRoots`` == ``want_roots``. Returns the parsed
    thread/start client message so a test can make further assertions on it."""
    init = await peer.read_client_message()
    assert init["method"] == "initialize"
    _respond_initialize(peer, init["id"])
    initialized = await peer.read_client_message()
    assert initialized["method"] == "initialized"
    start = await peer.read_client_message()
    assert start["method"] == "thread/start"
    assert start["params"].get("ephemeral") is expect_ephemeral, (
        f"thread/start ephemeral={start['params'].get('ephemeral')!r}, "
        f"expected {expect_ephemeral!r}")
    if want_roots is not None:
        assert start["params"].get("runtimeWorkspaceRoots") == want_roots
    peer.respond(
        start["id"],
        {"thread": {"id": thread_id, "sessionId": thread_id, "ephemeral": expect_ephemeral},
         "model": "gw/model-a", "sandbox": {"type": "dangerFullAccess"},
         "approvalPolicy": "never", "runtimeWorkspaceRoots": want_roots or []},
    )
    peer.notify("thread/started", {"thread": {"id": thread_id}})
    turn = await peer.read_client_message()
    assert turn["method"] == "turn/start"
    peer.respond(turn["id"], {"turn": {"id": TURN_ID, "status": "inProgress"}})
    return start


async def _drive_turn_to_completion(peer: FakePeer, handle, *, turn_id=TURN_ID,
                                    final_text=FINAL_JSON_TEXT, item_id=None):
    """Pump one turn's notifications to terminal and return (events, outcome)."""
    consume = asyncio.ensure_future(_collect(handle))
    _notify_turn_started(peer, turn_id=turn_id)
    kwargs = {"text": final_text, "turn_id": turn_id}
    if item_id is not None:
        kwargs["item_id"] = item_id
    _notify_final_agent_message(peer, **kwargs)
    _notify_token_usage(peer, turn_id=turn_id)
    _notify_turn_completed(peer, status="completed", turn_id=turn_id)
    events = await asyncio.wait_for(consume, timeout=2.0)
    outcome = await handle.outcome()
    return events, outcome


# --------------------------------------------------------------------------- #
# 1. persist mode thread/start: ephemeral:false + runtimeWorkspaceRoots.        #
# --------------------------------------------------------------------------- #

async def test_persist_mode_thread_start_sends_ephemeral_false(tmp_path):
    """``persist_thread=True`` → ``thread/start`` carries ``ephemeral:false`` so a
    rollout persists (default ephemeral:true stays leak-safe). The minted resume
    handle's token then carries the NO-GATING posture (the cross-process re-supply)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport,
                               mode="structured", persist_thread=True)
    try:
        driver = asyncio.ensure_future(
            _serve_persist_handshake(peer, expect_ephemeral=False))
        spec = AgentRunSpec(prompt="Do it.", tools=(), cwd=tmp_path,
                            output_schema=None)
        handle = await be.start(spec, _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)

        _events, outcome = await _drive_turn_to_completion(peer, handle)
        assert outcome.status == "completed"
        # Persisted run → resume token carries the posture for cross-process re-supply.
        assert outcome.resume is not None
        assert outcome.resume.backend == "codex_app_server"
        assert outcome.resume.token == {
            "thread_id": THREAD_ID, "sandbox": "danger-full-access",
            "approvalPolicy": "never"}
    finally:
        # aclose archives the persisted thread; drain that frame so the peer is clean.
        archived = asyncio.ensure_future(_read_archive(peer))
        await be.aclose()
        await asyncio.wait_for(archived, timeout=2.0)
        await peer.aclose()


async def test_default_mode_thread_start_stays_ephemeral_true(tmp_path):
    """The DEFAULT (no persist_thread) keeps ``ephemeral:true`` and a posture-FREE
    resume token (resident-reuse only; cross-process resume would fail-fast)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        driver = asyncio.ensure_future(
            _serve_persist_handshake(peer, expect_ephemeral=True))
        spec = AgentRunSpec(prompt="Do it.", tools=(), cwd=tmp_path, output_schema=None)
        handle = await be.start(spec, _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)
        _events, outcome = await _drive_turn_to_completion(peer, handle)
        assert outcome.status == "completed"
        # Ephemeral → token is thread_id ONLY (no posture).
        assert outcome.resume is not None
        assert outcome.resume.token == {"thread_id": THREAD_ID}
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 2. context_dirs → runtimeWorkspaceRoots (absolute) on thread/start.           #
# --------------------------------------------------------------------------- #

async def test_context_dirs_become_runtime_workspace_roots_on_thread_start(tmp_path):
    """``spec.context_dirs`` → ``runtimeWorkspaceRoots`` = the ABSOLUTE paths, on
    thread/start (parity with Claude add_dirs / CodexExec --add-dir). A relative path
    is resolved to absolute per schema (AbsolutePathBuf)."""
    d1 = tmp_path / "ctx_one"
    d1.mkdir()
    d2 = tmp_path / "ctx_two"
    d2.mkdir()
    want = [str(d1.resolve()), str(d2.resolve())]

    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        driver = asyncio.ensure_future(
            _serve_persist_handshake(peer, expect_ephemeral=True, want_roots=want))
        spec = AgentRunSpec(
            prompt="Do it.", tools=(), cwd=tmp_path, output_schema=None,
            context_dirs=(ContextDirView(id="one", path=d1),
                          ContextDirView(id="two", path=d2)),
        )
        handle = await be.start(spec, _NullExecutor())
        start_msg = await asyncio.wait_for(driver, timeout=2.0)
        # The roots are absolute (schema requires AbsolutePathBuf).
        for r in start_msg["params"]["runtimeWorkspaceRoots"]:
            assert Path(r).is_absolute(), f"runtimeWorkspaceRoots entry not absolute: {r!r}"
        _events, outcome = await _drive_turn_to_completion(peer, handle)
        assert outcome.status == "completed"
    finally:
        await be.aclose()
        await peer.aclose()


async def test_no_context_dirs_omits_runtime_workspace_roots(tmp_path):
    """With no context_dirs, ``runtimeWorkspaceRoots`` is OMITTED from thread/start
    (not sent as an empty list — the field is null/absent in the schema)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        driver = asyncio.ensure_future(
            _serve_persist_handshake(peer, expect_ephemeral=True))
        spec = AgentRunSpec(prompt="Do it.", tools=(), cwd=tmp_path, output_schema=None)
        handle = await be.start(spec, _NullExecutor())
        start_msg = await asyncio.wait_for(driver, timeout=2.0)
        assert "runtimeWorkspaceRoots" not in start_msg["params"], (
            "runtimeWorkspaceRoots must be omitted when there are no context_dirs")
        await _drive_turn_to_completion(peer, handle)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 3. Cross-instance resume: SECOND instance, thread/resume re-supplies policy.   #
# --------------------------------------------------------------------------- #

async def _read_archive(peer: FakePeer, *, timeout=2.0) -> dict:
    """Read the next client message, asserting it is a ``thread/archive`` request, and
    answer it (so aclose's request future resolves rather than timing out)."""
    msg = await peer.read_client_message(timeout=timeout)
    assert msg["method"] == "thread/archive", (
        f"expected thread/archive on aclose, got {msg['method']!r}")
    peer.respond(msg["id"], {})
    return msg


async def _serve_resume_handshake(peer: FakePeer, *, thread_id=THREAD_ID,
                                  resume_result=None, want_roots=None):
    """Answer a CROSS-INSTANCE resume handshake: initialize → initialized →
    thread/resume → turn/start. Asserts the thread/resume RE-SUPPLIED the NO-GATING
    posture (sandbox + approvalPolicy) and (optionally) runtimeWorkspaceRoots; answers
    thread/resume with ``resume_result`` (default = NO-GATING). Returns the parsed
    thread/resume client message."""
    init = await peer.read_client_message()
    assert init["method"] == "initialize"
    _respond_initialize(peer, init["id"])
    initialized = await peer.read_client_message()
    assert initialized["method"] == "initialized"
    resume = await peer.read_client_message()
    assert resume["method"] == "thread/resume", (
        f"cross-instance resume must issue thread/resume, got {resume['method']!r}")
    # The CORE assertion: policy is RE-SUPPLIED on resume (the probe2 downgrade fix).
    assert resume["params"]["threadId"] == thread_id
    assert resume["params"]["sandbox"] == "danger-full-access"
    assert resume["params"]["approvalPolicy"] == "never"
    if want_roots is not None:
        assert resume["params"].get("runtimeWorkspaceRoots") == want_roots
    result = dict(resume_result if resume_result is not None else _NO_GATING_RESULT)
    result["thread"] = {"id": thread_id, "sessionId": thread_id, "ephemeral": False}
    peer.respond(resume["id"], result)
    turn = await peer.read_client_message()
    assert turn["method"] == "turn/start"
    peer.respond(turn["id"], {"turn": {"id": TURN_ID_2, "status": "inProgress"}})
    return resume


async def test_cross_instance_resume_resupplies_policy_and_preserves_no_gating(tmp_path):
    """A SECOND backend instance fed a persisted resume handle (thread NOT resident in
    it) issues a native ``thread/resume`` RE-SUPPLYING sandbox+approvalPolicy, and the
    scripted resume result reports NO GATING → the run proceeds (turn/start, completes).
    This is the probe_resume happy path at the unit layer."""
    peer = FakePeer()
    factory = _CountingFactory(peer)
    # A fresh instance — nothing resident, so the resume MUST be a real cross-process
    # thread/resume (the token carries the posture → persisted).
    be2 = CodexAppServerBackend(transport_factory=factory, mode="structured",
                                persist_thread=True)
    try:
        driver = asyncio.ensure_future(
            _serve_resume_handshake(peer, resume_result=_NO_GATING_RESULT))
        spec = _resume_spec(tmp_path, token=_persisted_token())
        handle = await be2.start(spec, _NullExecutor())
        resume_msg = await asyncio.wait_for(driver, timeout=2.0)
        # Re-supplied posture proven inside _serve_resume_handshake; sanity here too.
        assert resume_msg["params"]["sandbox"] == "danger-full-access"
        assert resume_msg["params"]["approvalPolicy"] == "never"
        assert factory.built == 1, "cross-instance resume opens exactly one fresh transport"

        _events, outcome = await _drive_turn_to_completion(
            peer, handle, turn_id=TURN_ID_2, final_text=FINAL_JSON_TEXT_2,
            item_id="msg_resumed")
        assert outcome.status == "completed"
        assert outcome.structured_output == json.loads(FINAL_JSON_TEXT_2)
        # The resumed thread is now resident + persisted in be2.
        assert outcome.resume is not None
        assert outcome.resume.token == _persisted_token()
    finally:
        archived = asyncio.ensure_future(_read_archive(peer))
        await be2.aclose()
        await asyncio.wait_for(archived, timeout=2.0)
        await peer.aclose()


async def test_cross_instance_resume_resupplies_runtime_workspace_roots(tmp_path):
    """The cross-process ``thread/resume`` also re-supplies ``runtimeWorkspaceRoots``
    (absolute) from spec.context_dirs, so an @ref'd dir survives the restart as an
    explicit workspace root."""
    ctx = tmp_path / "refdir"
    ctx.mkdir()
    want = [str(ctx.resolve())]

    peer = FakePeer()
    be2 = CodexAppServerBackend(transport_factory=peer.make_transport,
                               mode="structured", persist_thread=True)
    try:
        driver = asyncio.ensure_future(
            _serve_resume_handshake(peer, want_roots=want))
        spec = _resume_spec(tmp_path, token=_persisted_token(),
                            context_dirs=(ContextDirView(id="ref", path=ctx),))
        handle = await be2.start(spec, _NullExecutor())
        resume_msg = await asyncio.wait_for(driver, timeout=2.0)
        assert resume_msg["params"]["runtimeWorkspaceRoots"] == want
        await _drive_turn_to_completion(peer, handle, turn_id=TURN_ID_2,
                                        final_text=FINAL_JSON_TEXT_2,
                                        item_id="msg_resumed")
    finally:
        archived = asyncio.ensure_future(_read_archive(peer))
        await be2.aclose()
        await asyncio.wait_for(archived, timeout=2.0)
        await peer.aclose()


async def test_cross_instance_resume_downgraded_posture_fails_fast(tmp_path):
    """If ``thread/resume`` reports a DOWNGRADED (gated) posture despite the re-supplied
    policy (the probe2 silent-downgrade trap), the backend FAILS FAST rather than run
    under a silently-gated posture — and does NOT issue a turn/start."""
    peer = FakePeer()
    factory = _CountingFactory(peer)
    be2 = CodexAppServerBackend(transport_factory=factory, mode="structured",
                                persist_thread=True)

    async def _serve_downgraded():
        init = await peer.read_client_message()
        _respond_initialize(peer, init["id"])
        initialized = await peer.read_client_message()
        assert initialized["method"] == "initialized"
        resume = await peer.read_client_message()
        assert resume["method"] == "thread/resume"
        # Policy WAS re-supplied (the backend always does), but the server reports a
        # downgraded posture anyway — the trap. Backend must reject.
        assert resume["params"]["approvalPolicy"] == "never"
        result = dict(_DOWNGRADED_RESULT)
        result["thread"] = {"id": THREAD_ID, "ephemeral": False}
        peer.respond(resume["id"], result)
        # No turn/start should follow — PROVE absence. The backend fails fast and
        # closes the freshly-opened transport, so this is EITHER a TimeoutError (no
        # frame yet) OR an EOFError (transport already closed the c2s pipe). Both prove
        # no turn/start was issued; a parsed thread/start/turn/start frame would not.
        with pytest.raises((asyncio.TimeoutError, EOFError)):
            await peer.read_client_message(timeout=0.4)

    try:
        driver = asyncio.ensure_future(_serve_downgraded())
        spec = _resume_spec(tmp_path, token=_persisted_token())
        with pytest.raises(RuntimeError, match="downgraded the NO-GATING posture"):
            await be2.start(spec, _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)
        # The failed resume's transport was closed (not leaked) and nothing is resident.
        assert factory.built == 1
        assert not be2._live, "a downgraded resume must leave nothing resident"
        assert not be2._persisted
    finally:
        await be2.aclose()
        await peer.aclose()


async def test_cross_instance_resume_error_closes_transport(tmp_path):
    """A ``thread/resume`` that ERRORS (e.g. -32600 no rollout) propagates and the
    freshly-opened transport is closed (never leaked); nothing resident."""
    peer = FakePeer()
    factory = _CountingFactory(peer)
    be2 = CodexAppServerBackend(transport_factory=factory, mode="structured",
                                persist_thread=True)

    async def _serve_resume_error():
        init = await peer.read_client_message()
        _respond_initialize(peer, init["id"])
        initialized = await peer.read_client_message()
        resume = await peer.read_client_message()
        assert resume["method"] == "thread/resume"
        peer.respond_error(resume["id"], -32600, "no rollout found")

    try:
        driver = asyncio.ensure_future(_serve_resume_error())
        spec = _resume_spec(tmp_path, token=_persisted_token())
        # The thread/resume JSON-RPC error must propagate out of start().
        with pytest.raises(JsonRpcError, match="rollout"):
            await be2.start(spec, _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)
        assert not be2._live
        assert not be2._persisted
        # The transport opened for the resume attempt was closed on the failure.
        assert peer._transport is not None and peer._transport._closed
    finally:
        await be2.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 4. Non-persist resume-miss → fail-fast (unchanged M2 behavior).               #
# --------------------------------------------------------------------------- #

async def test_resume_miss_no_posture_fails_fast(tmp_path):
    """A resume token WITHOUT posture (ephemeral thread) that is not resident →
    fail-fast (the M2 behavior). No transport built, no thread/resume."""
    peer = FakePeer()
    factory = _CountingFactory(peer)
    be = CodexAppServerBackend(transport_factory=factory, mode="structured")
    try:
        spec = _resume_spec(tmp_path, token={"thread_id": "not-resident"})
        with pytest.raises(RuntimeError, match="requires persist_thread"):
            await be.start(spec, _NullExecutor())
        assert factory.built == 0, "ephemeral resume-miss fail-fast must not build a transport"
    finally:
        await be.aclose()
        await peer.aclose()


async def test_resume_missing_thread_id_token_is_value_error(tmp_path):
    """A resume token with no ``thread_id`` is a ValueError (malformed handle)."""
    be = CodexAppServerBackend(mode="structured")
    spec = _resume_spec(tmp_path, token={"sandbox": "danger-full-access"})
    with pytest.raises(ValueError, match="missing thread_id"):
        await be.start(spec, _NullExecutor())
    await be.aclose()


async def test_resume_foreign_backend_still_value_error(tmp_path):
    """The cross-backend guard (RT-SESS-5) is unchanged: a handle minted by a DIFFERENT
    backend id raises ValueError before any resident lookup or resume attempt."""
    be = CodexAppServerBackend(mode="structured", persist_thread=True)
    spec = AgentRunSpec(prompt="x", tools=(), cwd=tmp_path, output_schema=None,
                        resume=ResumeHandle("codex", {"session_id": "abc"}))
    with pytest.raises(ValueError, match="codex_app_server"):
        await be.start(spec, _NullExecutor())
    await be.aclose()


# --------------------------------------------------------------------------- #
# 5. Within-run RESIDENT reuse still works (NO thread/resume), in persist mode.  #
# --------------------------------------------------------------------------- #

async def test_persist_mode_resident_reuse_is_still_turn_start_only(tmp_path):
    """Even in persist mode, a resume handle pointing at a RESIDENT thread reuses the
    connection: round 2 issues ONLY a turn/start (NO thread/resume, NO new transport).
    Resident-reuse takes precedence over the cross-process resume path."""
    peer = FakePeer()
    factory = _CountingFactory(peer)
    be = CodexAppServerBackend(transport_factory=factory, mode="structured",
                               persist_thread=True)
    try:
        # round 1: persist-mode start (ephemeral:false) + a complete turn.
        driver = asyncio.ensure_future(
            _serve_persist_handshake(peer, expect_ephemeral=False))
        spec1 = AgentRunSpec(prompt="r1", tools=(), cwd=tmp_path, output_schema=None)
        handle1 = await be.start(spec1, _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)
        _events1, outcome1 = await _drive_turn_to_completion(peer, handle1)
        assert outcome1.status == "completed"
        assert outcome1.resume.token == _persisted_token()
        assert factory.built == 1

        # round 2: resume with the RESIDENT thread_id (token has posture, but it is
        # resident → reuse wins). The ONLY client frame before turn/completed must be a
        # turn/start on the same thread — NOT thread/resume, NOT a new transport.
        spec2 = _resume_spec(tmp_path, token=outcome1.resume.token, prompt="r2")
        start2 = asyncio.ensure_future(be.start(spec2, _NullExecutor()))
        msg = await peer.read_client_message(timeout=2.0)
        assert msg["method"] == "turn/start", (
            f"resident reuse must issue turn/start, got {msg['method']!r}")
        assert msg["params"]["threadId"] == THREAD_ID
        peer.respond(msg["id"], {"turn": {"id": TURN_ID_2, "status": "inProgress"}})
        handle2 = await asyncio.wait_for(start2, timeout=2.0)
        assert factory.built == 1, "resident reuse must NOT build a new transport"

        _events2, outcome2 = await _drive_turn_to_completion(
            peer, handle2, turn_id=TURN_ID_2, final_text=FINAL_JSON_TEXT_2,
            item_id="msg_round2")
        assert outcome2.status == "completed"
        # The reused thread is still persisted → its outcome token still carries posture.
        assert outcome2.resume.token == _persisted_token()
    finally:
        archived = asyncio.ensure_future(_read_archive(peer))
        await be.aclose()
        await asyncio.wait_for(archived, timeout=2.0)
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 6. aclose archives persisted threads (RT-APP-HOME-LEAK).                       #
# --------------------------------------------------------------------------- #

async def test_aclose_archives_persisted_thread(tmp_path):
    """``aclose()`` best-effort ``thread/archive``'s a PERSISTED resident thread before
    closing its transport (so the rollout/daemon thread is reaped, not leaked)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport,
                               mode="structured", persist_thread=True)
    driver = asyncio.ensure_future(
        _serve_persist_handshake(peer, expect_ephemeral=False))
    spec = AgentRunSpec(prompt="Do it.", tools=(), cwd=tmp_path, output_schema=None)
    handle = await be.start(spec, _NullExecutor())
    await asyncio.wait_for(driver, timeout=2.0)
    _events, outcome = await _drive_turn_to_completion(peer, handle)
    assert outcome.status == "completed"

    transport = peer._transport
    assert transport is not None and transport._closed is False

    # aclose must issue thread/archive {threadId} BEFORE closing the transport.
    archived = asyncio.ensure_future(_read_archive(peer))
    aclose = asyncio.ensure_future(be.aclose())
    archive_msg = await asyncio.wait_for(archived, timeout=2.0)
    assert archive_msg["params"] == {"threadId": THREAD_ID}
    await asyncio.wait_for(aclose, timeout=2.0)
    assert transport._closed is True, "aclose must still close the transport after archive"
    await peer.aclose()


async def test_aclose_does_not_archive_ephemeral_thread(tmp_path):
    """An EPHEMERAL (default) thread has no rollout → aclose closes its transport
    WITHOUT issuing thread/archive. Proven by the absence of any client frame after
    the run before the transport closes."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    driver = asyncio.ensure_future(
        _serve_persist_handshake(peer, expect_ephemeral=True))
    spec = AgentRunSpec(prompt="Do it.", tools=(), cwd=tmp_path, output_schema=None)
    handle = await be.start(spec, _NullExecutor())
    await asyncio.wait_for(driver, timeout=2.0)
    await _drive_turn_to_completion(peer, handle)

    # No thread/archive may be sent for an ephemeral thread.
    await be.aclose()
    with pytest.raises((asyncio.TimeoutError, EOFError)):
        # After aclose the pipes are EOF; either no frame (TimeoutError pre-close) or
        # EOFError (pipe closed). A thread/archive frame would parse instead → fail.
        msg = await peer.read_client_message(timeout=0.4)
        assert msg["method"] != "thread/archive", "ephemeral thread must NOT be archived"
    await peer.aclose()


async def test_aclose_archive_failure_is_suppressed(tmp_path):
    """aclose stays robust: if ``thread/archive`` ERRORS, aclose swallows it and still
    closes the transport (the archive is strictly best-effort)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport,
                               mode="structured", persist_thread=True)
    driver = asyncio.ensure_future(
        _serve_persist_handshake(peer, expect_ephemeral=False))
    spec = AgentRunSpec(prompt="Do it.", tools=(), cwd=tmp_path, output_schema=None)
    handle = await be.start(spec, _NullExecutor())
    await asyncio.wait_for(driver, timeout=2.0)
    await _drive_turn_to_completion(peer, handle)
    transport = peer._transport

    async def _error_the_archive():
        msg = await peer.read_client_message(timeout=2.0)
        assert msg["method"] == "thread/archive"
        peer.respond_error(msg["id"], -32601, "archive not supported")

    errd = asyncio.ensure_future(_error_the_archive())
    await be.aclose()  # must NOT raise despite the archive error
    await asyncio.wait_for(errd, timeout=2.0)
    assert transport is not None and transport._closed is True
    await peer.aclose()
