"""Layered capability detection for CodexAppServerBackend (RT-CAP-1).

In **tools mode** the backend's contract (return/todo/finish + custom
:class:`ToolDef`s) rides on ``dynamicTools``, which REQUIRES that ``initialize``
negotiated ``capabilities.experimentalApi:true``. If that negotiation does NOT hold,
the live ``codex app-server`` rejects ``thread/start`` (which carries
``dynamicTools``) with ``-32600 "requires experimentalApi capability"``.

Critically, the ``initialize`` RESULT does **NOT** echo negotiated capabilities —
InitializeResponse carries only
``codexHome/platformFamily/platformOs/userAgent``, with NO ``capabilities`` field. So
the ``initialize`` result is NOT an observable signal for experimentalApi acceptance;
the ``thread/start`` ``-32600`` rejection is the ONLY runtime signal. RT-CAP-1 turns
that opaque ``-32600`` into a CLEAR fail-fast — the tools-mode backend MUST surface a
capability error naming experimentalApi, and MUST NOT silently degrade to a tool-less
run.

These tests drive a *real* :class:`AppServerTransport` through the in-memory
``FakePeer`` (NO codex spawn, NO gateway). The ``-32600`` is injected verbatim via
``peer.respond_error`` on the ``thread/start`` id; happy-path / structured frames are
lifted from ``test_backend`` / ``test_backend_tools`` (themselves verbatim from the
golden transcripts).

asyncio_mode="auto" (pyproject) → async tests need no decorator.
"""

from __future__ import annotations

import asyncio

import pytest

from tilldone.backends.app_server.backend import (
    CodexAppServerBackend,
    _EXPERIMENTAL_API_ERROR_CODE,
)
from tilldone.core.enums import EventKind
from tilldone.core.spec import AgentRunSpec

from .fake_peer import FakePeer
from .test_backend import (
    THREAD_ID,
    TURN_ID,
    _respond_initialize,
    _respond_thread_start,
)
from .test_backend_tools import (
    RecordingExecutor,
    _CONTRACT_TOOLS,
    _notify_narration,
    _notify_turn_completed,
    _notify_turn_started,
    _spec,
)

# The live wording: thread/start.dynamicTools without a
# negotiated experimentalApi → JSON-RPC -32600 with this message.
_LIVE_EXPERIMENTAL_MSG = "requires experimentalApi capability"


async def _collect(handle):
    return [ev async for ev in handle.events]


# --------------------------------------------------------------------------- #
# offline gate: probe_capabilities() — schema method-set membership            #
# --------------------------------------------------------------------------- #

def test_probe_capabilities_tools_mode_reports_dynamic_tools_requirement():
    """The offline gate (akin to codex_exec.probe_host_tools): tools mode declares it
    NEEDS dynamicTools, which the known app-server request method-set DOES contain
    (thread/start + turn/start). So the offline layer passes — the experimentalApi
    negotiation can only be confirmed at runtime (the -32600 signal)."""
    probe = CodexAppServerBackend(mode="tools").probe_capabilities()
    assert probe["mode"] == "tools"
    assert probe["needs_dynamic_tools"] is True
    # The methods this mode drives are all in the known request set → offline-OK.
    assert probe["offline_ok"] is True
    # The offline gate CANNOT observe experimentalApi acceptance (initialize result
    # does not echo capabilities) — it is explicitly a runtime/canary signal.
    assert probe["experimental_api_observable_offline"] is False


def test_probe_capabilities_structured_mode_needs_no_dynamic_tools():
    """Structured mode drives NO dynamicTools, so the experimentalApi gate does not
    apply to it at all (the capability gate is tools-mode-only)."""
    probe = CodexAppServerBackend(mode="structured").probe_capabilities()
    assert probe["mode"] == "structured"
    assert probe["needs_dynamic_tools"] is False
    assert probe["offline_ok"] is True


# --------------------------------------------------------------------------- #
# THE load-bearing test: tools-mode thread/start -32600 → CLEAR fail-fast       #
# --------------------------------------------------------------------------- #

async def test_tools_mode_thread_start_minus_32600_fails_fast_with_clear_error(tmp_path):
    """experimentalApi was NOT negotiated → live app-server rejects thread/start
    (carrying dynamicTools) with -32600 "requires experimentalApi capability".

    The backend MUST translate that into a CLEAR capability error that names
    experimentalApi / capability — NOT leak the opaque -32600, NOT silently fall
    through to a tool-less run (no text-protocol fallback)."""
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)  # default tools
    try:
        start_task = asyncio.ensure_future(be.start(_spec(tmp_path), ex))

        init = await peer.read_client_message()
        assert init["method"] == "initialize"
        # The handshake itself SUCCEEDS — experimentalApi acceptance is NOT observable
        # here (initialize result echoes no capabilities), so this cannot be the gate.
        _respond_initialize(peer, init["id"])
        initialized = await peer.read_client_message()
        assert initialized["method"] == "initialized"

        start = await peer.read_client_message()
        assert start["method"] == "thread/start"
        # Sanity: dynamicTools ARE present — this is exactly the request the server
        # rejects when experimentalApi was not negotiated.
        assert "dynamicTools" in start["params"]
        # Inject the live rejection verbatim.
        peer.respond_error(
            start["id"], _EXPERIMENTAL_API_ERROR_CODE,
            "thread/start.dynamicTools requires experimentalApi capability")

        with pytest.raises(Exception) as ei:
            await start_task
    finally:
        await peer.aclose()
        await be.aclose()

    msg = str(ei.value)
    # Names the missing capability — operator can see WHY it failed.
    assert "experimentalApi" in msg
    assert "capability" in msg.lower() or "dynamic" in msg.lower()
    # The fail-fast is explicit about refusing a silent text-protocol fallback.
    assert "fallback" in msg.lower() or "fail" in msg.lower() or "unavailable" in msg.lower()
    # And it must NOT have silently invoked the host executor (no tool-less spin).
    assert ex.calls == []


async def test_tools_mode_minus_32600_is_not_an_opaque_leak(tmp_path):
    """Regression on the failure mode: the raised error must be the capability error,
    NOT the bare transport JsonRpcError("JSON-RPC error -32600: ...")."""
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    try:
        start_task = asyncio.ensure_future(be.start(_spec(tmp_path), ex))
        init = await peer.read_client_message()
        _respond_initialize(peer, init["id"])
        await peer.read_client_message()  # initialized
        start = await peer.read_client_message()
        peer.respond_error(start["id"], _EXPERIMENTAL_API_ERROR_CODE,
                           "requires experimentalApi capability")
        with pytest.raises(Exception) as ei:
            await start_task
    finally:
        await peer.aclose()
        await be.aclose()

    # A raw JsonRpcError repr would START with "JSON-RPC error -32600" — the backend
    # must have wrapped it into a human capability message instead.
    msg = str(ei.value)
    assert not msg.startswith("JSON-RPC error")
    assert "experimentalApi" in msg


async def test_tools_mode_minus_32600_closes_transport_no_leak(tmp_path):
    """The fail-fast must not leak the child/connection: a thread/start rejection
    tears the transport down (same _start_fresh handshake-failure cleanup), so the
    backend keeps NO resident thread afterward."""
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    try:
        start_task = asyncio.ensure_future(be.start(_spec(tmp_path), ex))
        init = await peer.read_client_message()
        _respond_initialize(peer, init["id"])
        await peer.read_client_message()
        start = await peer.read_client_message()
        peer.respond_error(start["id"], _EXPERIMENTAL_API_ERROR_CODE,
                           "requires experimentalApi capability")
        with pytest.raises(Exception):
            await start_task
        # No resident thread was registered (thread/start never returned a thread.id).
        assert be._live == {}
        assert be._persisted == set()
    finally:
        await peer.aclose()
        await be.aclose()


# --------------------------------------------------------------------------- #
# happy path regression: experimentalApi accepted + dynamicTools accepted       #
# --------------------------------------------------------------------------- #

async def test_tools_mode_happy_path_still_works_when_experimental_api_accepted(tmp_path):
    """When experimentalApi negotiates (thread/start ACCEPTS dynamicTools and returns
    a thread.id), the capability gate is a no-op — the normal tools-mode run proceeds
    to a completed turn. Regression: the gate must not break the happy path."""
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    try:
        start_task = asyncio.ensure_future(be.start(_spec(tmp_path), ex))

        init = await peer.read_client_message()
        _respond_initialize(peer, init["id"])
        await peer.read_client_message()  # initialized
        start = await peer.read_client_message()
        assert start["method"] == "thread/start"
        assert "dynamicTools" in start["params"]
        # ACCEPT: return a real thread.id (experimentalApi was negotiated).
        _respond_thread_start(peer, start["id"])
        turn = await peer.read_client_message()
        assert turn["method"] == "turn/start"
        peer.respond(turn["id"], {"turn": {"id": TURN_ID, "status": "inProgress"}})

        handle = await start_task
        # Drive a trivial completed turn (no tool calls needed for this regression).
        _notify_turn_started(peer)
        _notify_narration(peer, text="Done.")
        _notify_turn_completed(peer, status="completed")

        events = await _collect(handle)
        outcome = await handle.outcome()
    finally:
        await peer.aclose()
        await be.aclose()

    assert outcome.status == "completed"
    assert events[-1].kind == EventKind.RUN_COMPLETED
    # tools mode → no structured_output (the tool channel is the contract authority).
    assert outcome.structured_output is None


# --------------------------------------------------------------------------- #
# structured mode is UNAFFECTED by the tools-mode capability gate               #
# --------------------------------------------------------------------------- #

async def test_structured_mode_unaffected_by_capability_gate(tmp_path):
    """Structured mode carries NO dynamicTools, so the experimentalApi gate does not
    apply: a normal structured handshake + turn settles to completed exactly as in M2.
    The capability gate must be strictly tools-mode-scoped."""
    from .test_backend import _drive_happy, _spec as _structured_spec

    peer = FakePeer()

    class _NullExec:
        async def invoke(self, call):  # pragma: no cover - structured never invokes
            raise AssertionError("structured mode must not invoke host tools")

    be = CodexAppServerBackend(mode="structured", transport_factory=peer.make_transport)
    try:
        start_task = asyncio.ensure_future(be.start(_structured_spec(tmp_path), _NullExec()))
        driver = asyncio.ensure_future(_drive_happy(peer))
        handle = await start_task
        events = await _collect(handle)
        outcome = await handle.outcome()
        await driver
    finally:
        await peer.aclose()
        await be.aclose()

    assert outcome.status == "completed"
    assert outcome.structured_output == {
        "returns": {"answer": "DELTA-9082"}, "todos_done": [], "finished": True}


async def test_structured_mode_thread_start_carries_no_dynamic_tools(tmp_path):
    """Belt-and-suspenders: structured-mode thread/start has NO dynamicTools, so it can
    never trip the experimentalApi gate in the first place (the gate is only relevant
    where dynamicTools are sent)."""
    from .test_backend import _spec as _structured_spec

    peer = FakePeer()

    class _NullExec:
        async def invoke(self, call):  # pragma: no cover
            raise AssertionError("structured mode must not invoke host tools")

    be = CodexAppServerBackend(mode="structured", transport_factory=peer.make_transport)
    try:
        start_task = asyncio.ensure_future(be.start(_structured_spec(tmp_path), _NullExec()))
        init = await peer.read_client_message()
        _respond_initialize(peer, init["id"])
        await peer.read_client_message()  # initialized
        start = await peer.read_client_message()
        assert start["method"] == "thread/start"
        assert "dynamicTools" not in start["params"]
        _respond_thread_start(peer, start["id"])
        turn = await peer.read_client_message()
        peer.respond(turn["id"], {"turn": {"id": TURN_ID, "status": "inProgress"}})
        handle = await start_task
        _notify_turn_completed(peer, status="completed")
        await _collect(handle)
    finally:
        await peer.aclose()
        await be.aclose()
