"""model_map + RT-APP-REROUTE — explicit model resolution + reroute notice
for :class:`CodexAppServerBackend`.

``spec.model_map`` is a concrete, CALLER-SUPPLIED ``{alias → model_id}`` dict
resolved for THIS backend only (explicit mapping; does NOT define quality
equivalence across backends):

  * if ``spec.model`` is a KEY in ``spec.model_map`` → resolve it to that concrete
    model_id and send THAT on ``thread/start``;
  * otherwise ``spec.model`` passes through unchanged.

This is orthogonal to ``CodexExecBackend``'s fail-fast when ``spec.model_map`` is
set (that backend refuses cross-backend tier aliases it cannot resolve). Real-run
proof lives in (a) the ``model`` actually sent on the wire and (b) live env-gated
tests.

RT-APP-REROUTE: the server may reroute a turn mid-flight
(``{fromModel, toModel, reason, threadId, turnId}``). A reroute MUST be OBSERVABLE —
surfaced as an explicit notice event, never silently swallowed — and the FINAL model
that actually executed must be recorded.

All tests drive a *real* :class:`AppServerTransport` over the in-memory ``FakePeer``
(NO codex spawn, NO gateway). ``asyncio_mode="auto"`` (pyproject) → async tests need
no decorator.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from tilldone.backends.app_server.backend import CodexAppServerBackend
from tilldone.backends.app_server.events import map_notification
from tilldone.core.enums import EventKind
from tilldone.core.spec import AgentRunSpec

from .fake_peer import FakePeer
from .test_backend import (
    FINAL_JSON_TEXT,
    THREAD_ID,
    TURN_ID,
    _NullExecutor,
    _notify_final_agent_message,
    _notify_turn_completed,
    _notify_turn_started,
    _respond_initialize,
    _respond_thread_start,
    _serve_handshake,
    _spec,
)
from .test_backend_multiround import _collect


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

async def _read_thread_start_params(peer: FakePeer, *, thread_id=THREAD_ID) -> dict:
    """Drive initialize → initialized, then return the ``thread/start`` params dict
    (so a test can assert the wire ``model``), then finish thread/start + turn/start.

    Mirrors ``_serve_handshake`` but EXPOSES the thread/start params (which
    ``_serve_handshake`` swallows) — the model_map resolution is asserted on the wire.
    """
    init = await peer.read_client_message()
    assert init["method"] == "initialize"
    _respond_initialize(peer, init["id"])
    initialized = await peer.read_client_message()
    assert initialized["method"] == "initialized"
    start = await peer.read_client_message()
    assert start["method"] == "thread/start"
    start_params = start["params"]
    _respond_thread_start(peer, start["id"], thread_id=thread_id)
    turn = await peer.read_client_message()
    assert turn["method"] == "turn/start"
    peer.respond(turn["id"], {"turn": {"id": TURN_ID, "status": "inProgress"}})
    return start_params


def _notify_model_rerouted(peer: FakePeer, *, from_model, to_model,
                           reason="highRiskCyberActivity", turn_id=TURN_ID,
                           thread_id=THREAD_ID):
    """Emit a ``model/rerouted`` notification in the EXACT live wire shape
    (ModelReroutedNotification: top-level fromModel/toModel/reason/threadId/turnId)."""
    peer.notify(
        "model/rerouted",
        {"fromModel": from_model, "toModel": to_model, "reason": reason,
         "threadId": thread_id, "turnId": turn_id},
    )


# --------------------------------------------------------------------------- #
# events.py — pure mapper unit test for model/rerouted                          #
# --------------------------------------------------------------------------- #

def test_map_model_rerouted_to_backend_notice():
    """A ``model/rerouted`` notification maps to an OBSERVABLE BACKEND_NOTICE event
    carrying from/to/reason + correlation ids — never dropped (silently swallowed)."""
    ev = map_notification(
        {"method": "model/rerouted",
         "params": {"fromModel": "gpt-5.5", "toModel": "gpt-5.5-safety",
                    "reason": "highRiskCyberActivity",
                    "threadId": THREAD_ID, "turnId": TURN_ID}},
        seq=7,
    )
    assert ev is not None, "model/rerouted must NOT be dropped — a reroute must be observable"
    assert ev.kind == EventKind.BACKEND_NOTICE
    assert ev.seq == 7
    assert ev.data.get("notice") == "model_rerouted"
    assert ev.data.get("from_model") == "gpt-5.5"
    assert ev.data.get("to_model") == "gpt-5.5-safety"
    assert ev.data.get("reason") == "highRiskCyberActivity"
    assert ev.thread_id == THREAD_ID
    assert ev.turn_id == TURN_ID


# --------------------------------------------------------------------------- #
# 1. model_map resolves a tier alias → the CONCRETE model is sent on the wire.  #
# --------------------------------------------------------------------------- #

async def test_model_map_resolves_alias_to_concrete_model_on_wire(tmp_path):
    """``spec.model="smart"`` + ``model_map={"smart": "gw/model-a"}`` → the
    ``thread/start`` wire ``model`` is the RESOLVED concrete model, NOT the alias.
    This is the explicit per-backend resolution (NOT a cross-backend quality claim)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(be.start(
            _spec(tmp_path, model="smart", model_map={"smart": "gw/model-a"}),
            _NullExecutor(),
        ))
        start_params = await _read_thread_start_params(peer)
        # The ALIAS must have been resolved to the concrete model on the wire.
        assert start_params["model"] == "gw/model-a"
        assert start_params["model"] != "smart"

        handle = await asyncio.wait_for(start_task, timeout=2.0)
        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        _notify_final_agent_message(peer)
        _notify_turn_completed(peer)
        await asyncio.wait_for(consume, timeout=2.0)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 2. A plain model (no alias / model_map present) passes through UNCHANGED.     #
# --------------------------------------------------------------------------- #

async def test_plain_model_passes_through_unchanged(tmp_path):
    """No ``model_map`` at all → ``spec.model`` is sent verbatim (today's behavior)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(be.start(
            _spec(tmp_path, model="gw/model-a"), _NullExecutor()))
        start_params = await _read_thread_start_params(peer)
        assert start_params["model"] == "gw/model-a"

        handle = await asyncio.wait_for(start_task, timeout=2.0)
        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        _notify_final_agent_message(peer)
        _notify_turn_completed(peer)
        await asyncio.wait_for(consume, timeout=2.0)
    finally:
        await be.aclose()
        await peer.aclose()


async def test_model_not_in_map_passes_through_unchanged(tmp_path):
    """``model_map`` present but ``spec.model`` is NOT one of its keys → the model
    is a concrete id already and passes through verbatim (``.get(model, model)``)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(be.start(
            _spec(tmp_path, model="gw/model-a",
                  model_map={"smart": "some/other/model"}),
            _NullExecutor(),
        ))
        start_params = await _read_thread_start_params(peer)
        # The concrete model is not a key in the map, so it remains unchanged.
        assert start_params["model"] == "gw/model-a"

        handle = await asyncio.wait_for(start_task, timeout=2.0)
        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        _notify_final_agent_message(peer)
        _notify_turn_completed(peer)
        await asyncio.wait_for(consume, timeout=2.0)
    finally:
        await be.aclose()
        await peer.aclose()


async def test_model_map_without_model_sends_no_model(tmp_path):
    """``model_map`` present but ``spec.model`` is None → no resolution key, so NO
    ``model`` field on thread/start (the backend never invents one)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        start_task = asyncio.ensure_future(be.start(
            _spec(tmp_path, model_map={"smart": "gw/model-a"}), _NullExecutor()))
        start_params = await _read_thread_start_params(peer)
        assert "model" not in start_params
        handle = await asyncio.wait_for(start_task, timeout=2.0)
        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        _notify_final_agent_message(peer)
        _notify_turn_completed(peer)
        await asyncio.wait_for(consume, timeout=2.0)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 3. model/rerouted → an OBSERVABLE notice in the event stream + final model     #
#    recorded on the outcome (real-run proof of what actually executed).         #
# --------------------------------------------------------------------------- #

async def test_model_rerouted_surfaces_notice_and_records_final_model(tmp_path):
    """A turn that the server REROUTES to a different model emits a ``model/rerouted``
    notification. It must (a) surface as an observable BACKEND_NOTICE in the event
    stream (never silently swallowed) and (b) record the FINAL (rerouted-to) model on
    the outcome so the run reports what actually executed (RT-APP-REROUTE)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        driver = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(
            _spec(tmp_path, model="gw/model-a"), _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        _notify_model_rerouted(peer, from_model="gw/model-a",
                               to_model="gw/model-a-safety")
        _notify_final_agent_message(peer)
        _notify_turn_completed(peer)
        events = await asyncio.wait_for(consume, timeout=2.0)
        outcome = await handle.outcome()

        # (a) OBSERVABLE: a BACKEND_NOTICE for the reroute is in the stream.
        notices = [e for e in events
                   if e.kind == EventKind.BACKEND_NOTICE
                   and e.data.get("notice") == "model_rerouted"]
        assert len(notices) == 1, "the reroute must surface as exactly one observable notice"
        assert notices[0].data["from_model"] == "gw/model-a"
        assert notices[0].data["to_model"] == "gw/model-a-safety"

        # (b) RECORDED: the run reports the FINAL model that actually executed.
        assert outcome.status == "completed"
        assert outcome.usage is not None
        assert outcome.usage.get("final_model") == "gw/model-a-safety"
        assert outcome.usage.get("rerouted") is True
    finally:
        await be.aclose()
        await peer.aclose()


async def test_no_reroute_records_no_final_model_change(tmp_path):
    """Without a reroute, the outcome must NOT claim a reroute happened (no phantom
    notice, ``rerouted`` is falsy / absent)."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="structured")
    try:
        driver = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path, model="gw/model-a"), _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        _notify_final_agent_message(peer)
        _notify_turn_completed(peer)
        events = await asyncio.wait_for(consume, timeout=2.0)
        outcome = await handle.outcome()

        assert not [e for e in events
                    if e.kind == EventKind.BACKEND_NOTICE
                    and e.data.get("notice") == "model_rerouted"]
        assert outcome.usage is not None
        assert not outcome.usage.get("rerouted")
        assert "final_model" not in outcome.usage or outcome.usage.get("final_model") is None
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# 4. (LIVE, env-gated) a REAL turn ran the configured/resolved model — mirrors   #
#    RT-CFG-1's positive-proof style (SKIP != PASS).                             #
# --------------------------------------------------------------------------- #

_GATE = os.environ.get("TILLDONE_CODEX_E2E") == "1"


@pytest.mark.integration
@pytest.mark.skipif(not _GATE, reason="set TILLDONE_CODEX_E2E=1 to run live Codex tests")
async def test_app_server_e2e_executed_model_is_resolved(tmp_path):
    """RT-APP / model_map real-run proof (positive): a REAL ``codex app-server`` turn,
    driven with an explicit model_map alias, executes the RESOLVED concrete model — the
    real-run proof model_map must carry. Mirrors RT-CFG-1's style: the
    assertion is on what the live run actually reported, not on the config we sent.

    Live proof = the resolved concrete model is what the server ACCEPTED + RAN, proven by
    the turn reaching ``status=="completed"``: had the alias ``"smart"`` leaked to the wire
    unresolved, the real app-server would reject the unknown model and the turn would NOT
    complete. The exact wire ``model == concrete`` is pinned DETERMINISTICALLY by the unit
    ``test_model_map_resolves_alias_to_concrete_model_on_wire``. ``final_model`` is a
    REROUTE-ONLY field (``thread/start`` does NOT echo a model; the run
    only learns a "final model" when the server REROUTES) — it is recorded iff a
    ``model/rerouted`` notice arrives (see ``test_no_reroute_records_no_final_model_change``)."""
    from tilldone.testing.conformance import collect_run

    alias = "smart"
    concrete = "gw/model-a"
    spec = AgentRunSpec(
        prompt="Reply with exactly the word ACK and nothing else.",
        tools=(), cwd=tmp_path, model=alias, model_map={alias: concrete},
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"], "additionalProperties": False,
        },
    )
    codex_home = os.environ.get("CODEX_HOME")
    assert codex_home, "run live app-server model-map proof via scripts/run-live.sh"
    be = CodexAppServerBackend(mode="structured", codex_home=codex_home)
    try:
        events, outcome = await collect_run(be, spec, _NullExecutor())
    finally:
        await be.aclose()

    # Live real-run proof: reaching ``completed`` means the server ACCEPTED + RAN the
    # resolved concrete model. Had the alias "smart" leaked to the wire unresolved, the
    # real app-server would reject the unknown model and the turn would NOT complete.
    # (The exact wire ``model == concrete`` is pinned by the deterministic unit test.)
    assert outcome.status == "completed", f"live app-server turn did not complete: {outcome.error}"
    final_model = (outcome.usage or {}).get("final_model")
    rerouted_notices = [e for e in events
                        if e.kind == EventKind.BACKEND_NOTICE
                        and e.data.get("notice") == "model_rerouted"]
    if rerouted_notices:
        # The server rerouted → ``final_model`` records the (observable) rerouted-to model.
        assert final_model == rerouted_notices[-1].data["to_model"]
    else:
        # No reroute (the common case for a plain request): a real turn ran (usage present),
        # no phantom reroute is claimed, and ``final_model`` — a reroute-only field — is
        # absent. The alias-resolution proof is the ``completed`` status above plus the
        # deterministic unit test on the wire model, NOT a (nonexistent) echoed model.
        assert outcome.usage is not None, "a completed live turn must report usage"
        assert not outcome.usage.get("rerouted"), "no reroute happened — must not claim one"
        assert final_model is None, (
            f"final_model is reroute-only but was {final_model!r} without a reroute notice")
