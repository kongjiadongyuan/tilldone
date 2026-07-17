"""G8 (encoding) + S2 (empty contract) + W3 (todo↔nonce mechanism) closing tests
for the app-server backend (M7 Lane D).

Three deferred matrix rows, closed at the level each is actually testable at:

* **G8 / RT-ENC-1**: returns / tool
  results carrying CJK / emoji / 4-byte code points must round-trip intact. The
  app-server transport frames ``json.dumps(..., ensure_ascii=False)`` → ``.encode("utf-8")``
  and the reader decodes UTF-8, so non-ASCII is preserved on the wire; here we prove it
  END-TO-END through the tools-mode ``item/tool/call`` → ``ToolExecutor`` → ``ContractState``
  path (a return value with emoji/CJK/4-byte text is stored byte-identical) AND that the
  ``DynamicToolCallResponse`` the backend sends back carries the non-ASCII tool content.

* **S2**: the empty contract (no
  output / return / todo / finish) behavior was DEFINED by the loop (one round, returns
  None) but never asserted. We pin it on the app-server tools backend: exactly ONE turn
  is driven and the result is None.

* **W3**: a pure-todo task has no
  ground truth unless a todo is bound to a runtime nonce that lands on disk. The TRUE
  binding is a LIVE fixture (``test_backend_tools_live.py`` runs a real nonce side-effect);
  here we close the deterministic MECHANISM half — a custom tool that writes a
  handler-generated nonce to an absolute path is dispatched host-side via ``item/tool/call``
  and its side effect (the nonce file) is observable, so a todo gated on that tool would
  have real ground truth. (The model-attestation half stays live.)

All deterministic over the in-memory :class:`FakePeer` (NO codex spawn, NO gateway).
``asyncio_mode="auto"`` (pyproject) → async tests need no decorator.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tilldone.backends.app_server.backend import CodexAppServerBackend
from tilldone.core.contract import (
    CompletionContract,
    ContractEvaluator,
    ContractState,
    ReturnExpectation,
)
from tilldone.core.enums import EventKind
from tilldone.core.flow import Flow
from tilldone.core.registry import ToolRegistry
from tilldone.core.tools import ToolCall, ToolDef, ToolExecutor, ToolResult

from .fake_peer import FakePeer
from .test_backend import (
    THREAD_ID,
    TURN_ID,
    _respond_initialize,
    _respond_thread_start,
)
from .test_backend_tools import (
    _notify_narration,
    _notify_tool_call_completed,
    _notify_tool_call_started,
    _notify_turn_completed,
    _notify_turn_started,
)

# A nasty-encoding payload: CJK + emoji (4-byte astral) + a combining sequence.
_UNICODE_ANSWER = "你好 🌏 café 𝕏 — 完了✅"


async def _collect(handle):
    return [ev async for ev in handle.events]


# =========================================================================== #
# G8 — encoding round-trip through the tools-mode item/tool/call path          #
# =========================================================================== #

async def test_g8_unicode_return_value_round_trips_through_item_tool_call(tmp_path):
    """A ``return`` whose value has CJK + emoji + 4-byte code points is dispatched via
    ``item/tool/call`` and stored byte-identical in ContractState (no mojibake, no
    ensure_ascii escaping leaking through, no UTF-8 truncation)."""
    state = ContractState()
    contract = CompletionContract(
        outputs=(), returns=(ReturnExpectation("answer", "str", True),),
        todos=(), require_finish=False)
    registry = ToolRegistry(state, contract)
    registry.add_contract_tools()

    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="tools")
    contract_defs = ContractEvaluator(contract).tool_defs(host_tool_calls=True)
    spec = _tools_spec(tmp_path, tools=registry.tool_defs(contract_defs))
    try:
        start = asyncio.ensure_future(be.start(spec, registry))
        init = await peer.read_client_message()
        _respond_initialize(peer, init["id"])
        await peer.read_client_message()  # initialized
        ts = await peer.read_client_message()
        assert ts["method"] == "thread/start"
        _respond_thread_start(peer, ts["id"])
        turn = await peer.read_client_message()
        peer.respond(turn["id"], {"turn": {"id": TURN_ID, "status": "inProgress"}})
        handle = await start

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        # The model calls return with the unicode value.
        _notify_tool_call_started(peer, call_id="c1", tool="return",
                                  arguments={"answer": _UNICODE_ANSWER})
        peer.server_request(
            "item/tool/call",
            {"threadId": THREAD_ID, "turnId": TURN_ID, "callId": "c1",
             "namespace": "contract", "tool": "return",
             "arguments": {"answer": _UNICODE_ANSWER}}, id=1)
        reply = await peer.read_client_message()  # DynamicToolCallResponse
        _notify_tool_call_completed(peer, call_id="c1", tool="return",
                                    arguments={"answer": _UNICODE_ANSWER})
        _notify_narration(peer, text="完了 ✅")
        _notify_turn_completed(peer, status="completed")
        await asyncio.wait_for(consume, timeout=2.0)
        await handle.outcome()
    finally:
        await be.aclose()
        await peer.aclose()

    # The non-ASCII value survived the wire + dispatch byte-for-byte.
    assert state.returns["answer"] == _UNICODE_ANSWER
    # The DynamicToolCallResponse is well-formed and its (success) reply round-tripped.
    assert reply["result"]["success"] is True


async def test_g8_unicode_tool_content_preserved_in_dynamic_tool_response(tmp_path):
    """A custom tool returning non-ASCII CONTENT has that content carried verbatim in the
    ``DynamicToolCallResponse.contentItems[0].text`` the backend sends back (the model
    must see the real bytes, not ``\\uXXXX`` escapes or replacement chars)."""

    class _UnicodeExecutor:
        async def invoke(self, call: ToolCall) -> ToolResult:
            return ToolResult(content=f"result={_UNICODE_ANSWER}")

    custom = ToolDef("lookup", "custom", {"type": "object", "properties": {}})
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="tools")
    spec = _tools_spec(tmp_path, tools=(custom,))
    try:
        start = asyncio.ensure_future(be.start(spec, _UnicodeExecutor()))
        init = await peer.read_client_message()
        _respond_initialize(peer, init["id"])
        await peer.read_client_message()
        ts = await peer.read_client_message()
        _respond_thread_start(peer, ts["id"])
        turn = await peer.read_client_message()
        peer.respond(turn["id"], {"turn": {"id": TURN_ID, "status": "inProgress"}})
        handle = await start

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        _notify_tool_call_started(peer, call_id="lk", tool="lookup", arguments={},
                                  namespace=None)
        peer.server_request(
            "item/tool/call",
            {"threadId": THREAD_ID, "turnId": TURN_ID, "callId": "lk",
             "tool": "lookup", "arguments": {}}, id=1)
        reply = await peer.read_client_message()
        _notify_tool_call_completed(peer, call_id="lk", tool="lookup", arguments={},
                                    namespace=None)
        _notify_narration(peer)
        _notify_turn_completed(peer, status="completed")
        await asyncio.wait_for(consume, timeout=2.0)
        await handle.outcome()
    finally:
        await be.aclose()
        await peer.aclose()

    # The non-ASCII content reached the wire intact (FakePeer parsed real UTF-8, not
    # escapes), exactly as the contentItems inputText payload.
    text = reply["result"]["contentItems"][0]["text"]
    assert text == f"result={_UNICODE_ANSWER}"
    assert "🌏" in text and "你好" in text  # belt-and-suspenders: real glyphs, not \uXXXX


def test_g8_transport_frames_non_ascii_without_escaping():
    """Unit-floor: the transport's own framing is ensure_ascii=False + utf-8 (the property
    G8 stands on), so a notification body with emoji serializes to real UTF-8 bytes, not
    ``\\uXXXX``. (Guards against a regression that flips ensure_ascii back on.)"""
    from tilldone.backends.app_server import transport as _t
    import inspect

    src = inspect.getsource(_t.AppServerTransport._send_raw)
    assert "ensure_ascii=False" in src
    assert 'encode("utf-8")' in src


# =========================================================================== #
# S2 — empty contract: exactly one round, result is None                       #
# =========================================================================== #

async def test_s2_empty_contract_one_round_returns_none(tmp_path):
    """The empty contract (no output / return / todo / finish) is satisfied immediately:
    exactly ONE turn is driven (the loop body runs once) and the result is None (zero
    declared returns). This PINS the previously-undefined S2 behavior on the app-server
    tools backend."""
    from tilldone.core.loop import CorePolicy, run_task

    empty = CompletionContract(outputs=(), returns=(), todos=(), require_finish=False)

    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="tools")

    turn_starts = 0

    async def _server():
        nonlocal turn_starts
        init = await peer.read_client_message()
        _respond_initialize(peer, init["id"])
        await peer.read_client_message()  # initialized
        ts = await peer.read_client_message()
        assert ts["method"] == "thread/start"
        # NB: core ALWAYS wires the 3 contract tools (return/todo/finish) in tools mode,
        # even for an empty contract — but the empty contract needs NONE of them called.
        _respond_thread_start(peer, ts["id"])
        turn = await peer.read_client_message()
        assert turn["method"] == "turn/start"
        turn_starts += 1
        peer.respond(turn["id"], {"turn": {"id": TURN_ID, "status": "inProgress"}})
        _notify_turn_started(peer)
        _notify_narration(peer, text="nothing to do")
        _notify_turn_completed(peer, status="completed")

    server = asyncio.ensure_future(_server())
    try:
        result = await asyncio.wait_for(
            run_task(empty, be, prompt="do nothing", cwd=tmp_path,
                     policy=CorePolicy(max_rounds=4)),
            timeout=5.0)
        await asyncio.wait_for(server, timeout=2.0)
    finally:
        await be.aclose()
        await peer.aclose()

    # Empty contract → satisfied after the FIRST round (no second turn/start), result None.
    assert result is None
    assert turn_starts == 1, f"empty contract must drive exactly ONE round, got {turn_starts}"


async def test_s2_empty_contract_via_flow_returns_none(tmp_path):
    """The same S2 behavior through the Flow SDK surface (Flow.run on the empty contract
    returns None) — so the empty-contract definition holds at the public API too."""
    empty = CompletionContract(outputs=(), returns=(), todos=(), require_finish=False)
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="tools")
    flow = Flow(be)

    async def _server():
        init = await peer.read_client_message()
        _respond_initialize(peer, init["id"])
        await peer.read_client_message()
        ts = await peer.read_client_message()
        _respond_thread_start(peer, ts["id"])
        turn = await peer.read_client_message()
        peer.respond(turn["id"], {"turn": {"id": TURN_ID, "status": "inProgress"}})
        _notify_turn_started(peer)
        _notify_narration(peer, text="ok")
        _notify_turn_completed(peer, status="completed")

    server = asyncio.ensure_future(_server())
    try:
        result = await asyncio.wait_for(
            flow.run(empty, prompt="noop", cwd=tmp_path), timeout=5.0)
        await asyncio.wait_for(server, timeout=2.0)
    finally:
        await be.aclose()
        await peer.aclose()

    assert result is None


# =========================================================================== #
# W3 — todo↔nonce ground-truth MECHANISM (deterministic half; full binding live)#
# =========================================================================== #

async def test_w3_custom_tool_nonce_side_effect_is_host_side(tmp_path):
    """W3 mechanism: a custom tool that writes a HANDLER-GENERATED nonce to an absolute
    path is dispatched host-side through ``item/tool/call`` and its side effect (the nonce
    file with the exact bytes) lands on disk — so a todo GATED on that tool would have real
    runtime ground truth (not pure model self-attestation). The model-attestation half
    (the model actually CALLING the tool) is proven separately by the live fixture."""
    import secrets

    nonce = secrets.token_hex(8)  # generated at RUN time, never pre-written
    marker = tmp_path / "w3_nonce.txt"

    class _NonceTool:
        """A host-side tool whose side effect is writing the run nonce to ``marker``."""
        async def invoke(self, call: ToolCall) -> ToolResult:
            # Side effect happens HOST-side (in this process), not agent-side.
            marker.write_text(nonce, encoding="utf-8")
            return ToolResult(content=f"wrote {nonce}")

    custom = ToolDef("emit_nonce", "writes the run nonce",
                     {"type": "object", "properties": {}})
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=peer.make_transport, mode="tools")
    spec = _tools_spec(tmp_path, tools=(custom,))
    try:
        start = asyncio.ensure_future(be.start(spec, _NonceTool()))
        init = await peer.read_client_message()
        _respond_initialize(peer, init["id"])
        await peer.read_client_message()
        ts = await peer.read_client_message()
        _respond_thread_start(peer, ts["id"])
        turn = await peer.read_client_message()
        peer.respond(turn["id"], {"turn": {"id": TURN_ID, "status": "inProgress"}})
        handle = await start

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        # Pre-condition: the nonce file does NOT exist until the tool actually runs.
        assert not marker.exists(), "nonce must not be pre-written (anti-cheat, W2/W3)"
        _notify_tool_call_started(peer, call_id="n1", tool="emit_nonce", arguments={},
                                  namespace=None)
        peer.server_request(
            "item/tool/call",
            {"threadId": THREAD_ID, "turnId": TURN_ID, "callId": "n1",
             "tool": "emit_nonce", "arguments": {}}, id=1)
        await peer.read_client_message()
        _notify_tool_call_completed(peer, call_id="n1", tool="emit_nonce", arguments={},
                                    namespace=None)
        _notify_narration(peer)
        _notify_turn_completed(peer, status="completed")
        await asyncio.wait_for(consume, timeout=2.0)
        await handle.outcome()
    finally:
        await be.aclose()
        await peer.aclose()

    # Ground truth: the nonce file exists with the EXACT run-time bytes — the tool truly
    # ran host-side (a todo bound to this would be falsifiable, not self-attested).
    assert marker.exists(), "host-side tool side effect (nonce file) must land on disk"
    assert marker.read_text(encoding="utf-8") == nonce


# -- shared tools-mode spec helper -------------------------------------------

def _tools_spec(tmp_path, *, tools):
    from tilldone.core.spec import AgentRunSpec
    return AgentRunSpec(
        prompt="drive the tools", tools=tools, cwd=tmp_path, output_schema=None)
