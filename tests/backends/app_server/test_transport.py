"""Base protocol unit tests for AppServerTransport.

All tests run against the in-memory ``FakePeer`` — deterministic, no real codex
spawn, no gateway. The repo uses ``asyncio_mode = "auto"`` (see pyproject), so
async test functions need no decorator.
"""

from __future__ import annotations

import ast
import asyncio
import sys
from pathlib import Path

import pytest

from tilldone.backends.app_server.transport import (
    AppServerTransport,
    JsonRpcError,
    TransportClosed,
)

from .fake_peer import FakePeer


# --------------------------------------------------------------------------- #
# request / response id pairing                                               #
# --------------------------------------------------------------------------- #

async def test_request_response_roundtrip():
    """A client request is sent on the wire with an id, and the matching response
    resolves the awaiting call with its result."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        fut = asyncio.ensure_future(t.request("initialize", {"hello": "world"}))

        sent = await peer.read_client_message()
        assert sent["method"] == "initialize"
        assert sent["params"] == {"hello": "world"}
        assert "id" in sent
        assert "jsonrpc" not in sent  # never send the jsonrpc field

        peer.respond(sent["id"], {"ok": True})
        result = await asyncio.wait_for(fut, timeout=1.0)
        assert result == {"ok": True}
    finally:
        await peer.aclose()


async def test_concurrent_requests_resolve_out_of_order_to_correct_results():
    """Three concurrent requests, responded to OUT OF ORDER, must each resolve to
    their own correct result — proving id correlation (not arrival order)."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        f1 = asyncio.ensure_future(t.request("m1"))
        f2 = asyncio.ensure_future(t.request("m2"))
        f3 = asyncio.ensure_future(t.request("m3"))

        s1 = await peer.read_client_message()
        s2 = await peer.read_client_message()
        s3 = await peer.read_client_message()
        assert [s1["method"], s2["method"], s3["method"]] == ["m1", "m2", "m3"]
        # The client must use distinct ids.
        assert len({s1["id"], s2["id"], s3["id"]}) == 3

        # Respond in reverse order, each tagged with the result for that method.
        peer.respond(s3["id"], {"who": "m3"})
        peer.respond(s1["id"], {"who": "m1"})
        peer.respond(s2["id"], {"who": "m2"})

        r1, r2, r3 = await asyncio.wait_for(asyncio.gather(f1, f2, f3), timeout=1.0)
        assert r1 == {"who": "m1"}
        assert r2 == {"who": "m2"}
        assert r3 == {"who": "m3"}
    finally:
        await peer.aclose()


async def test_request_error_result_raises_jsonrpcerror():
    """An ``error`` response raises JsonRpcError carrying code/message/data verbatim."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        fut = asyncio.ensure_future(t.request("boom"))
        sent = await peer.read_client_message()
        peer.respond_error(sent["id"], -32600, "bad request", data={"why": "test"})

        with pytest.raises(JsonRpcError) as ei:
            await asyncio.wait_for(fut, timeout=1.0)
        assert ei.value.code == -32600
        assert ei.value.message == "bad request"
        assert ei.value.data == {"why": "test"}
    finally:
        await peer.aclose()


# --------------------------------------------------------------------------- #
# notify is emitted on the wire                                               #
# --------------------------------------------------------------------------- #

async def test_notify_emitted_on_wire():
    """notify() puts a method-only (no id) object on the wire."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        await t.notify("initialized", {"client": "tilldone"})
        sent = await peer.read_client_message()
        assert sent == {"method": "initialized", "params": {"client": "tilldone"}}
        assert "id" not in sent
    finally:
        await peer.aclose()


async def test_notify_without_params_omits_params_key():
    peer = FakePeer()
    t = peer.make_transport()
    try:
        await t.notify("ping")
        sent = await peer.read_client_message()
        assert sent == {"method": "ping"}
    finally:
        await peer.aclose()


# --------------------------------------------------------------------------- #
# newline framing robustness                                                  #
# --------------------------------------------------------------------------- #

async def test_framing_split_mid_line():
    """A response object split across two writes (mid-line) still parses."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        fut = asyncio.ensure_future(t.request("split"))
        sent = await peer.read_client_message()
        # Split the response bytes roughly in the middle.
        obj = {"id": sent["id"], "result": {"value": "reassembled"}}
        import json as _json

        full = (_json.dumps(obj) + "\n").encode("utf-8")
        peer.send_split(obj, at=len(full) // 2)

        result = await asyncio.wait_for(fut, timeout=1.0)
        assert result == {"value": "reassembled"}
    finally:
        await peer.aclose()


async def test_framing_packed_two_objects_one_read():
    """Two objects delivered in a SINGLE write both parse and route correctly:
    one resolves a pending request, the other lands as a notification."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        fut = asyncio.ensure_future(t.request("packed"))
        sent = await peer.read_client_message()

        notes = t.notifications()
        # Pack the response AND a notification into one write.
        peer.send_packed(
            {"id": sent["id"], "result": {"v": 1}},
            {"method": "thread/event", "params": {"n": 2}},
        )

        result = await asyncio.wait_for(fut, timeout=1.0)
        assert result == {"v": 1}

        note = await asyncio.wait_for(notes.__anext__(), timeout=1.0)
        assert note == {"method": "thread/event", "params": {"n": 2}}
    finally:
        await peer.aclose()


async def test_garbage_line_is_skipped_pump_survives():
    """A non-JSON garbage line is skipped; a subsequent valid response still works."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        fut = asyncio.ensure_future(t.request("after-garbage"))
        sent = await peer.read_client_message()

        peer.send_garbage_line("this is not json <<<")
        peer.respond(sent["id"], {"survived": True})

        result = await asyncio.wait_for(fut, timeout=1.0)
        assert result == {"survived": True}
    finally:
        await peer.aclose()


# --------------------------------------------------------------------------- #
# per-request timeout                                                         #
# --------------------------------------------------------------------------- #

async def test_request_timeout_raises_and_transport_stays_usable():
    """A request with no response times out (TimeoutError); the transport remains
    usable for a subsequent request that IS answered."""
    peer = FakePeer()
    t = peer.make_transport(request_timeout_s=0.1)
    try:
        # First request: peer never responds → must time out.
        first = asyncio.ensure_future(t.request("never"))
        _ = await peer.read_client_message()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(first, timeout=1.0)

        # Transport still usable: second request, answered, succeeds.
        second = asyncio.ensure_future(t.request("ok"))
        s2 = await peer.read_client_message()
        peer.respond(s2["id"], {"alive": True})
        assert await asyncio.wait_for(second, timeout=1.0) == {"alive": True}
    finally:
        await peer.aclose()


async def test_late_response_after_timeout_is_ignored():
    """A response that arrives AFTER its request timed out must not crash the pump
    nor wrongly resolve a later request."""
    peer = FakePeer()
    t = peer.make_transport(request_timeout_s=0.1)
    try:
        first = asyncio.ensure_future(t.request("slow"))
        s1 = await peer.read_client_message()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(first, timeout=1.0)

        # Late response for the timed-out id — must be silently ignored.
        peer.respond(s1["id"], {"too": "late"})
        await asyncio.sleep(0.02)  # let the pump process it

        # A fresh request still behaves correctly.
        second = asyncio.ensure_future(t.request("fresh"))
        s2 = await peer.read_client_message()
        assert s2["id"] != s1["id"]
        peer.respond(s2["id"], {"fresh": True})
        assert await asyncio.wait_for(second, timeout=1.0) == {"fresh": True}
    finally:
        await peer.aclose()


# --------------------------------------------------------------------------- #
# server-request routing (registered handler)                                 #
# --------------------------------------------------------------------------- #

async def test_server_request_routed_to_handler_result_uses_server_id():
    """A server→client request is dispatched to its registered handler, and the
    handler's result is sent back with the SERVER's id (not the client counter)."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        received: dict = {}

        async def handler(req: dict):
            received.update(req)
            return {"success": True, "echo": req["params"]["callId"]}

        t.on_server_request("item/tool/call", handler)

        # Server uses its own id space (id=0, overlapping client ids on purpose).
        peer.server_request("item/tool/call", {"callId": "c-1", "name": "return"}, id=0)

        reply = await peer.read_client_message()
        assert reply["id"] == 0  # echoes the SERVER's id
        assert reply["result"] == {"success": True, "echo": "c-1"}
        # Handler received method + params.
        assert received["method"] == "item/tool/call"
        assert received["params"] == {"callId": "c-1", "name": "return"}
        # Not recorded as unexpected.
        assert t.unexpected_server_requests == []
    finally:
        await peer.aclose()


async def test_handler_exception_sends_jsonrpc_error_back():
    """A handler that raises causes a JSON-RPC error response (not a hang/crash)."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        async def handler(req: dict):
            raise RuntimeError("handler kaboom")

        t.on_server_request("item/tool/call", handler)
        peer.server_request("item/tool/call", {"callId": "x"}, id=7)

        reply = await peer.read_client_message()
        assert reply["id"] == 7
        assert "error" in reply
        assert "handler kaboom" in reply["error"]["message"]
        # Pump still alive: a subsequent client request works.
        fut = asyncio.ensure_future(t.request("still-alive"))
        sent = await peer.read_client_message()
        peer.respond(sent["id"], {"yes": True})
        assert await asyncio.wait_for(fut, timeout=1.0) == {"yes": True}
    finally:
        await peer.aclose()


# --------------------------------------------------------------------------- #
# unexpected server-request (no handler)                                       #
# --------------------------------------------------------------------------- #

async def test_unexpected_server_request_errors_and_is_recorded():
    """A server request with NO registered handler is auto-answered with a JSON-RPC
    error AND recorded in unexpected_server_requests; the pump stays alive."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        peer.server_request("requestUserInput", {"prompt": "approve?"}, id=3)

        reply = await peer.read_client_message()
        assert reply["id"] == 3
        assert reply["error"]["code"] == -32601
        assert "requestUserInput" in reply["error"]["message"]

        recorded = t.unexpected_server_requests
        assert len(recorded) == 1
        assert recorded[0]["method"] == "requestUserInput"
        assert recorded[0]["params"] == {"prompt": "approve?"}
        assert recorded[0]["id"] == 3

        # Returned list is a copy — mutating it must not affect internal state.
        recorded.clear()
        assert len(t.unexpected_server_requests) == 1

        # Pump still alive.
        fut = asyncio.ensure_future(t.request("ping"))
        sent = await peer.read_client_message()
        peer.respond(sent["id"], "pong")
        assert await asyncio.wait_for(fut, timeout=1.0) == "pong"
    finally:
        await peer.aclose()


# --------------------------------------------------------------------------- #
# reentrancy: server request arrives WHILE awaiting a client response          #
# --------------------------------------------------------------------------- #

async def test_reentrant_server_request_while_awaiting_response():
    """A server item/tool/call arriving while a turn/start request is in-flight must
    be dispatched (not deadlocked). The handler can even complete BEFORE the
    awaited turn/start response arrives."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        tool_called = asyncio.Event()

        async def handler(req: dict):
            tool_called.set()
            return {"success": True}

        t.on_server_request("item/tool/call", handler)

        # Client starts a long request (turn/start) — not yet answered.
        turn = asyncio.ensure_future(t.request("turn/start", {"x": 1}))
        sent = await peer.read_client_message()
        assert sent["method"] == "turn/start"

        # WHILE that awaits, the server fires a tool call.
        peer.server_request("item/tool/call", {"callId": "mid"}, id=0)

        # The handler runs and its reply hits the wire — proving the pump was not
        # blocked by the pending turn/start.
        tool_reply = await peer.read_client_message()
        assert tool_reply["id"] == 0
        assert tool_reply["result"] == {"success": True}
        assert tool_called.is_set()

        # Now the server answers the original turn/start.
        peer.respond(sent["id"], {"turn": {"id": "t-1"}})
        assert await asyncio.wait_for(turn, timeout=1.0) == {"turn": {"id": "t-1"}}
    finally:
        await peer.aclose()


# --------------------------------------------------------------------------- #
# aclose: cancels pending request + idempotent                                 #
# --------------------------------------------------------------------------- #

async def test_aclose_cancels_pending_request_with_transportclosed():
    """A request in flight when aclose() is called gets TransportClosed (never hangs)."""
    peer = FakePeer()
    t = peer.make_transport()
    pending = asyncio.ensure_future(t.request("inflight"))
    await peer.read_client_message()  # ensure it was actually sent

    await t.aclose()

    with pytest.raises(TransportClosed):
        await asyncio.wait_for(pending, timeout=1.0)
    # cleanup peer pipes (transport already closed).
    await peer.aclose()


async def test_aclose_is_idempotent():
    """Calling aclose() twice is safe and raises nothing."""
    peer = FakePeer()
    t = peer.make_transport()
    await t.aclose()
    await t.aclose()  # must not raise
    await peer.aclose()


async def test_peer_death_fails_pending_with_transportclosed():
    """If the server stream hits EOF (peer dies) with a request in flight, the
    awaiter gets TransportClosed rather than hanging — without an explicit aclose."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        pending = asyncio.ensure_future(t.request("orphan"))
        await peer.read_client_message()

        peer.close_server_stream()  # server→client EOF only

        with pytest.raises(TransportClosed):
            await asyncio.wait_for(pending, timeout=1.0)
    finally:
        await peer.aclose()


async def test_request_after_close_raises_transportclosed():
    """Issuing a request after aclose() raises TransportClosed immediately."""
    peer = FakePeer()
    t = peer.make_transport()
    await t.aclose()
    with pytest.raises(TransportClosed):
        await t.request("nope")
    await peer.aclose()


async def test_notifications_iterator_terminates_on_close():
    """The notifications() async iterator ends cleanly when the transport closes."""
    peer = FakePeer()
    t = peer.make_transport()
    notes = t.notifications()

    peer.notify("event/one", {"a": 1})
    first = await asyncio.wait_for(notes.__anext__(), timeout=1.0)
    assert first == {"method": "event/one", "params": {"a": 1}}

    await t.aclose()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(notes.__anext__(), timeout=1.0)
    await peer.aclose()


# --------------------------------------------------------------------------- #
# 0-core-diff: transport must not import tilldone.core                        #
# --------------------------------------------------------------------------- #

def test_transport_does_not_import_tilldone_core_via_ast():
    """Static guard: the transport module's source imports NOTHING from
    tilldone.core (parsed from the AST — independent of import side effects)."""
    src_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "tilldone"
        / "backends"
        / "app_server"
        / "transport.py"
    )
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tilldone.core" or alias.name.startswith("tilldone.core."):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "tilldone.core" or mod.startswith("tilldone.core."):
                offending.append(mod)
    assert offending == [], f"transport.py must not import tilldone.core; found {offending}"


def test_importing_transport_does_not_load_tilldone_core():
    """Dynamic guard: importing the transport must not pull tilldone.core into
    sys.modules (catches transitive imports the AST check would miss)."""
    # Drop any previously-imported core modules so this assertion is meaningful
    # even if another test imported core first.
    for name in list(sys.modules):
        if name == "tilldone.core" or name.startswith("tilldone.core."):
            del sys.modules[name]
    # Force a fresh import of the transport module.
    for name in list(sys.modules):
        if name.startswith("tilldone.backends.app_server"):
            del sys.modules[name]

    import importlib

    importlib.import_module("tilldone.backends.app_server.transport")

    leaked = [
        n for n in sys.modules
        if n == "tilldone.core" or n.startswith("tilldone.core.")
    ]
    assert leaked == [], f"importing transport leaked tilldone.core modules: {leaked}"
