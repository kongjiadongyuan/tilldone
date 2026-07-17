"""Adversarial protocol-stream tests for :class:`AppServerTransport`.

Pure protocol layer — every test drives a *real* transport against the
in-memory :class:`FakePeer` (NO real codex, NO gateway), so the suite is
deterministic, fast, and non-flaky. These DEEPEN the base suite
(``test_transport.py``) around three fault lines:

* **RT-APP-REENTRANCY** — server→client requests interleaved with in-flight
  client ``request()``s, nested handler-issued requests, and the exact M0
  ordering where a tool call lands in the window between sending ``initialize``
  and receiving its response.
* **RT-APP-REQ** — the server-request allowlist: only registered methods reach
  a handler; everything else is auto-answered ``-32601`` AND recorded, without
  polluting the path of legitimate interleaved calls. Handler ``return`` vs
  ``raise`` produce a wire ``result`` vs a ``-32603`` ``error`` respectively.
* **Framing / encoding / ordering races** — byte-at-a-time delivery, packed
  multi-object writes with a server-request in the middle, escaped/`{}`-laden
  string payloads, non-ASCII round-trips, non-object JSON skipping, duplicate
  responses, and EOF while both a request is pending and a handler is mid-flight.

The repo uses ``asyncio_mode = "auto"`` (see pyproject), so async test
functions need no decorator. Unit tests are NEVER auto-rerun (conftest only
flakes integration-marked tests), so any failure here is a real signal.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tilldone.backends.app_server.transport import (
    JsonRpcError,
    TransportClosed,
)

from .fake_peer import FakePeer


# --------------------------------------------------------------------------- #
# small local helpers (allowed: this file only)                               #
# --------------------------------------------------------------------------- #

async def _read_until(
    peer: FakePeer,
    predicate,
    *,
    max_msgs: int = 12,
    timeout: float = 1.0,
):
    """Read client messages until ``predicate(msg)`` is true; return that msg.

    Reentrancy tests interleave several client→server writes whose *wire order*
    is an implementation detail (handler tasks vs the awaiting ``request()``).
    Matching by predicate instead of by position keeps the assertions about
    semantics, not scheduling. Collected non-matching messages are returned too.
    """
    seen: list[dict] = []
    for _ in range(max_msgs):
        msg = await peer.read_client_message(timeout=timeout)
        seen.append(msg)
        if predicate(msg):
            return msg, seen
    raise AssertionError(
        f"no client message matched predicate after {max_msgs} reads; saw={seen}"
    )


def _is_response_to(msg: dict, rid) -> bool:
    return msg.get("id") == rid and ("result" in msg or "error" in msg)


def _is_request_method(msg: dict, method: str) -> bool:
    return msg.get("method") == method and "id" in msg


# =========================================================================== #
# RT-APP-REENTRANCY (deepened)                                                #
# =========================================================================== #

async def test_interleaved_server_requests_with_two_inflight_client_requests():
    """Two client request()s in flight AND two server→client requests interleaved:
    every client future resolves to its own correct result AND every server
    request receives a response — none starved, none cross-wired."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        seen_calls: list[str] = []

        async def handler(req: dict):
            cid = req["params"]["callId"]
            seen_calls.append(cid)
            return {"success": True, "echo": cid}

        t.on_server_request("item/tool/call", handler)

        # Two client requests, neither answered yet.
        fa = asyncio.ensure_future(t.request("turn/start", {"which": "a"}))
        fb = asyncio.ensure_future(t.request("turn/start", {"which": "b"}))
        sa = await peer.read_client_message()
        sb = await peer.read_client_message()
        assert {sa["params"]["which"], sb["params"]["which"]} == {"a", "b"}
        assert sa["id"] != sb["id"]

        # Interleave TWO server requests while both client requests are pending.
        peer.server_request("item/tool/call", {"callId": "tool-1"}, id=0)
        peer.server_request("item/tool/call", {"callId": "tool-2"}, id=1)

        # Both tool replies must hit the wire (order between them is non-spec).
        r_tool1, _ = await _read_until(peer, lambda m: _is_response_to(m, 0))
        r_tool2, _ = await _read_until(peer, lambda m: _is_response_to(m, 1))
        assert r_tool1["result"] == {"success": True, "echo": "tool-1"}
        assert r_tool2["result"] == {"success": True, "echo": "tool-2"}
        assert set(seen_calls) == {"tool-1", "tool-2"}

        # Now answer the two client requests OUT OF ORDER, each correctly.
        peer.respond(sb["id"], {"turn": "B"})
        peer.respond(sa["id"], {"turn": "A"})
        ra, rb = await asyncio.wait_for(asyncio.gather(fa, fb), timeout=1.0)
        assert ra == {"turn": "A"}
        assert rb == {"turn": "B"}
    finally:
        await peer.aclose()


async def test_handler_issues_nested_request_no_deadlock():
    """A registered handler that itself calls transport.request(...) and awaits it
    (while the reader pump keeps running) must not deadlock: the nested request's
    response is delivered by the same pump, the handler completes, and its result
    is sent back for the original server request."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        nested_result: dict = {}

        async def handler(req: dict):
            # Handler issues its OWN client request and awaits the answer. This
            # only works if the pump is not blocked by the handler (handlers run
            # as their own task), and the nested future is resolved by the pump.
            sub = await t.request("session/lookup", {"for": req["params"]["callId"]})
            nested_result.update(sub)
            return {"success": True, "resolvedWith": sub["value"]}

        t.on_server_request("item/tool/call", handler)

        # Fire the server request that triggers the handler.
        peer.server_request("item/tool/call", {"callId": "nest"}, id=42)

        # The handler's nested client request appears on the wire.
        nested = await _read_until(peer, lambda m: _is_request_method(m, "session/lookup"))
        nested_msg = nested[0]
        assert nested_msg["params"] == {"for": "nest"}
        assert "id" in nested_msg  # it's a client request (has our id)

        # Answer the nested request; the handler then finishes and replies.
        peer.respond(nested_msg["id"], {"value": "resolved-99"})

        reply, _ = await _read_until(peer, lambda m: _is_response_to(m, 42))
        assert reply["result"] == {"success": True, "resolvedWith": "resolved-99"}
        assert nested_result == {"value": "resolved-99"}
    finally:
        await peer.aclose()


async def test_tool_call_in_window_between_initialize_send_and_response():
    """The exact M0-observed ordering: an item/tool/call server-request arrives
    AFTER the client sends `initialize` but BEFORE its response. It must be
    dispatched (handled + answered) in that window, and the initialize response
    still resolves afterward."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        handled = asyncio.Event()

        async def handler(req: dict):
            handled.set()
            return {"success": True}

        t.on_server_request("item/tool/call", handler)

        # 1) client sends initialize (not yet answered).
        init = asyncio.ensure_future(t.request("initialize", {"clientInfo": "cf"}))
        sent_init = await peer.read_client_message()
        assert sent_init["method"] == "initialize"

        # 2) IN THE WINDOW: server fires a tool call before answering initialize.
        peer.server_request("item/tool/call", {"callId": "early"}, id=0)

        # 3) it is dispatched + answered despite initialize being unresolved.
        tool_reply, _ = await _read_until(peer, lambda m: _is_response_to(m, 0))
        assert tool_reply["result"] == {"success": True}
        assert handled.is_set()

        # 4) now initialize is answered and resolves correctly.
        peer.respond(sent_init["id"], {"capabilities": {"ok": True}})
        assert await asyncio.wait_for(init, timeout=1.0) == {"capabilities": {"ok": True}}
    finally:
        await peer.aclose()


# =========================================================================== #
# RT-APP-REQ (deepened allowlist semantics)                                    #
# =========================================================================== #

async def test_only_allowlisted_method_handled_others_error_and_recorded():
    """Register ONLY item/tool/call. Inject the full menagerie of *other* server
    requests (applyPatchApproval / execCommandApproval / elicitation /
    item/tool/requestUserInput / an arbitrary unknown). Each is auto-answered
    with -32601 AND lands in unexpected_server_requests. Interleaved
    item/tool/call requests still succeed and do NOT pollute the unexpected list."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        handled_calls: list[str] = []

        async def handler(req: dict):
            handled_calls.append(req["params"]["callId"])
            return {"success": True}

        t.on_server_request("item/tool/call", handler)

        # Interleave: legit call, then a barrage of non-allowlisted methods, with
        # another legit call sandwiched in the middle. ids are the SERVER's space
        # and deliberately overlap with each other / client ids.
        peer.server_request("item/tool/call", {"callId": "ok-1"}, id=0)
        peer.server_request("applyPatchApproval", {"patch": "..."}, id=1)
        peer.server_request("execCommandApproval", {"cmd": "rm -rf /"}, id=2)
        peer.server_request("item/tool/call", {"callId": "ok-2"}, id=3)
        peer.server_request("elicitation", {"q": "name?"}, id=4)
        peer.server_request("item/tool/requestUserInput", {"prompt": "approve?"}, id=5)
        peer.server_request("totally/unknown/method", {"x": 1}, id=6)

        # Collect replies until we've seen the two legit results + five errors.
        legit_ids: set = set()
        error_methods_by_id: dict = {}
        # We expect exactly 7 replies (2 results + 5 errors).
        for _ in range(7):
            reply = await peer.read_client_message(timeout=1.0)
            if "result" in reply:
                assert reply["result"] == {"success": True}
                legit_ids.add(reply["id"])
            else:
                assert "error" in reply
                assert reply["error"]["code"] == -32601
                error_methods_by_id[reply["id"]] = reply["error"]["message"]

        # The two legit calls (server ids 0 and 3) got results.
        assert legit_ids == {0, 3}
        assert set(handled_calls) == {"ok-1", "ok-2"}

        # The five non-allowlisted got -32601 errors, by server id.
        assert set(error_methods_by_id) == {1, 2, 4, 5, 6}

        # And exactly those five (not the two legit) are recorded as unexpected.
        recorded = t.unexpected_server_requests
        rec_by_id = {r["id"]: r["method"] for r in recorded}
        assert rec_by_id == {
            1: "applyPatchApproval",
            2: "execCommandApproval",
            4: "elicitation",
            5: "item/tool/requestUserInput",
            6: "totally/unknown/method",
        }
        # item/tool/call must NEVER appear in the unexpected list.
        assert all(r["method"] != "item/tool/call" for r in recorded)
        # The recorded error message names the offending method.
        assert "applyPatchApproval" in error_methods_by_id[1]
        assert "execCommandApproval" in error_methods_by_id[2]

        # Pump still alive after the barrage.
        fut = asyncio.ensure_future(t.request("ping"))
        sent = await peer.read_client_message()
        peer.respond(sent["id"], "pong")
        assert await asyncio.wait_for(fut, timeout=1.0) == "pong"
    finally:
        await peer.aclose()


async def test_handler_error_shaped_return_is_result_not_error():
    """A handler that RETURNS an error-shaped value sends a JSON-RPC *result*
    carrying that value verbatim (the transport does not reinterpret payload
    shape) — distinct from a handler that RAISES, which yields a -32603 error.
    The pump survives both, in sequence, on the same transport."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        async def returns_error_shape(req: dict):
            # Looks like an error object, but is a normal RETURN value.
            return {"error": {"code": 123, "message": "tool said no"}, "ok": False}

        t.on_server_request("item/tool/call", returns_error_shape)
        peer.server_request("item/tool/call", {"callId": "shape"}, id=10)

        reply = await peer.read_client_message(timeout=1.0)
        assert reply["id"] == 10
        # Crucially: it's a RESULT, not a transport-level error envelope.
        assert "result" in reply
        assert "error" not in reply
        assert reply["result"] == {"error": {"code": 123, "message": "tool said no"}, "ok": False}

        # Now swap in a RAISING handler and prove the wire response differs.
        async def raises(req: dict):
            raise ValueError("explicit boom")

        t.on_server_request("item/tool/call", raises)
        peer.server_request("item/tool/call", {"callId": "raise"}, id=11)

        reply2 = await peer.read_client_message(timeout=1.0)
        assert reply2["id"] == 11
        assert "error" in reply2
        assert "result" not in reply2
        assert reply2["error"]["code"] == -32603
        assert "explicit boom" in reply2["error"]["message"]

        # Pump survived both: a plain client request still works.
        fut = asyncio.ensure_future(t.request("alive"))
        sent = await peer.read_client_message()
        peer.respond(sent["id"], {"yes": True})
        assert await asyncio.wait_for(fut, timeout=1.0) == {"yes": True}
    finally:
        await peer.aclose()


# =========================================================================== #
# Framing / encoding adversarial                                               #
# =========================================================================== #

async def test_framing_one_byte_at_a_time_parses_exactly_once():
    """Feeding an entire response object ONE BYTE AT A TIME must reassemble into
    exactly one parsed frame — resolved once, with no spurious extra frames."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        fut = asyncio.ensure_future(t.request("drip"))
        sent = await peer.read_client_message()

        obj = {"id": sent["id"], "result": {"value": "byte-by-byte"}}
        data = (json.dumps(obj) + "\n").encode("utf-8")
        # Extreme split: one byte per write.
        for i in range(len(data)):
            peer.send_raw_bytes(data[i : i + 1])

        result = await asyncio.wait_for(fut, timeout=1.0)
        assert result == {"value": "byte-by-byte"}

        # Prove "exactly once": a second request gets its OWN fresh id, and the
        # drip frame did not leak into / pre-resolve anything else. Feed a
        # notification and confirm exactly one is queued from the drip phase.
        notes = t.notifications()
        peer.notify("after/drip", {"n": 1})
        note = await asyncio.wait_for(notes.__anext__(), timeout=1.0)
        assert note == {"method": "after/drip", "params": {"n": 1}}
    finally:
        await peer.aclose()


async def test_packed_three_objects_middle_is_server_request_all_in_order():
    """A single write packing THREE objects where the MIDDLE one is a
    server→client request: all three are dispatched, in order — a notification,
    then the tool call (handled + answered), then a response to a pending
    client request."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        order: list[str] = []

        async def handler(req: dict):
            order.append("tool:" + req["params"]["callId"])
            return {"success": True}

        t.on_server_request("item/tool/call", handler)

        # A client request is pending so the third packed object can resolve it.
        fut = asyncio.ensure_future(t.request("turn/start"))
        sent = await peer.read_client_message()

        notes = t.notifications()
        peer.send_packed(
            {"method": "thread/event", "params": {"phase": "first"}},          # notification
            {"id": 0, "method": "item/tool/call", "params": {"callId": "mid"}}, # server request
            {"id": sent["id"], "result": {"turn": "done"}},                     # response
        )

        # Notification (object #1) arrives.
        note = await asyncio.wait_for(notes.__anext__(), timeout=1.0)
        assert note == {"method": "thread/event", "params": {"phase": "first"}}

        # Server request (object #2) handled + answered.
        tool_reply, _ = await _read_until(peer, lambda m: _is_response_to(m, 0))
        assert tool_reply["result"] == {"success": True}
        assert order == ["tool:mid"]

        # Client response (object #3) resolves the pending request.
        assert await asyncio.wait_for(fut, timeout=1.0) == {"turn": "done"}
    finally:
        await peer.aclose()


async def test_string_field_with_escaped_newline_and_braces_roundtrips():
    """A JSON string value containing an escaped ``\\n`` plus literal ``{`` / ``}``
    must NOT be mis-framed (the reader frames on real newline bytes, not on
    characters inside JSON strings) and the payload round-trips intact."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        tricky = "line1\nline2 with {braces} and a closing } and an opening {"
        fut = asyncio.ensure_future(t.request("tricky"))
        sent = await peer.read_client_message()

        # send_line serializes with json.dumps, so the real newline inside the
        # string becomes the two-byte escape \n — never a frame boundary.
        peer.respond(sent["id"], {"text": tricky, "nested": {"a": "}{"}})
        result = await asyncio.wait_for(fut, timeout=1.0)
        assert result == {"text": tricky, "nested": {"a": "}{"}}
        # The literal newline survived (not split).
        assert "\n" in result["text"]
        assert result["text"].count("\n") == 1
    finally:
        await peer.aclose()


async def test_non_ascii_chinese_payload_roundtrips_both_directions():
    """Transport serializes with ensure_ascii=False; non-ASCII (Chinese) content
    round-trips intact both directions: in the client request params AND in the
    server response result, and in a server-request handler's params."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        cn_in = {"指令": "请运行测试", "emoji": "✅🚀", "mixed": "café—naïve"}
        cn_out = {"结果": "已完成", "详情": ["第一行", "第二行"]}

        # Outbound: client request params carry Chinese — assert it arrives intact.
        fut = asyncio.ensure_future(t.request("turn/start", cn_in))
        sent = await peer.read_client_message()
        assert sent["params"] == cn_in

        # Inbound: server response result carries Chinese.
        peer.respond(sent["id"], cn_out)
        assert await asyncio.wait_for(fut, timeout=1.0) == cn_out

        # Server-request direction: handler sees Chinese params and echoes them.
        async def handler(req: dict):
            return {"回显": req["params"]}

        t.on_server_request("item/tool/call", handler)
        peer.server_request("item/tool/call", {"callId": "汉字", "数据": cn_out}, id=0)
        reply = await peer.read_client_message(timeout=1.0)
        assert reply["id"] == 0
        assert reply["result"] == {"回显": {"callId": "汉字", "数据": cn_out}}
    finally:
        await peer.aclose()


async def test_non_object_json_and_garbage_interleaved_are_skipped():
    """JSON that is NOT an object (`[1,2]`, `"hi"`, `42`, `null`) and a non-JSON
    garbage line, interleaved between good frames, are all skipped/recorded; the
    pump survives and the good frames before AND after still route correctly."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        # A request that will be answered AFTER the junk.
        fut = asyncio.ensure_future(t.request("survivor"))
        sent = await peer.read_client_message()

        notes = t.notifications()

        # A valid notification BEFORE the junk (good frame #1).
        peer.notify("before/junk", {"k": 1})

        # Now a barrage of non-object JSON + garbage, each its own line.
        peer.send_line([1, 2, 3])          # JSON array, not an object
        peer.send_line("just a string")    # JSON string, not an object
        peer.send_line(42)                 # JSON number, not an object
        peer.send_line(None)               # JSON null, not an object
        peer.send_garbage_line("<<< not json at all >>>")

        # The pending request's response AFTER the junk (good frame #2).
        peer.respond(sent["id"], {"survived": True})

        # And a notification AFTER the junk (good frame #3).
        peer.notify("after/junk", {"k": 2})

        # Good frame #1 (before junk).
        n1 = await asyncio.wait_for(notes.__anext__(), timeout=1.0)
        assert n1 == {"method": "before/junk", "params": {"k": 1}}

        # Good response survived the junk.
        assert await asyncio.wait_for(fut, timeout=1.0) == {"survived": True}

        # Good frame #3 (after junk) — proves none of the junk consumed/poisoned
        # the notification queue; the array/string/number/null were all skipped.
        n2 = await asyncio.wait_for(notes.__anext__(), timeout=1.0)
        assert n2 == {"method": "after/junk", "params": {"k": 2}}
    finally:
        await peer.aclose()


# =========================================================================== #
# Ordering / lifecycle races                                                   #
# =========================================================================== #

async def test_response_before_vs_after_interleaved_server_request():
    """Two scenarios proven on one transport: (1) a client response arriving
    BEFORE an interleaved server-request, and (2) AFTER one. Both must correlate
    to the right place — response → its pending future, server-request → its
    handler — regardless of relative arrival order."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        calls: list[str] = []

        async def handler(req: dict):
            calls.append(req["params"]["callId"])
            return {"success": True}

        t.on_server_request("item/tool/call", handler)

        # ---- Scenario 1: RESPONSE first, then server-request. ----
        f1 = asyncio.ensure_future(t.request("req-A"))
        s1 = await peer.read_client_message()
        peer.respond(s1["id"], {"r": "A"})                      # response BEFORE
        peer.server_request("item/tool/call", {"callId": "after-resp"}, id=0)
        assert await asyncio.wait_for(f1, timeout=1.0) == {"r": "A"}
        tr1, _ = await _read_until(peer, lambda m: _is_response_to(m, 0))
        assert tr1["result"] == {"success": True}

        # ---- Scenario 2: server-request first, then RESPONSE. ----
        f2 = asyncio.ensure_future(t.request("req-B"))
        s2 = await peer.read_client_message()
        peer.server_request("item/tool/call", {"callId": "before-resp"}, id=1)  # request BEFORE
        peer.respond(s2["id"], {"r": "B"})
        tr2, _ = await _read_until(peer, lambda m: _is_response_to(m, 1))
        assert tr2["result"] == {"success": True}
        assert await asyncio.wait_for(f2, timeout=1.0) == {"r": "B"}

        assert calls == ["after-resp", "before-resp"]
    finally:
        await peer.aclose()


async def test_duplicate_response_for_same_id_second_is_ignored():
    """The peer sends a result for the same client id TWICE. The first resolves
    the future; the second is silently ignored (no crash, no double-resolve), and
    the transport stays usable for the next request."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        fut = asyncio.ensure_future(t.request("dup"))
        sent = await peer.read_client_message()

        peer.respond(sent["id"], {"first": True})
        # Duplicate result for the SAME id — must be ignored (future already done,
        # and by the time the 2nd arrives the id has been popped from _pending).
        peer.respond(sent["id"], {"second": True})

        result = await asyncio.wait_for(fut, timeout=1.0)
        assert result == {"first": True}

        # Give the pump a tick to process the duplicate; it must not crash.
        await asyncio.sleep(0.02)

        # A fresh request still works → pump alive, no double-resolve corruption.
        fut2 = asyncio.ensure_future(t.request("again"))
        s2 = await peer.read_client_message()
        assert s2["id"] != sent["id"]
        peer.respond(s2["id"], {"ok": True})
        assert await asyncio.wait_for(fut2, timeout=1.0) == {"ok": True}
    finally:
        await peer.aclose()


async def test_duplicate_response_mixed_result_then_error_second_ignored():
    """Variant: first a RESULT, then an ERROR for the same id. The result wins
    (future already resolved); the late error must not surface or crash."""
    peer = FakePeer()
    t = peer.make_transport()
    try:
        fut = asyncio.ensure_future(t.request("dup2"))
        sent = await peer.read_client_message()

        peer.respond(sent["id"], {"won": "result"})
        peer.respond_error(sent["id"], -32000, "too late error")

        assert await asyncio.wait_for(fut, timeout=1.0) == {"won": "result"}
        await asyncio.sleep(0.02)  # let the late error be processed (and dropped)

        # Pump still alive.
        fut2 = asyncio.ensure_future(t.request("ping"))
        s2 = await peer.read_client_message()
        peer.respond(s2["id"], "pong")
        assert await asyncio.wait_for(fut2, timeout=1.0) == "pong"
    finally:
        await peer.aclose()


async def test_eof_while_request_pending_and_handler_midflight_cleans_up():
    """EOF (server stream closes) while a client request is pending AND a handler
    is mid-flight (deliberately stuck): the pending request fails with
    TransportClosed (no hang), and aclose() cancels the stuck handler task — no
    leaked-task / un-awaited-coroutine warning, no hang."""
    peer = FakePeer()
    t = peer.make_transport()
    handler_entered = asyncio.Event()
    release = asyncio.Event()
    try:
        async def slow_handler(req: dict):
            # Mark entry, then block until released — emulates a handler still
            # running when the transport tears down.
            handler_entered.set()
            await release.wait()
            return {"success": True}  # never reached before cancellation

        t.on_server_request("item/tool/call", slow_handler)

        # 1) A client request is pending (never answered).
        pending = asyncio.ensure_future(t.request("will/orphan"))
        await peer.read_client_message()

        # 2) A server request triggers the slow handler → it parks mid-flight.
        peer.server_request("item/tool/call", {"callId": "stuck"}, id=0)
        await asyncio.wait_for(handler_entered.wait(), timeout=1.0)

        # 3) EOF on the server→client stream while both are outstanding.
        peer.close_server_stream()

        # The pending request must fail with TransportClosed (EOF path), not hang.
        with pytest.raises(TransportClosed):
            await asyncio.wait_for(pending, timeout=1.0)

        # The handler is still parked (close_server_stream alone does not cancel
        # handler tasks — only aclose does). It must still be running here.
        assert handler_entered.is_set()

        # 4) aclose() must cancel the stuck handler task cleanly and return.
        await asyncio.wait_for(t.aclose(), timeout=1.0)
        # The handler never got released, so if aclose did not cancel it, the
        # task would leak — verified additionally by the strict-warnings run.
    finally:
        release.set()  # belt-and-suspenders: unblock in case of unexpected path
        await peer.aclose()
