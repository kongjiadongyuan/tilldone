"""Tools-mode tests for CodexAppServerBackend (M3 Lane A).

The MAIN app-server path: ``mode="tools"`` →
``host_tool_calls=True``, ``thread/start`` carries ``dynamicTools`` (the contract
return/todo/finish + custom :class:`ToolDef`s core already placed in
``spec.tools``), ``turn/start`` carries **no** ``outputSchema``, and the model
drives the contract by calling those tools. Each call arrives as an
``item/tool/call`` server→client request, is dispatched to the host executor (the
core :class:`~tilldone.core.registry.ToolRegistry`), and is answered with a
``DynamicToolCallResponse {success, contentItems:[{type:"inputText",text}]}``.
The trailing ``agentMessage`` ("Done.") is narration — never JSON-parsed —
so ``outcome.structured_output`` is ``None`` (the tool channel is the sole
ContractState authority).

All tests drive a *real* :class:`AppServerTransport` through the in-memory
``FakePeer`` (NO codex spawn, NO gateway). Wire shapes match the live tools-mode
protocol: ``dynamicTools`` on thread/start; ``item/tool/call`` server-request with
``{threadId,turnId,callId,namespace,tool,arguments}``; reply
``{success:true,contentItems:[{type:"inputText",text:"ok"}]}``; trailing
``"Done."`` agentMessage.

asyncio_mode="auto" (pyproject) → async tests need no decorator.
"""

from __future__ import annotations

import asyncio

import pytest

from tilldone.backends.app_server.backend import (
    _dynamic_tool_entry,
    _tool_call_from_params,
    CodexAppServerBackend,
)
from tilldone.core.enums import EventKind
from tilldone.core.spec import AgentRunSpec
from tilldone.core.tools import ToolCall, ToolDef, ToolResult
from tilldone.testing.conformance import assert_protocol_invariants

from .fake_peer import FakePeer
from .test_backend import (
    THREAD_ID,
    TURN_ID,
    _respond_initialize,
    _respond_thread_start,
    _serve_handshake,
    _notify_token_usage,
    _notify_turn_completed,
    _notify_turn_started,
)

# The three core contract ToolDefs, exactly as ContractEvaluator.tool_defs(
# host_tool_calls=True) emits them (typed return schema, namespace="contract"),
# plus a custom tool to prove namespace=None is omitted on the wire.
_RETURN_DEF = ToolDef(
    "return", "Submit the declared return values using their exact names and types.",
    {"type": "object", "properties": {"answer": {"type": "string"}},
     "additionalProperties": False, "required": ["answer"]},
    kind="contract", namespace="contract",
)
_TODO_DEF = ToolDef(
    "todo", "Mark a declared todo complete.",
    {"type": "object", "properties": {"name": {"type": "string"}, "note": {"type": "string"}},
     "required": ["name"]},
    kind="contract", namespace="contract",
)
_FINISH_DEF = ToolDef(
    "finish", "Declare the task complete.",
    {"type": "object", "properties": {"note": {"type": "string"}}},
    kind="contract", namespace="contract",
)
_CUSTOM_DEF = ToolDef(
    "search", "A host-side custom tool.",
    {"type": "object", "properties": {"q": {"type": "string"}}},
    kind="custom",  # namespace defaults to None → must be omitted on the wire.
)
_CONTRACT_TOOLS = (_RETURN_DEF, _TODO_DEF, _FINISH_DEF, _CUSTOM_DEF)


class RecordingExecutor:
    """A host ToolExecutor (ToolRegistry stand-in) that records every dispatched
    ToolCall and returns a scripted ToolResult keyed by tool name.

    Mirrors the ToolExecutor protocol (``async invoke(ToolCall) -> ToolResult``).
    The backend dispatches each ``item/tool/call`` here; the test then asserts the
    parsed ToolCall (bare name, namespace travels separately, arguments, call_id).
    """

    def __init__(self, results: dict[str, ToolResult] | None = None) -> None:
        self.calls: list[ToolCall] = []
        self._results = results or {}

    async def invoke(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return self._results.get(call.name, ToolResult(content="ok"))


def _spec(tmp_path, *, tools=_CONTRACT_TOOLS, **kw) -> AgentRunSpec:
    """A tools-mode spec: core sends NO output_schema (host_tool_calls=True) and
    spec.tools already contains the contract + custom ToolDefs."""
    return AgentRunSpec(
        prompt=kw.pop("prompt", "Submit answer=DELTA-9082 then finish. Use the tools."),
        tools=tools,
        cwd=tmp_path,
        output_schema=None,  # core sends none in tools mode (loop.py)
        **kw,
    )


async def _collect(handle):
    return [ev async for ev in handle.events]


# --------------------------------------------------------------------------- #
# Tools-mode server-side scripting (frame shapes from probe1_1A_tool.jsonl)    #
# --------------------------------------------------------------------------- #

def _notify_tool_call_started(peer, *, call_id, tool, arguments, namespace="contract",
                              turn_id=TURN_ID, thread_id=THREAD_ID):
    """item/started for a dynamicToolCall (the TOOL_CALL half of the pair)."""
    peer.notify(
        "item/started",
        {"item": {"type": "dynamicToolCall", "id": call_id, "namespace": namespace,
                  "tool": tool, "arguments": arguments, "status": "inProgress",
                  "contentItems": None, "success": None},
         "threadId": thread_id, "turnId": turn_id},
    )


def _notify_tool_call_completed(peer, *, call_id, tool, arguments, success=True,
                                content="ok", namespace="contract",
                                turn_id=TURN_ID, thread_id=THREAD_ID):
    """item/completed for a dynamicToolCall (the TOOL_RESULT half — same call_id)."""
    peer.notify(
        "item/completed",
        {"item": {"type": "dynamicToolCall", "id": call_id, "namespace": namespace,
                  "tool": tool, "arguments": arguments, "status": "completed",
                  "contentItems": [{"type": "inputText", "text": content}],
                  "success": success},
         "threadId": thread_id, "turnId": turn_id},
    )


def _notify_narration(peer, *, text="Done.", turn_id=TURN_ID, thread_id=THREAD_ID,
                      item_id="msg_narr"):
    """The trailing agentMessage (phase=final_answer) — pure narration in tools mode."""
    peer.notify(
        "item/started",
        {"item": {"type": "agentMessage", "id": item_id, "text": "", "phase": "final_answer"},
         "threadId": thread_id, "turnId": turn_id},
    )
    peer.notify(
        "item/completed",
        {"item": {"type": "agentMessage", "id": item_id, "text": text, "phase": "final_answer"},
         "threadId": thread_id, "turnId": turn_id},
    )


async def _serve_one_tool_call(peer, *, server_id, call_id, tool, arguments,
                               namespace="contract", reply_content="ok"):
    """Drive ONE full tool round-trip and assert the backend's reply.

    started → server_request(item/tool/call) → [backend invokes executor + replies]
    → read + assert the DynamicToolCallResponse → completed. Returns the reply msg.
    """
    _notify_tool_call_started(peer, call_id=call_id, tool=tool, arguments=arguments,
                              namespace=namespace)
    peer.server_request(
        "item/tool/call",
        {"threadId": THREAD_ID, "turnId": TURN_ID, "callId": call_id,
         "namespace": namespace, "tool": tool, "arguments": arguments},
        id=server_id,
    )
    reply = await peer.read_client_message()
    _notify_tool_call_completed(peer, call_id=call_id, tool=tool, arguments=arguments,
                                content=reply_content)
    return reply


# --------------------------------------------------------------------------- #
# capabilities() reflects mode                                                 #
# --------------------------------------------------------------------------- #

def test_capabilities_tools_mode_is_default():
    """Default mode is 'tools' → host_tool_calls=True + dynamic_tools=True."""
    caps = CodexAppServerBackend().capabilities()
    assert caps.host_tool_calls is True
    assert caps.dynamic_tools is True
    assert caps.structured_final_output is False
    # Shared event/resume/multimodal flags are identical across modes.
    assert caps.session_resume is True
    assert caps.streaming_events is True
    assert caps.multimodal_input is True


def test_capabilities_structured_mode_explicit():
    """mode='structured' → the M2 surface (host_tool_calls=False, no dynamic_tools)."""
    caps = CodexAppServerBackend(mode="structured").capabilities()
    assert caps.host_tool_calls is False
    assert caps.dynamic_tools is False
    assert caps.structured_final_output is True


def test_bad_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        CodexAppServerBackend(mode="nonsense")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# pure helpers: ToolDef → dynamicTools entry; params → ToolCall                #
# --------------------------------------------------------------------------- #

def test_dynamic_tool_entry_maps_schema_and_namespace():
    entry = _dynamic_tool_entry(_RETURN_DEF)
    assert entry["name"] == "return"
    assert entry["description"].startswith("Submit the declared return")
    assert entry["inputSchema"] == _RETURN_DEF.input_schema
    assert entry["namespace"] == "contract"


def test_dynamic_tool_entry_omits_none_namespace():
    """A namespace-less (custom) tool must NOT carry a ``namespace`` key."""
    entry = _dynamic_tool_entry(_CUSTOM_DEF)
    assert entry["name"] == "search"
    assert "namespace" not in entry


def test_tool_call_from_params_uses_bare_tool_name():
    """ToolCall.name is the BARE tool name (ToolRegistry.invoke keys on it);
    namespace is a sibling wire field, NOT folded into the dispatch name."""
    call = _tool_call_from_params(
        {"threadId": THREAD_ID, "turnId": TURN_ID, "callId": "c1",
         "namespace": "contract", "tool": "return", "arguments": {"answer": "X"}})
    assert call.name == "return"  # bare, not "contract/return" or "contract.return"
    assert call.raw_name == "return"
    assert call.call_id == "c1"
    assert call.arguments == {"answer": "X"}


def test_tool_call_from_params_missing_arguments_defaults_empty():
    call = _tool_call_from_params(
        {"callId": "c2", "tool": "finish"})  # finish often has no/empty args
    assert call.name == "finish"
    assert call.arguments == {}


# --------------------------------------------------------------------------- #
# thread/start carries dynamicTools mapped from spec.tools                      #
# --------------------------------------------------------------------------- #

async def test_thread_start_carries_dynamic_tools_and_turn_has_no_output_schema(tmp_path):
    """thread/start.dynamicTools = mapped spec.tools (return/todo/finish + custom,
    namespace='contract' on the contract three, omitted on the custom one);
    turn/start carries NO outputSchema (core sends none in tools mode)."""
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)  # default tools
    try:
        start_task = asyncio.ensure_future(be.start(_spec(tmp_path), ex))

        init = await peer.read_client_message()
        assert init["params"]["capabilities"]["experimentalApi"] is True
        _respond_initialize(peer, init["id"])
        initialized = await peer.read_client_message()
        assert initialized["method"] == "initialized"

        start = await peer.read_client_message()
        assert start["method"] == "thread/start"
        dyn = start["params"]["dynamicTools"]
        by_name = {d["name"]: d for d in dyn}
        assert set(by_name) == {"return", "todo", "finish", "search"}
        assert by_name["return"]["namespace"] == "contract"
        assert by_name["todo"]["namespace"] == "contract"
        assert by_name["finish"]["namespace"] == "contract"
        assert "namespace" not in by_name["search"]  # custom tool: namespace omitted
        assert by_name["return"]["inputSchema"] == _RETURN_DEF.input_schema
        _respond_thread_start(peer, start["id"])

        turn = await peer.read_client_message()
        assert turn["method"] == "turn/start"
        assert "outputSchema" not in turn["params"], "tools mode sends NO outputSchema"
        assert turn["params"]["threadId"] == THREAD_ID
        peer.respond(turn["id"], {"turn": {"id": TURN_ID, "status": "inProgress"}})

        handle = await asyncio.wait_for(start_task, timeout=2.0)
        # Finish the run cleanly so the consumer ends.
        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        _notify_narration(peer)
        _notify_turn_completed(peer)
        await asyncio.wait_for(consume, timeout=2.0)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# item/tool/call → executor.invoke → DynamicToolCallResponse                    #
# --------------------------------------------------------------------------- #

async def test_tool_call_dispatched_to_executor_and_replied_success(tmp_path):
    """A scripted item/tool/call for 'return' → backend invokes the executor with
    the right ToolCall (bare name, namespace separate, args, callId) and replies
    {success:true, contentItems:[{type:'inputText', text:<result.content>}]}."""
    peer = FakePeer()
    ex = RecordingExecutor({"return": ToolResult(content="recorded=['answer']")})
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    try:
        driver_handshake = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), ex)
        await asyncio.wait_for(driver_handshake, timeout=2.0)

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        reply = await _serve_one_tool_call(
            peer, server_id=0, call_id="call_R", tool="return",
            arguments={"answer": "DELTA-9082"}, reply_content="recorded=['answer']")
        _notify_narration(peer)
        _notify_turn_completed(peer)
        await asyncio.wait_for(consume, timeout=2.0)

        # The backend's reply to the server request.
        assert reply["id"] == 0  # server's own id space (echoed back)
        assert reply["result"]["success"] is True
        assert reply["result"]["contentItems"] == [
            {"type": "inputText", "text": "recorded=['answer']"}]

        # The executor saw exactly one dispatched ToolCall, correctly parsed.
        assert len(ex.calls) == 1
        call = ex.calls[0]
        assert call.name == "return"          # BARE name (ToolRegistry dispatch key)
        assert call.raw_name == "return"
        assert call.call_id == "call_R"
        assert call.arguments == {"answer": "DELTA-9082"}
    finally:
        await be.aclose()
        await peer.aclose()


async def test_tool_error_result_replies_success_false_with_error_text(tmp_path):
    """is_error=True ToolResult → success:false, error text still in inputText
."""
    peer = FakePeer()
    ex = RecordingExecutor(
        {"return": ToolResult(content="rejected=['answer (expected int)']", is_error=True)})
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    try:
        driver_handshake = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), ex)
        await asyncio.wait_for(driver_handshake, timeout=2.0)

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        reply = await _serve_one_tool_call(
            peer, server_id=0, call_id="call_bad", tool="return",
            arguments={"answer": 4}, reply_content="rejected=['answer (expected int)']")
        _notify_narration(peer)
        _notify_turn_completed(peer)
        await asyncio.wait_for(consume, timeout=2.0)

        assert reply["result"]["success"] is False
        assert reply["result"]["contentItems"] == [
            {"type": "inputText", "text": "rejected=['answer (expected int)']"}]
        assert ex.calls[0].arguments == {"answer": 4}
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# events: dynamicToolCall → TOOL_CALL + TOOL_RESULT (matching call_id)          #
# --------------------------------------------------------------------------- #

async def test_dynamic_tool_call_emits_tool_call_and_result_paired(tmp_path):
    """item/started+completed for a dynamicToolCall → TOOL_CALL then TOOL_RESULT
    sharing the same call_id; assert_protocol_invariants' pairing rule holds."""
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    try:
        driver_handshake = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), ex)
        await asyncio.wait_for(driver_handshake, timeout=2.0)

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        await _serve_one_tool_call(
            peer, server_id=0, call_id="call_R", tool="return",
            arguments={"answer": "DELTA-9082"})
        await _serve_one_tool_call(
            peer, server_id=1, call_id="call_F", tool="finish", arguments={"note": "done"})
        _notify_narration(peer)
        _notify_turn_completed(peer)
        events = await asyncio.wait_for(consume, timeout=2.0)
        outcome = await handle.outcome()

        # Pairing: each TOOL_CALL has a matching TOOL_RESULT with the same call_id.
        calls = [e for e in events if e.kind == EventKind.TOOL_CALL]
        results = [e for e in events if e.kind == EventKind.TOOL_RESULT]
        assert {e.call_id for e in calls} == {"call_R", "call_F"}
        assert {e.call_id for e in results} == {"call_R", "call_F"}
        for r in results:
            assert r.data["is_error"] is False
        # The whole stream obeys the cross-backend generalization gate.
        assert_protocol_invariants(events, outcome)
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# outcome: structured_output is None in tools mode; status by turn.status       #
# --------------------------------------------------------------------------- #

async def test_outcome_tools_mode_no_structured_output_status_completed(tmp_path):
    """tools mode: outcome.structured_output is None (contract satisfied via tool
    calls); status=completed by turn.status; the trailing 'Done.' agentMessage is
    narration (final_text) and is NEVER strict-JSON parsed."""
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    try:
        driver_handshake = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), ex)
        await asyncio.wait_for(driver_handshake, timeout=2.0)

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        await _serve_one_tool_call(
            peer, server_id=0, call_id="call_R", tool="return",
            arguments={"answer": "DELTA-9082"})
        _notify_token_usage(peer)
        _notify_narration(peer, text="Done.")  # NOT JSON — must not break anything
        _notify_turn_completed(peer, status="completed")
        await asyncio.wait_for(consume, timeout=2.0)
        outcome = await handle.outcome()

        assert outcome.status == "completed"
        assert outcome.structured_output is None  # tool channel is the authority
        assert outcome.final_text == "Done."      # narration captured, not parsed
        assert outcome.error is None
        assert outcome.resume is not None
        assert outcome.resume.backend == "codex_app_server"
        assert outcome.resume.token["thread_id"] == THREAD_ID
        assert outcome.usage is not None
    finally:
        await be.aclose()
        await peer.aclose()


async def test_outcome_tools_mode_turn_failed_is_failed(tmp_path):
    """A failed turn in tools mode → outcome.status=failed with a classified error;
    no JSON-parse path is taken (structured_output stays None)."""
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    try:
        driver_handshake = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), ex)
        await asyncio.wait_for(driver_handshake, timeout=2.0)

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        _notify_turn_completed(
            peer, status="failed",
            error={"message": "overloaded", "codexErrorInfo": "serverOverloaded"})
        events = await asyncio.wait_for(consume, timeout=2.0)
        outcome = await handle.outcome()

        assert events[-1].kind == EventKind.RUN_FAILED
        assert outcome.status == "failed"
        assert outcome.error is not None
        assert outcome.error.code == "transient"
        assert outcome.structured_output is None
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# tools mode: an UNEXPECTED (non item/tool/call) server-request is a protocol err #
# --------------------------------------------------------------------------- #

async def test_tools_mode_unexpected_other_server_request_is_protocol_error(tmp_path):
    """Tools mode registers a handler ONLY for item/tool/call; any OTHER
    server-request still lands in unexpected_server_requests → protocol error."""
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    try:
        driver_handshake = asyncio.ensure_future(_serve_handshake(peer))
        handle = await be.start(_spec(tmp_path), ex)
        await asyncio.wait_for(driver_handshake, timeout=2.0)

        consume = asyncio.ensure_future(_collect(handle))
        _notify_turn_started(peer)
        # A server-request the tools path does NOT model (e.g. an approval prompt).
        peer.server_request("applyPatchApproval", {"callId": "x"}, id=7)
        await asyncio.sleep(0.02)
        _notify_narration(peer)
        _notify_turn_completed(peer, status="completed")
        _ = await asyncio.wait_for(consume, timeout=2.0)
        outcome = await handle.outcome()

        assert outcome.status == "failed"
        assert outcome.error is not None
        assert outcome.error.code == "protocol"
        # The executor was never invoked (the unexpected method isn't item/tool/call).
        assert ex.calls == []
    finally:
        await be.aclose()
        await peer.aclose()


# --------------------------------------------------------------------------- #
# tools-mode multi-round reuse: resident transport, re-wired handler            #
# --------------------------------------------------------------------------- #

TURN_ID_2 = "019ef56d-cccc-7c41-9ae0-c9958f25576b"


async def test_tools_mode_multiround_reuses_resident_and_dispatches(tmp_path):
    """A within-run second round (resume handle → resident transport) issues another
    turn/start on the SAME thread (no re-handshake, no thread/resume) and STILL
    dispatches the round-2 tool call to the executor (handler re-wired)."""
    peer = FakePeer()
    ex = RecordingExecutor()
    be = CodexAppServerBackend(transport_factory=peer.make_transport)
    try:
        # Round 1.
        driver_handshake = asyncio.ensure_future(_serve_handshake(peer))
        handle1 = await be.start(_spec(tmp_path), ex)
        await asyncio.wait_for(driver_handshake, timeout=2.0)
        consume1 = asyncio.ensure_future(_collect(handle1))
        _notify_turn_started(peer)
        await _serve_one_tool_call(
            peer, server_id=0, call_id="call_R1", tool="return",
            arguments={"answer": "ONE"})
        _notify_narration(peer)
        _notify_turn_completed(peer)
        await asyncio.wait_for(consume1, timeout=2.0)
        outcome1 = await handle1.outcome()
        assert outcome1.resume is not None

        # Round 2 — resume against the resident thread: ONLY a turn/start (no
        # initialize / thread/start / thread/resume).
        spec2 = _spec(tmp_path, resume=outcome1.resume)
        start_task = asyncio.ensure_future(be.start(spec2, ex))
        msg = await peer.read_client_message()
        assert msg["method"] == "turn/start"
        assert msg["params"]["threadId"] == THREAD_ID
        assert "outputSchema" not in msg["params"]
        peer.respond(msg["id"], {"turn": {"id": TURN_ID_2, "status": "inProgress"}})
        handle2 = await asyncio.wait_for(start_task, timeout=2.0)

        consume2 = asyncio.ensure_future(_collect(handle2))
        _notify_turn_started(peer, turn_id=TURN_ID_2)
        # A round-2 tool call (turn_id=TURN_ID_2) must reach the executor.
        _notify_tool_call_started(peer, call_id="call_R2", tool="return",
                                  arguments={"answer": "TWO"}, turn_id=TURN_ID_2)
        peer.server_request(
            "item/tool/call",
            {"threadId": THREAD_ID, "turnId": TURN_ID_2, "callId": "call_R2",
             "namespace": "contract", "tool": "return", "arguments": {"answer": "TWO"}},
            id=1)
        reply2 = await peer.read_client_message()
        _notify_tool_call_completed(peer, call_id="call_R2", tool="return",
                                    arguments={"answer": "TWO"}, turn_id=TURN_ID_2)
        _notify_narration(peer, turn_id=TURN_ID_2)
        _notify_turn_completed(peer, turn_id=TURN_ID_2)
        await asyncio.wait_for(consume2, timeout=2.0)
        outcome2 = await handle2.outcome()

        assert reply2["result"]["success"] is True
        assert outcome2.status == "completed"
        assert outcome2.structured_output is None
        # Both rounds' calls reached the SAME executor instance.
        assert [c.arguments["answer"] for c in ex.calls] == ["ONE", "TWO"]
    finally:
        await be.aclose()
        await peer.aclose()
