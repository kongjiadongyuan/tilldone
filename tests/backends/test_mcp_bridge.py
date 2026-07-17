"""MCP bridge tests — exercised with a real MCP streamable-HTTP CLIENT (no Codex, no billing)."""

from __future__ import annotations

import asyncio

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from tilldone.backends.mcp_bridge import McpToolBridge
from tilldone.core.contract import (
    CompletionContract,
    ContractEvaluator,
    ContractState,
    ReturnExpectation,
    TodoExpectation,
)
from tilldone.core.registry import ToolRegistry


def _setup(contract: CompletionContract):
    st = ContractState()
    reg = ToolRegistry(st, contract)
    reg.add_contract_tools()
    defs = ContractEvaluator(contract).tool_defs(host_tool_calls=True)
    return st, reg, defs


_BOOL_CONTRACT = CompletionContract(
    outputs=(), returns=(ReturnExpectation("ok", "bool", True),),
    todos=(TodoExpectation("build"),), require_finish=True)


async def test_bridge_lists_and_forwards_to_same_state():
    st, reg, defs = _setup(_BOOL_CONTRACT)
    bridge = McpToolBridge(defs, reg)
    info = await bridge.start()
    try:
        assert info.url.startswith("http://127.0.0.1:") and info.url.endswith("/mcp")
        async with streamablehttp_client(
            info.url, headers={"Authorization": f"Bearer {info.token}"}
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                assert {t.name for t in listed.tools} == {"return", "todo", "finish"}
                result = await session.call_tool("return", {"ok": True})
                assert result.isError in (False, None)
        # the forwarded MCP call mutated the SAME core ContractState
        assert st.returns == {"ok": True}
        assert bridge.call_count == 1
    finally:
        await bridge.stop()


async def test_bridge_advertises_typed_return_and_rejects_wrong_type():
    contract = CompletionContract(outputs=(), returns=(ReturnExpectation("answer", "int", True),),
                                  todos=(), require_finish=False)
    st, reg, defs = _setup(contract)
    bridge = McpToolBridge(defs, reg)
    info = await bridge.start()
    try:
        async with streamablehttp_client(
            info.url, headers={"Authorization": f"Bearer {info.token}"}
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                ret = next(t for t in listed.tools if t.name == "return")
                assert ret.inputSchema["properties"]["answer"] == {"type": "integer"}
                # wrong type is rejected by the registry (RT-RET-2) and not recorded
                result = await session.call_tool("return", {"answer": "4"})
                assert result.isError is True
                assert "answer" not in st.returns
    finally:
        await bridge.stop()


async def test_bridge_rejects_missing_bearer_token():
    st, reg, defs = _setup(_BOOL_CONTRACT)
    bridge = McpToolBridge(defs, reg)
    info = await bridge.start()

    async def _connect_without_token():
        async with streamablehttp_client(info.url) as (read, write, _):  # no Authorization
            async with ClientSession(read, write) as session:
                await session.initialize()

    try:
        with pytest.raises(Exception):
            await asyncio.wait_for(_connect_without_token(), timeout=15)
    finally:
        await bridge.stop()


async def test_two_bridges_get_distinct_ephemeral_ports():
    _, reg1, defs1 = _setup(_BOOL_CONTRACT)
    _, reg2, defs2 = _setup(_BOOL_CONTRACT)
    b1, b2 = McpToolBridge(defs1, reg1), McpToolBridge(defs2, reg2)
    i1 = await b1.start()
    i2 = await b2.start()
    try:
        assert i1.url != i2.url and i1.token != i2.token  # ephemeral ports + per-run tokens
    finally:
        await b1.stop()
        await b2.stop()
