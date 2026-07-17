"""Localhost streamable-HTTP MCP bridge — exposes core ToolDefs to a Codex subprocess.

Codex (``codex exec``) cannot call in-process Python tools, so the host runs a tiny
localhost MCP server and injects it via ``-c mcp_servers.tilldone.url=...``. Each MCP
tool call is forwarded to the core ToolExecutor (same handlers as Claude's in-process
MCP), so contract + custom tools behave identically across backends.

Security (RT-MCP-2): binds 127.0.0.1 only, on an ephemeral port (RT-MCP-1), and requires
a per-run bearer token. Lifecycle is explicit (start/stop) for clean teardown (RT-MCP-1).
Uses the low-level mcp Server so ToolDef JSON Schemas pass through verbatim (POC note).
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, TextContent, Tool
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Mount

from tilldone.core.tools import ToolCall, ToolDef, ToolExecutor

TOKEN_ENV_VAR = "TILLDONE_MCP_TOKEN"
SERVER_NAME = "tilldone"


def _normalized_schema(schema: dict[str, Any]) -> dict[str, Any]:
    out = dict(schema)
    if out.get("type") == "object" and "properties" not in out:
        out["properties"] = {}
    return out


@dataclass(frozen=True)
class BridgeInfo:
    url: str
    token: str
    token_env_var: str = TOKEN_ENV_VAR


def _bearer_guard(inner, token: str):
    """Pure-ASGI middleware: reject any HTTP request lacking the bearer token.

    Pure ASGI (not BaseHTTPMiddleware) so it does not buffer/break MCP's SSE streams.
    """
    async def app(scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            if headers.get(b"authorization", b"").decode() != f"Bearer {token}":
                await Response("unauthorized", status_code=401)(scope, receive, send)
                return
        await inner(scope, receive, send)

    return app


class McpToolBridge:
    """A per-run localhost MCP server forwarding tool calls to a core ToolExecutor."""

    def __init__(self, tools: Sequence[ToolDef], executor: ToolExecutor, *, token: str | None = None):
        self._tools = list(tools)
        self._executor = executor
        self._token = token or secrets.token_urlsafe(32)
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None
        self._port: int | None = None
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def _build_app(self) -> Starlette:
        mcp_server: Server = Server(SERVER_NAME)
        tool_objs = [
            Tool(name=td.name, description=td.description,
                 inputSchema=_normalized_schema(dict(td.input_schema)))
            for td in self._tools
        ]

        @mcp_server.list_tools()
        async def _list_tools() -> list[Tool]:
            return tool_objs

        @mcp_server.call_tool()
        async def _call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
            self._call_count += 1
            result = await self._executor.invoke(
                ToolCall(call_id=f"codex-{name}-{self._call_count}", name=name,
                         arguments=arguments or {}, raw_name=f"mcp__{SERVER_NAME}__{name}")
            )
            return CallToolResult(
                content=[TextContent(type="text", text=result.content)],
                isError=result.is_error,
            )

        # json_response=True returns plain JSON (not an SSE stream) for each request — more
        # robust over a short-lived per-run server (avoids "ASGI callable returned without
        # completing response" from unfinished SSE streams); the streamable-HTTP spec allows
        # either, and the Codex client accepts JSON.
        manager = StreamableHTTPSessionManager(app=mcp_server, stateless=True, json_response=True)

        @contextlib.asynccontextmanager
        async def lifespan(_app):
            async with manager.run():
                yield

        async def handle(scope, receive, send):
            await manager.handle_request(scope, receive, send)

        app = Starlette(routes=[Mount("/mcp", app=handle)], lifespan=lifespan)
        return app

    async def start(self) -> BridgeInfo:
        app = _bearer_guard(self._build_app(), self._token)
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning",
                                lifespan="on")
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        # Wait for the socket to bind so we can read the ephemeral port.
        while not self._server.started:
            await asyncio.sleep(0.01)
            if self._task.done():  # startup failed
                self._task.result()  # re-raise
        sock = self._server.servers[0].sockets[0]
        self._port = sock.getsockname()[1]
        return BridgeInfo(url=f"http://127.0.0.1:{self._port}/mcp", token=self._token)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._task, timeout=5)
        self._server = None
        self._task = None

    async def __aenter__(self) -> BridgeInfo:
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.stop()
