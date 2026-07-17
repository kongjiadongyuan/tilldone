"""Archived TillDone v2 · app-server JSON-RPC transport.

This package is retained as a v2.x reference asset and is no longer part of the
product Codex backend. The current public Codex path is
``tilldone.backends.codex.CodexBackend`` (``codex exec --json`` + MCP injection).
Default pytest collection excludes the app-server tests; run them manually with
``--include-archived-app-server`` when maintaining this archive.

Scope: pure JSON-RPC stdio transport over ``codex app-server --stdio`` — spawn,
newline framing, request/response multiplexing, server-request routing,
notification pump, per-request timeout, and clean shutdown. It is deliberately
ignorant of any Backend protocol, event normalization, or thread/turn/contract
semantics. It MUST NOT import :mod:`tilldone.core`.
"""

from __future__ import annotations

from .transport import (
    AppServerTransport,
    JsonRpcError,
    TransportClosed,
)

__all__ = [
    "AppServerTransport",
    "JsonRpcError",
    "TransportClosed",
]
