"""Archived app-server rollback boundary proof.

This v2 asset is kept for manual app-server verification, but default pytest
collection excludes ``tests/backends/app_server``.
"""

from __future__ import annotations

import asyncio

from tilldone.backends.app_server.backend import CodexAppServerBackend

from .fake_peer import FakePeer
from .test_backend import THREAD_ID, _NullExecutor, _serve_handshake, _spec


async def test_rollback_is_history_only_keeps_files(tmp_path) -> None:
    """``thread/rollback`` rolls back thread history only and does not touch files."""
    peer = FakePeer()
    be = CodexAppServerBackend(transport_factory=lambda **kw: peer.make_transport(**kw),
                               mode="tools")
    try:
        driver = asyncio.ensure_future(_serve_handshake(peer))
        await be.start(_spec(tmp_path), _NullExecutor())
        await asyncio.wait_for(driver, timeout=2.0)

        marker = tmp_path / "produced_by_a_turn.txt"
        marker.write_text("artifact of the turn being rolled back", encoding="utf-8")
        before = sorted(p.name for p in tmp_path.iterdir())

        rb = asyncio.ensure_future(be.rollback(THREAD_ID, num_turns=1))
        req = await peer.read_client_message()
        assert req["method"] == "thread/rollback"
        assert req["params"]["numTurns"] == 1
        peer.respond(req["id"], {"thread": {"id": THREAD_ID, "turns": []}})
        await asyncio.wait_for(rb, timeout=2.0)

        after = sorted(p.name for p in tmp_path.iterdir())
        assert after == before, "rollback created/removed a file; it must touch zero fs"
        assert marker.exists(), "history-only rollback must leave the file on disk"
        assert marker.read_text(encoding="utf-8") == (
            "artifact of the turn being rolled back")
    finally:
        await be.aclose()
