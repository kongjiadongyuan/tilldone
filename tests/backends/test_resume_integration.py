"""LIVE resume-continuity probe (RT-SESS-1 / W2, TOP-3 'must-test-or-explode').

Verifies that same-backend resume preserves model context — NOT merely that `resume`
returns exit-code 0. A high-entropy nonce is established in turn 1 (and is NOT present in
turn 2's prompt); turn 2, resumed from turn 1, must recall it. This machine has
`disable_response_storage = true` for Codex, the exact high-risk case this environment hits.

Run:  TILLDONE_CLAUDE_E2E=1 TILLDONE_CODEX_E2E=1 uv run pytest tests/backends/test_resume_integration.py -v
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import pytest

from tilldone.backends.claude import ClaudeBackend
from tilldone.backends.codex_exec import CodexExecBackend
from tilldone.core.spec import AgentRunSpec
from tilldone.core.tools import ToolCall, ToolResult

pytestmark = pytest.mark.integration

_claude = pytest.mark.skipif(os.environ.get("TILLDONE_CLAUDE_E2E") != "1",
                             reason="set TILLDONE_CLAUDE_E2E=1")
_codex = pytest.mark.skipif(os.environ.get("TILLDONE_CODEX_E2E") != "1",
                            reason="set TILLDONE_CODEX_E2E=1")


class _StubExecutor:
    async def invoke(self, call: ToolCall) -> ToolResult:
        return ToolResult(content="ok")


async def _drain_text(handle) -> str:
    parts = []
    async for ev in handle.events:
        if ev.kind.value == "assistant_text":
            parts.append(ev.data.get("text", ""))
    outcome = await handle.outcome()
    parts.append(outcome.final_text or "")
    return " ".join(parts)


async def _two_turn_recall(backend, nonce: str, cwd: Path) -> str:
    # turn 1 — establish the nonce (appears ONLY here). Neutral framing ("reference code")
    # so the probe tests RESUME CONTEXT, not a model's willingness to echo a "secret token"
    # (some aligned models refuse the latter framing — orthogonal to resume continuity).
    spec1 = AgentRunSpec(
        prompt=f"For this working session, the reference code is {nonce}. "
               f"Acknowledge by replying: noted",
        tools=(), cwd=cwd)
    h1 = await backend.start(spec1, _StubExecutor())
    async for _ in h1.events:
        pass
    out1 = await h1.outcome()
    assert out1.resume is not None and out1.resume.token.get("session_id"), "no resume handle"

    # turn 2 — recall, resumed; the nonce is NOT in this prompt
    spec2 = AgentRunSpec(
        prompt="What is the reference code for this working session? "
               "Reply with only the code, nothing else.",
        tools=(), cwd=cwd, resume=out1.resume)
    h2 = await backend.start(spec2, _StubExecutor())
    return await _drain_text(h2)


@_claude
async def test_claude_resume_preserves_context(tmp_path: Path):
    nonce = secrets.token_hex(8)
    recalled = await _two_turn_recall(ClaudeBackend(), nonce, tmp_path)
    assert nonce in recalled, f"claude resume lost context: {nonce!r} not recalled in {recalled!r}"


@_codex
async def test_codex_resume_preserves_context(tmp_path: Path):
    nonce = secrets.token_hex(8)
    recalled = await _two_turn_recall(
        CodexExecBackend(extra_config=['model_reasoning_effort="low"']), nonce, tmp_path)
    assert nonce in recalled, (
        f"codex resume lost context (disable_response_storage=true): "
        f"{nonce!r} not recalled in {recalled!r}")
