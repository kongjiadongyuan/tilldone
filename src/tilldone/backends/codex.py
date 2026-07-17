"""Public Codex backend.

The product Codex path is intentionally lightweight: ``codex exec --json`` plus the
TillDone localhost MCP bridge for contract and custom tools. The richer
``codex app-server`` driver is archived under :mod:`tilldone.backends.app_server`.
"""

from __future__ import annotations

from tilldone.backends.codex_exec import CodexExecBackend


class CodexBackend(CodexExecBackend):
    """Default Codex backend: exec + MCP injection."""


__all__ = ["CodexBackend"]
