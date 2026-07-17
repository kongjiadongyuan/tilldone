"""Standing architecture guard: the **0-core-diff** north star.

core must never depend on any backend. These tests are cheap, deterministic, and
run on every unit pass so a regression is caught the moment someone writes a
named-backend branch into the contract loop.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys

_CORE = pathlib.Path(__file__).resolve().parent.parent / "src" / "tilldone" / "core"


def _imported_modules(py: pathlib.Path):
    """Yield every module name imported by a source file (absolute + relative)."""
    tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module


def test_core_imports_no_backend() -> None:
    """No file under tilldone.core may import anything from tilldone.backends.

    Backend capabilities reach the system via the backend protocol — never by
    core reaching down into a concrete backend package.
    """
    offenders: list[str] = []
    for py in sorted(_CORE.glob("*.py")):
        for mod in _imported_modules(py):
            if "backends" in mod.split("."):
                offenders.append(f"{py.name} → {mod}")
    assert not offenders, (
        "0-core-diff violated: core imports a backend:\n  " + "\n  ".join(offenders))


def test_run_task_is_backend_agnostic() -> None:
    """run_task / the contract loop must contain no named-backend branch.

    The seam is `backend.start(spec, executor)` + capabilities() — never an
    `if <backend>` in core.
    """
    loop_src = (_CORE / "loop.py").read_text(encoding="utf-8")
    for marker in ("app_server", "codex_app_server", "CodexAppServer", "codex_exec", "ClaudeBackend"):
        assert marker not in loop_src, (
            f"loop.py references {marker!r} — run_task must stay backend-agnostic")


def test_run_task_is_ignorant_of_rich_control() -> None:
    """run_task / the contract loop must reference NONE of the rich-control surface
    (steer / fork / rollback / SteerCapable / BranchingCapable).

    This is the highest-risk 0-core-diff area: fork/steer/rollback are a
    **capability-scoped control plane**, NOT part of ``run_task``. A rollback branch
    leaking into the contract loop is an architecture failure. The rich capabilities
    reach the system ONLY via the optional Protocols (``SteerCapable`` /
    ``BranchingCapable``), reachable by a control-plane caller — never from
    ``run_task``, which stays bound to ``backend.start`` + ``capabilities()`` alone.
    """
    loop_src = (_CORE / "loop.py").read_text(encoding="utf-8")
    for marker in ("steer", "fork", "rollback", "SteerCapable", "BranchingCapable"):
        assert marker not in loop_src, (
            f"loop.py references rich-control {marker!r} — run_task must stay ignorant of "
            "the steer/fork/rollback control plane (0-core-diff: a rollback branch in the "
            "contract loop is an architecture failure)")


def test_importing_core_does_not_load_app_server() -> None:
    """Importing the core contract loop must not transitively pull in the
    app_server backend package (a runtime complement to the static AST scan)."""
    for mod in list(sys.modules):
        if "tilldone.backends.app_server" in mod:
            del sys.modules[mod]
    importlib.import_module("tilldone.core.loop")
    leaked = [m for m in sys.modules if "tilldone.backends.app_server" in m]
    assert not leaked, f"importing core loaded app_server modules: {leaked}"
