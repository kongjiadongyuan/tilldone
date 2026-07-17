"""The public SDK surface is importable from the top-level package (dependency-free)."""

from __future__ import annotations


def test_public_api_exports_core_surface():
    import tilldone

    expected = {
        "run_task", "CorePolicy", "MaxRoundsExceeded", "BackendFailure",
        "CompletionContract", "OutputExpectation", "ReturnExpectation", "TodoExpectation",
        "ContractState", "ContractEvaluator",
        "ToolDef", "ToolCall", "ToolResult", "ToolExecutor",
        "AgentRunSpec", "RunOutcome", "ResumeHandle", "BackendError",
        "BackendCapabilities", "ContextDirView",
        "AgentEvent", "EventKind", "Backend", "RunHandle", "SessionCapable",
        "Flow", "ActiveRunError", "JsonlEventSink",
        "SessionIndex", "SessionInfo", "SessionRestoreError",
    }
    missing = [n for n in expected if not hasattr(tilldone, n)]
    assert missing == [], f"missing public exports: {missing}"
    assert set(tilldone.__all__) == expected


def test_importing_tilldone_does_not_require_backend_sdks():
    # `import tilldone` must not transitively import a backend SDK (dependency-free core).
    # Checked in a CLEAN subprocess (this session's other tests already import backends,
    # which would pollute sys.modules here).
    import os
    import pathlib
    import subprocess
    import sys

    src = str(pathlib.Path(__file__).resolve().parents[2] / "src")
    code = (
        "import tilldone, sys\n"
        "assert hasattr(tilldone, 'run_task')\n"
        "assert 'tilldone.backends.claude' not in sys.modules\n"
        "assert 'tilldone.backends.codex' not in sys.modules\n"
        "assert 'tilldone.backends.codex_exec' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": src})
    assert result.returncode == 0, result.stderr
