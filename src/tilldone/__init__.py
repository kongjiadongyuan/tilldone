"""TillDone — a pluggable agent-backend compatibility layer.

The public surface is the backend-agnostic core. Backends pull in optional SDKs, so they
are imported explicitly by the caller (keeps ``import tilldone`` dependency-free):

    from tilldone import run_task, CompletionContract, ReturnExpectation
    from tilldone.backends.claude import ClaudeBackend      # needs `claude` extra
    from tilldone.backends.codex import CodexBackend        # needs `codex` extra
"""

from __future__ import annotations

from tilldone.core.backend import Backend, RunHandle, SessionCapable
from tilldone.core.contract import (
    CompletionContract,
    ContractEvaluator,
    ContractState,
    OutputExpectation,
    ReturnExpectation,
    TodoExpectation,
)
from tilldone.core.enums import EventKind
from tilldone.core.events import AgentEvent
from tilldone.core.loop import BackendFailure, CorePolicy, MaxRoundsExceeded, run_task
from tilldone.core.spec import (
    AgentRunSpec,
    BackendCapabilities,
    BackendError,
    ContextDirView,
    ResumeHandle,
    RunOutcome,
)
from tilldone.core.tools import ToolCall, ToolDef, ToolExecutor, ToolResult
from tilldone.core.event_sink import JsonlEventSink
from tilldone.core.flow import ActiveRunError, Flow
from tilldone.core.sessions import SessionIndex, SessionInfo, SessionRestoreError

__all__ = [
    # driver
    "run_task",
    "CorePolicy",
    "MaxRoundsExceeded",
    "BackendFailure",
    # completion contract
    "CompletionContract",
    "OutputExpectation",
    "ReturnExpectation",
    "TodoExpectation",
    "ContractState",
    "ContractEvaluator",
    # tools
    "ToolDef",
    "ToolCall",
    "ToolResult",
    "ToolExecutor",
    # run spec / outcome / capabilities
    "AgentRunSpec",
    "RunOutcome",
    "ResumeHandle",
    "BackendError",
    "BackendCapabilities",
    "ContextDirView",
    # events
    "AgentEvent",
    "EventKind",
    # backend seam
    "Backend",
    "RunHandle",
    "SessionCapable",
    # SDK surface: Flow, viewer feed, session index
    "Flow",
    "ActiveRunError",
    "JsonlEventSink",
    "SessionIndex",
    "SessionInfo",
    "SessionRestoreError",
]
