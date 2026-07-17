from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tilldone.core.inputs import InputItem
from tilldone.core.tools import ToolDef


@dataclass(frozen=True)
class ContextDirView:
    id: str
    path: Path
    description: str = ""
    writable: bool = False


@dataclass(frozen=True)
class ResumeHandle:
    backend: str
    token: Mapping[str, Any]


@dataclass(frozen=True)
class AgentRunSpec:
    prompt: "str | list[InputItem]"
    tools: Sequence[ToolDef]
    cwd: Path
    context_dirs: Sequence[ContextDirView] = ()
    system_instructions: str | None = None
    developer_instructions: str | None = None
    output_schema: Mapping[str, Any] | None = None
    model: str | None = None
    model_map: Mapping[str, str] | None = None   # tiered model aliases; cross-backend = v2 non-goal
    timeout_s: float | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    resume: ResumeHandle | None = None
    native: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendError:
    code: Literal["transient", "auth", "timeout", "protocol", "fatal"]
    message: str
    status: int | None = None
    raw: Any = None


@dataclass(frozen=True)
class RunOutcome:
    status: Literal["completed", "failed"]
    resume: ResumeHandle | None
    final_text: str = ""
    structured_output: Mapping[str, Any] | None = None
    usage: Mapping[str, Any] | None = None
    error: BackendError | None = None


@dataclass(frozen=True)
class BackendCapabilities:
    host_tool_calls: bool
    structured_final_output: bool = False
    session_resume: bool = False
    streaming_events: bool = True
    reasoning_events: bool = False
    command_events: bool = False
    file_change_events: bool = False
    in_process_mcp: bool = False
    mcp_stdio: bool = False
    mcp_http: bool = False
    dynamic_tools: bool = False
    mid_turn_steer: bool = False
    fork: bool = False
    multimodal_input: bool = False
    skills: bool = False
