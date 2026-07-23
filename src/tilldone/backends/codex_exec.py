"""CodexExecBackend — drives a single turn over `codex exec --json` (live).

Per-turn driver: spawns `codex exec`, injecting a localhost MCP bridge (so contract +
custom tools work identically to Claude's in-process MCP), parses the thread/turn/item
JSONL stream into normalized AgentEvents, and resolves a RunOutcome. NO GATING: launches
`--dangerously-bypass-approvals-and-sandbox`. Verified against codex-cli 0.139.0.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator
from time import monotonic
from typing import Any

from tilldone.backends.mcp_bridge import SERVER_NAME, BridgeInfo, McpToolBridge
from tilldone.core.enums import EventKind
from tilldone.core.events import AgentEvent
from tilldone.core.inputs import TextInput, normalize_prompt
from tilldone.core.spec import (
    AgentRunSpec,
    BackendCapabilities,
    BackendError,
    ResumeHandle,
    RunOutcome,
)
from tilldone.core.tools import ToolExecutor

DEFAULT_CODEX_BIN = os.environ.get("TILLDONE_CODEX_BIN", "codex")


async def _iter_lines(stream: Any) -> AsyncIterator[bytes]:
    """Yield newline-delimited byte lines without StreamReader.readline's size limit."""
    buf = bytearray()
    read = getattr(stream, "read", None)

    if read is None:
        # Existing minimal test doubles predate the chunked stream interface.
        async for chunk in stream:
            buf.extend(chunk)
            while (nl := buf.find(b"\n")) >= 0:
                line = bytes(buf[:nl])
                del buf[:nl + 1]
                yield line
        if buf:
            yield bytes(buf)
        return

    while True:
        chunk = await read(65536)
        if not chunk:
            if buf:
                yield bytes(buf)
            return
        buf.extend(chunk)
        while (nl := buf.find(b"\n")) >= 0:
            line = bytes(buf[:nl])
            del buf[:nl + 1]
            yield line


def _extra_config_key(kv: str) -> str:
    """Return the TOML override key part from a `codex -c key=value` fragment."""
    return kv.split("=", 1)[0].strip()


def _targets_reserved_tilldone_server(kv: str) -> bool:
    key = _extra_config_key(kv)
    reserved_roots = (
        f"mcp_servers.{SERVER_NAME}",
        f'mcp_servers."{SERVER_NAME}"',
        f"mcp_servers.'{SERVER_NAME}'",
    )
    return any(key == root or key.startswith(root + ".") for root in reserved_roots)


def _validate_extra_config(extra_config: list[str] | None) -> None:
    for kv in extra_config or []:
        if _targets_reserved_tilldone_server(kv):
            raise ValueError(
                "mcp server name 'tilldone' is reserved for TillDone contract and "
                "custom tools; choose a different mcp_servers.<name> in extra_config"
            )


class _NullExecutor:
    """A no-op executor for bridge health probes (no tools => invoke never called)."""

    async def invoke(self, call):  # pragma: no cover - never reached during a probe
        from tilldone.core.tools import ToolResult
        return ToolResult(content="", is_error=True)


_NULL_EXECUTOR = _NullExecutor()


def _prompt_text(prompt: Any) -> str:
    return "\n".join(i.text for i in normalize_prompt(prompt) if isinstance(i, TextInput))


# Codex collapses upstream failures into a structured error event carrying a message string.
# These markers (matched on the STRUCTURED message field, not raw stderr) mark a retry-able
# condition — capacity/overload/rate-limit/timeout/5xx; everything else is fatal. Keeps
# classification on a structured signal, not a stderr regex (RT-CFG-4).
_TRANSIENT_ERROR_MARKERS = (
    "at capacity", "capacity", "unavailable", "overloaded", "rate limit", "rate_limit",
    "try again", "timeout", "timed out", "temporarily", "503", "429", "500", "502", "504",
)


def _classify_codex_error(detail: str) -> str:
    low = detail.lower()
    return "transient" if any(m in low for m in _TRANSIENT_ERROR_MARKERS) else "fatal"


def build_argv(spec: AgentRunSpec, bridge: BridgeInfo, *,
               codex_bin: str = DEFAULT_CODEX_BIN,
               extra_config: list[str] | None = None) -> list[str]:
    """Build the codex argv. RT-GATE-1: MUST contain --dangerously-bypass-approvals-and-sandbox."""
    _validate_extra_config(extra_config)
    common = [
        "--dangerously-bypass-approvals-and-sandbox",
        "--json",
        "--skip-git-repo-check",
        "-c", f'mcp_servers.{SERVER_NAME}.url="{bridge.url}"',
        "-c", f'mcp_servers.{SERVER_NAME}.bearer_token_env_var="{bridge.token_env_var}"',
    ]
    for kv in extra_config or []:
        common += ["-c", kv]
    if spec.model:
        common += ["-m", spec.model]
    prompt = _prompt_text(spec.prompt)

    if spec.resume is not None:
        session_id = spec.resume.token.get("session_id")
        # `resume` reuses the session's original cwd/dirs, so no -C / --add-dir here.
        return [codex_bin, "exec", "resume", *common, str(session_id), prompt]

    argv = [codex_bin, "exec", *common, "-C", str(spec.cwd)]
    for cd in spec.context_dirs:
        argv += ["--add-dir", str(cd.path)]
    argv.append(prompt)
    return argv


class CodexRunHandle:
    """Parses codex `exec --json` (thread/turn/item) stdout into AgentEvents + a RunOutcome."""

    def __init__(self, proc, bridge, *, timeout_s: float | None = None,
                 idle_timeout_s: float | None = None):
        self._proc = proc
        self._bridge = bridge
        self._timeout = timeout_s
        self._idle_timeout = idle_timeout_s
        self._last_activity = monotonic()
        self._timeout_kind: str | None = None
        self._stderr_tail: list[str] = []
        self._seq = 0
        self._thread_id: str | None = None
        self._final = ""
        self._usage: dict[str, Any] | None = None
        self._failed = False
        self._error: BackendError | None = None
        self._terminated = False
        self._timed_out = False

    def _ev(self, kind: EventKind, data: dict[str, Any], **kw) -> AgentEvent:
        self._seq += 1
        return AgentEvent(kind, self._seq, data, session_id=self._thread_id, **kw)

    async def _watchdog(self) -> None:
        # idle: 距上次 codex 事件超过 idle_timeout → 判定静默卡死;
        # hard: 距开始超过 timeout_s → 绝对上限兜底。任一触发即 kill codex。
        start = monotonic()
        thresholds = [t for t in (self._idle_timeout, self._timeout) if t]
        tick = min(min(thresholds), 5.0) if thresholds else 5.0
        while True:
            await asyncio.sleep(tick)
            now = monotonic()
            if self._idle_timeout and now - self._last_activity > self._idle_timeout:
                self._timeout_kind = "idle"
            elif self._timeout and now - start > self._timeout:
                self._timeout_kind = "hard"
            else:
                continue
            self._timed_out = True
            with contextlib.suppress(ProcessLookupError):
                self._proc.kill()
            return

    async def _drain_stderr(self) -> None:
        # stderr MUST be drained or its OS pipe fills and the codex child blocks/leaks.
        stderr = getattr(self._proc, "stderr", None)
        if stderr is None:
            return
        try:
            async for raw in _iter_lines(stderr):
                line = raw.decode(errors="replace").rstrip()
                if line:
                    self._stderr_tail.append(line)
                    del self._stderr_tail[:-20]  # keep only the last 20 lines
        except Exception:
            pass

    @property
    async def events(self) -> AsyncIterator[AgentEvent]:
        watchdog = (asyncio.create_task(self._watchdog())
                    if (self._timeout or self._idle_timeout) else None)
        stderr_task = asyncio.create_task(self._drain_stderr())
        try:
            async for raw in _iter_lines(self._proc.stdout):
                self._last_activity = monotonic()  # 收到一行 codex 事件即重置 idle 计时
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # codex may interleave non-JSON; ignore for the stable view
                for ev in self._adapt(obj):
                    yield ev
            await self._proc.wait()
            if self._timed_out:
                self._failed = True
                self._terminated = True
                if self._timeout_kind == "idle":
                    msg = f"codex idle (no output) for {self._idle_timeout}s"
                else:
                    msg = f"codex exceeded hard timeout {self._timeout}s"
                self._error = BackendError(code="timeout", message=msg)
                yield self._ev(EventKind.RUN_FAILED, {"error": "timeout"})
            elif not self._terminated:
                code = self._proc.returncode
                tail = " | ".join(self._stderr_tail[-3:])
                self._failed = True
                if self._error is None:  # keep a detail captured from an `error` notice
                    self._error = BackendError(
                        code="protocol" if code in (0, None) else "fatal",
                        message=f"codex exited {code} without turn.completed: {tail}"[:500])
                yield self._ev(EventKind.RUN_FAILED, {"exit": code})
        finally:
            # deterministic cleanup: kill a still-running child, reap it, drain tasks, stop bridge.
            if self._proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    self._proc.kill()
                with contextlib.suppress(Exception):
                    await self._proc.wait()
            for task in (watchdog, stderr_task):
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
            await self._bridge.stop()

    def _adapt(self, obj: dict[str, Any]) -> list[AgentEvent]:
        t = obj.get("type")
        if t == "thread.started":
            self._thread_id = obj.get("thread_id") or self._thread_id
            return [self._ev(EventKind.RUN_STARTED, {"thread_id": self._thread_id}, raw=obj)]
        if t == "turn.started":
            return []
        if t in ("item.started", "item.completed"):
            return self._adapt_item(t, obj.get("item") or {}, obj)
        if t == "turn.completed":
            self._usage = obj.get("usage")
            self._terminated = True
            return [self._ev(EventKind.RUN_COMPLETED, {"final": self._final}, raw=obj)]
        if t == "error":
            # Codex emits `error` THEN `turn.failed` for one failure. Treat `error` as a
            # non-terminal notice (capture detail for classification); the single terminal
            # is the following `turn.failed` — avoids emitting two terminal events.
            detail = obj.get("message") or obj.get("error") or obj
            self._error = BackendError(code=_classify_codex_error(str(detail)),
                                       message=str(detail)[:500], raw=obj)
            return [self._ev(EventKind.BACKEND_NOTICE, {"error": str(detail)[:200]}, raw=obj)]
        if t == "turn.failed":
            self._terminated = True
            self._failed = True
            detail = obj.get("error") or obj.get("message") or obj
            if self._error is None:
                self._error = BackendError(code=_classify_codex_error(str(detail)),
                                           message=str(detail)[:500], raw=obj)
            return [self._ev(EventKind.RUN_FAILED, {"error": str(detail)[:200]}, raw=obj)]
        return []

    def _adapt_item(self, phase: str, item: dict[str, Any], obj: dict[str, Any]) -> list[AgentEvent]:
        it = item.get("type")
        call_id = item.get("id")
        if it == "agent_message":
            if phase == "item.completed":
                self._final = item.get("text", "") or self._final
                return [self._ev(EventKind.ASSISTANT_TEXT, {"text": item.get("text", "")},
                                 call_id=call_id, raw=obj)]
            return []
        if it == "reasoning":
            if phase == "item.completed":
                return [self._ev(EventKind.REASONING, {"text": item.get("text", "")},
                                 call_id=call_id, raw=obj)]
            return []
        if it == "command_execution":
            return [self._ev(EventKind.COMMAND,
                             {"command": item.get("command"), "exit_code": item.get("exit_code"),
                              "status": item.get("status"),
                              "phase": "begin" if phase == "item.started" else "end"},
                             call_id=call_id, raw=obj)]
        if it == "mcp_tool_call":
            name = item.get("tool")
            if phase == "item.started":
                return [self._ev(EventKind.TOOL_CALL, {"name": name, "server": item.get("server")},
                                 call_id=call_id, raw=obj)]
            result = item.get("result") or {}
            is_error = item.get("status") == "failed" or bool(result.get("isError"))
            return [self._ev(EventKind.TOOL_RESULT, {"name": name, "is_error": is_error},
                             call_id=call_id, raw=obj)]
        if it in ("file_change", "patch", "file_update"):
            return [self._ev(EventKind.FILE_CHANGED, {"item_type": it}, call_id=call_id, raw=obj)]
        return []  # unknown item type -> not part of the stable view

    async def outcome(self) -> RunOutcome:
        resume = (
            ResumeHandle("codex", {"session_id": self._thread_id})
            if self._thread_id
            else None
        )
        if self._failed:
            return RunOutcome(status="failed", resume=resume, final_text=self._final,
                              usage=self._usage, error=self._error)
        return RunOutcome(status="completed", resume=resume, final_text=self._final,
                          usage=self._usage)

    async def cancel(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            self._proc.kill()
        await self._bridge.stop()


class CodexExecBackend:
    """Backend over `codex exec --json` + a localhost MCP bridge (one turn per start())."""

    name = "codex"

    def __init__(self, *, codex_bin: str = DEFAULT_CODEX_BIN, extra_config: list[str] | None = None,
                 bridge_factory=McpToolBridge):
        _validate_extra_config(extra_config)
        self._bin = codex_bin
        self._extra = list(extra_config or [])
        self._bridge_factory = bridge_factory  # injectable for capability probing / tests

    async def probe_host_tools(self) -> bool:
        """RT-CAP-1 runtime probe: can the MCP bridge (host tool calls) actually start?
        Binds a throwaway bridge and tears it down. Returns False on failure — never hangs.
        The driver uses this to avoid wiring contract tools to a bridge that can't come up
        (else the model could never report the contract and the run would spin to max_rounds)."""
        bridge = self._bridge_factory((), _NULL_EXECUTOR)
        try:
            await bridge.start()
        except Exception:
            return False
        await bridge.stop()
        return True

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            host_tool_calls=True,
            structured_final_output=False,  # --output-schema read-path deferred (v1 uses MCP)
            session_resume=True,
            streaming_events=True,
            reasoning_events=True,
            command_events=True,
            file_change_events=True,
            mcp_http=True,
            multimodal_input=True,
        )

    async def start(self, spec: AgentRunSpec, executor: ToolExecutor) -> CodexRunHandle:
        if spec.model_map:
            # RT-CFG-1: never silently ignore model_map on Codex — cross-backend tiered model
            # aliases are a v2 non-goal. Fail fast so a caller can't believe haiku ran when it
            # didn't. Pass an explicit `model` instead.
            raise ValueError(
                "CodexExecBackend does not honor model_map (cross-backend model aliases are a v2 "
                "non-goal); pass an explicit model= instead — fail-fast per RT-CFG-1")
        if spec.resume is not None and spec.resume.backend != self.name:
            # RT-SESS-5: a resume handle minted by another backend must NOT be silently used as
            # an opaque string (cross-backend resume is a non-goal). Fail fast.
            raise ValueError(
                f"resume handle is for backend {spec.resume.backend!r}, not {self.name!r} — "
                "cross-backend resume is a non-goal (RT-SESS-5)")
        bridge = self._bridge_factory(spec.tools, executor)
        info = await bridge.start()  # fail-fast (raises) if the bridge can't bind — never hangs
        argv = build_argv(spec, info, codex_bin=self._bin, extra_config=self._extra)
        env = {**os.environ, **dict(spec.env), info.token_env_var: info.token}
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
        except Exception:
            await bridge.stop()
            raise
        return CodexRunHandle(proc, bridge, timeout_s=spec.timeout_s,
                              idle_timeout_s=spec.idle_timeout_s)

    async def aclose(self) -> None:
        ...
