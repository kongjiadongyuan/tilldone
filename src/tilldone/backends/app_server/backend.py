"""Archived CodexAppServerBackend — the ``codex app-server`` backend.

This driver is kept as a v2.x archive; it is no longer the product Codex backend.
Import ``tilldone.backends.codex.CodexBackend`` for the supported Codex path
(``codex exec --json`` + MCP injection).

Two mutually-exclusive contract channels live here, selected by ``mode`` (chosen
once at construction); core's ``run_task`` works UNCHANGED with either because
the backend declares a single coherent ``capabilities()`` and core branches only
on ``host_tool_calls``.

**tools mode** (``mode="tools"``, the default path, ``host_tool_calls=True``)
  Aligns with the Claude / CodexExec backends. ``thread/start`` carries
  ``dynamicTools`` = the contract (return/todo/finish) + custom :class:`ToolDef`s
  that core already placed in ``spec.tools``; ``turn/start`` carries **no**
  ``outputSchema``. The model drives the contract by calling those tools, which
  arrive as ``item/tool/call`` server→client requests; each is dispatched to the
  host ``executor`` (the core :class:`~tilldone.core.registry.ToolRegistry`,
  which mutates ``ContractState``) and answered with a
  ``DynamicToolCallResponse {success, contentItems:[{type:"inputText",text}]}``.
  The trailing ``agentMessage`` (e.g. ``"Done."``) is **narration only** — it is
  NOT parsed and ``structured_output`` is ``None`` (the tool channel is the sole
  ContractState authority). ``dynamicTools`` requires ``experimentalApi`` on
  ``initialize`` — already sent.

**structured mode** (``mode="structured"``, ``host_tool_calls=False``)
  Optional alternate path. ``thread/start`` wires **no** ``dynamicTools`` and
  registers **no** server-request handler — it expects zero server→client
  requests, so any that arrive land in
  :attr:`AppServerTransport.unexpected_server_requests` and are a protocol error
  at outcome time. ``turn/start`` carries an ``outputSchema``; the final
  ``agentMessage`` is strict-``json.loads``-parsed into ``structured_output``.

Style mirrors :mod:`tilldone.backends.codex_exec` (host-executor wiring,
event/outcome/resume/error-classification/cleanup) without importing it.

Wire shapes:
* ``thread/start`` uses the STRING sandbox form: ``sandbox:"danger-full-access"``.
* ``turn/start`` uses the OBJECT form: ``sandboxPolicy:{type:"dangerFullAccess"}``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from tilldone.backends.app_server.events import map_notification
from tilldone.backends.app_server.transport import AppServerTransport, JsonRpcError
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
from tilldone.core.tools import ToolCall, ToolDef, ToolExecutor

#: Stable backend id — distinct from ``codex`` (the codex_exec backend). The two
#: drivers have different capability surfaces, so they MUST be distinguishable
#: and resume handles are not cross-compatible.
BACKEND_NAME = "codex_app_server"

CLIENT_NAME = "tilldone"
CLIENT_VERSION = "0.0.0"
_TILLDONE_CODEX_HOME_ENV = "TILLDONE_CODEX_HOME"
_EPHEMERAL_CODEX_HOME_PREFIX = "run-"
_EPHEMERAL_CODEX_HOME_OWNER = ".tilldone-ephemeral-home.json"
_EPHEMERAL_CODEX_HOME_GC_GRACE_S = 60 * 60

#: The contract channel this backend instance drives (chosen once at construction).
Mode = Literal["tools", "structured"]

#: The server→client request the model issues for every dynamicTool invocation
#: The only server→client request this backend handles for dynamic tools. Its
#: ``params`` are ``DynamicToolCallParams`` ({threadId, turnId, callId, namespace,
#: tool, arguments}); the reply is a ``DynamicToolCallResponse``.
_TOOL_CALL_METHOD = "item/tool/call"

#: Trust-full sandbox/approval posture, in the two wire shapes app-server demands.
#: ``thread/start`` and ``thread/resume`` take the STRING ``sandbox`` form + a STRING
#: ``approvalPolicy``; ``turn/start`` takes the OBJECT ``sandboxPolicy`` form.
#: Re-supplying these two on ``thread/resume`` preserves the posture across a process
#: restart; omitting them silently downgrades to on-request/workspace-write.
_SANDBOX_STRING = "danger-full-access"
_APPROVAL_POLICY = "never"
#: The SandboxPolicy ``type`` a preserved resume result MUST report at top level.
_SANDBOX_OBJECT_TYPE = "dangerFullAccess"

# -- RT-CAP-1: layered capability detection -------
#
# Tools mode rides the contract (return/todo/finish + custom ToolDefs) on
# ``thread/start.dynamicTools``, which REQUIRES that ``initialize`` negotiated
# ``capabilities.experimentalApi:true``. When it was NOT negotiated the
# live ``codex app-server`` rejects ``thread/start`` (carrying dynamicTools) with
# JSON-RPC ``-32600 "requires experimentalApi capability"`` .
#
# The detection is LAYERED:
#   1. offline gate (:meth:`CodexAppServerBackend.probe_capabilities`) — the methods a
#      mode drives are all in the known app-server request set; cheap, no spawn.
#   2. lightweight runtime gate — the ``initialize``/``initialized`` handshake. BUT note
#      the experimentalApi acceptance is NOT observable here: the InitializeResponse
#      InitializeResponse carries ONLY {codexHome, platformFamily, platformOs, userAgent} — NO ``capabilities``
#      echo. So the handshake succeeding does not prove experimentalApi negotiated.
#   3. runtime check — the FIRST ``thread/start`` (it carries dynamicTools) is the check:
#      a ``-32600`` rejection is the authoritative "experimentalApi not negotiated"
#      signal. RT-CAP-1 turns that opaque -32600 into a CLEAR fail-fast — it must NOT
# silently degrade to a tool-less / text-protocol run.
#: JSON-RPC error code app-server returns for an unsupported/invalid request (the
#: experimentalApi-not-negotiated rejection of ``thread/start.dynamicTools``).
_EXPERIMENTAL_API_ERROR_CODE = -32600
#: Substrings that mark a JsonRpcError as the dynamicTools-unavailable signal (the live
#: message is "requires experimentalApi capability"; ``dynamic`` is kept as a defensive
#: alias in case a future build rewords it while still meaning the same gate).
_EXPERIMENTAL_API_MARKERS = ("experimentalapi", "experimental api")
#: The clear capability error raised on the fail-fast (RT-CAP-1). It NAMES experimentalApi
#: and states the no-fallback posture so an operator sees exactly why the run was refused.
_CAPABILITY_FAILFAST_MSG = (
    "app_server custom/dynamic tools unavailable: experimentalApi capability was not "
    "negotiated on initialize, so thread/start rejected the contract's dynamicTools "
    "(-32600) — failing fast, no text-protocol fallback (RT-CAP-1)"
)
#: The known app-server CLIENT-request method-set this backend drives, used by the offline
#: gate. Lifted from the live wire facts + the schema dir; NOT hard-coded
#: capability assumptions — just "do the methods this mode needs exist in the protocol".
_KNOWN_CLIENT_METHODS = frozenset({
    "initialize", "thread/start", "thread/resume", "turn/start", "turn/interrupt",
    "thread/archive",
})
#: The methods each mode actually drives (offline membership check input).
_MODE_REQUIRED_METHODS: dict[str, tuple[str, ...]] = {
    "tools": ("initialize", "thread/start", "turn/start"),
    "structured": ("initialize", "thread/start", "turn/start"),
}

# -- Rich-control CONTROL-PLANE wire facts (capability-scoped; NOT run_task) ---------
#
# steer / fork / rollback operate on a RESIDENT thread's transport. They are reachable
# ONLY via the SteerCapable / BranchingCapable optional protocols (core/backend.py),
# never from run_task — the 0-core-diff boundary (test_zero_core_diff.py) enforces that
# loop.py references none of them and that rollback touches zero files.

#: turn/steer (TurnSteerParams → TurnSteerResponse {turnId}). ``expectedTurnId`` is the
#: REQUIRED active-turn precondition (source = TurnStartResponse.turn.id); a mismatch /
#: already-completed / consecutive-steer surfaces as the activeTurnNotSteerable error.
_STEER_METHOD = "turn/steer"
#: thread/fork (ThreadForkParams → ThreadForkResponse) — result.thread.id is the NEW thread.
_FORK_METHOD = "thread/fork"
#: thread/rollback (ThreadRollbackParams {threadId, numTurns} → ThreadRollbackResponse).
#: numTurns schema min=0 but its DESCRIPTION requires >=1 ("Must be >= 1"); the client
#: enforces >=1 so a numTurns=0 call fails fast (defeats a no-op false-green). Rollback
#: rolls back thread HISTORY ONLY and does NOT revert local file changes.
_ROLLBACK_METHOD = "thread/rollback"
#: The codexErrorInfo marker for a steer-race (TurnError / JSON-RPC error data); used to
#: turn an opaque steer rejection into a clear, named fail-fast.
_STEER_RACE_MARKER = "activeturnnotsteerable"

# -- Skills CONTROL-PLANE wire facts (capability-scoped; NOT run_task) ---------------
#
# skills/list + skills/extraRoots/set operate on a RESIDENT thread's transport, reachable
# ONLY via the SkillsCapable optional protocol (core/backend.py) — never from run_task
# (skills discovery is not part of the contract loop; the 0-core-diff boundary holds).

#: skills/list (SkillsListParams {cwds?, forceReload?} → SkillsListResponse {data:[...]}).
#: ``data`` is one SkillsListEntry per cwd ({cwd, errors, skills:[SkillMetadata]}).
_SKILLS_LIST_METHOD = "skills/list"
#: skills/extraRoots/set (SkillsExtraRootsSetParams {extraRoots:[AbsolutePathBuf]} →
#: SkillsExtraRootsSetResponse {}). The roots are absolute (schema-required). RT-APP-
#: SKILLS-ROOT: this is GLOBAL, NOT thread-scoped — a registered root persists in the
#: (shared) home until reset, so a caller MUST clear it on teardown.
_SKILLS_EXTRA_ROOTS_SET_METHOD = "skills/extraRoots/set"


def _is_experimental_api_rejection(exc: JsonRpcError) -> bool:
    """True if ``exc`` is the dynamicTools-requires-experimentalApi rejection.

    Keyed on the structured ``-32600`` code AND the message marker ( live
    wording "requires experimentalApi capability") — not a blind code match, so an
    unrelated -32600 (e.g. a malformed param) is NOT swallowed by the capability path.
    """
    if exc.code != _EXPERIMENTAL_API_ERROR_CODE:
        return False
    low = str(exc.message or "").lower()
    return any(m in low for m in _EXPERIMENTAL_API_MARKERS)


def _normalize_codex_home(path: str | os.PathLike[str]) -> str:
    """Expand a CODEX_HOME path to an absolute filesystem path."""

    expanded = os.path.expandvars(os.fspath(path))
    return str(Path(expanded).expanduser().resolve())


def _default_codex_runs_root(project_dir: str | os.PathLike[str] | None = None) -> Path:
    """Project-local parent for backend-owned ephemeral Codex homes."""

    base = Path(project_dir) if project_dir is not None else Path.cwd()
    return Path(_normalize_codex_home(base / ".tilldone" / "codex-home" / "runs"))


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _ephemeral_codex_home_gc_candidate(home: Path, *, now: float) -> bool:
    """Return True when a backend-owned default home looks safe to reap."""

    try:
        stat = home.stat()
    except OSError:
        return False
    marker = home / _EPHEMERAL_CODEX_HOME_OWNER
    marker_created_at: float | None = None
    with contextlib.suppress(Exception):
        raw = json.loads(marker.read_text(encoding="utf-8"))
        pid = raw.get("pid")
        if isinstance(pid, int) and _pid_exists(pid):
            return False
        created_at = raw.get("created_at")
        if isinstance(created_at, (int, float)):
            marker_created_at = float(created_at)
    newest = max(stat.st_mtime, marker_created_at or 0.0)
    return now - newest >= _EPHEMERAL_CODEX_HOME_GC_GRACE_S


def _gc_stale_default_codex_homes(runs_root: Path) -> None:
    """Best-effort cleanup for crashed default homes under the project-local runs dir."""

    now = time.time()
    try:
        children = list(runs_root.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return
    except OSError:
        return
    for child in children:
        if not child.name.startswith(_EPHEMERAL_CODEX_HOME_PREFIX):
            continue
        if not _ephemeral_codex_home_gc_candidate(child, now=now):
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                child.unlink()


def _default_codex_home(project_dir: str | os.PathLike[str] | None = None) -> str:
    """Project-local, per-run ephemeral Codex home owned by this backend.

    The default is ``<project_dir>/.tilldone/codex-home/runs/run-<uuid>`` where
    ``project_dir`` is the same ``AgentRunSpec.cwd`` sent as app-server ``cwd``.
    Explicit ``CODEX_HOME`` channels opt into a persistent/shared home instead.
    """

    runs_root = _default_codex_runs_root(project_dir)
    _gc_stale_default_codex_homes(runs_root)
    return _normalize_codex_home(
        runs_root / f"{_EPHEMERAL_CODEX_HOME_PREFIX}{uuid.uuid4().hex}")


def _ensure_codex_home(path: str) -> None:
    """Create CODEX_HOME if needed without changing permissions on existing homes."""

    home = Path(path)
    existed = home.exists()
    home.mkdir(parents=True, exist_ok=True)
    if not home.is_dir():
        raise NotADirectoryError(f"CODEX_HOME is not a directory: {path}")
    if not existed:
        with contextlib.suppress(Exception):
            home.chmod(0o700)


def _mark_ephemeral_codex_home(path: str) -> None:
    marker = Path(path) / _EPHEMERAL_CODEX_HOME_OWNER
    with contextlib.suppress(Exception):
        marker.write_text(
            json.dumps({"pid": os.getpid(), "created_at": time.time()}),
            encoding="utf-8",
        )


def _dynamic_tool_entry(td: ToolDef) -> dict[str, Any]:
    """Map a core :class:`ToolDef` to a ``thread/start`` ``dynamicTools`` entry.

    ``ToolDef.input_schema → inputSchema``; ``ToolDef.namespace → namespace``
    (omitted when ``None`` so a namespace-less custom tool stays flat). Shape lifted
    from the golden transcript probe1_1A_tool.jsonl seq5.
    """
    entry: dict[str, Any] = {
        "name": td.name,
        "description": td.description,
        "inputSchema": dict(td.input_schema),
    }
    if td.namespace is not None:
        entry["namespace"] = td.namespace
    return entry


def _tool_call_from_params(params: Mapping[str, Any]) -> ToolCall:
    """Parse ``DynamicToolCallParams`` into a core :class:`ToolCall`.

    The dispatch key is the BARE ``tool`` name (NOT namespace-qualified):
    :meth:`ToolRegistry.invoke` looks handlers up by ``call.name`` keyed on the
    bare contract/custom names (return/todo/finish/<custom>), while ``namespace``
    travels on the wire as a sibling field. ``raw_name`` preserves the wire name
    verbatim. ``arguments`` defaults to ``{}`` when the model omits it.
    """
    return ToolCall(
        call_id=str(params.get("callId") or ""),
        name=str(params.get("tool") or ""),
        arguments=params.get("arguments") or {},
        raw_name=params.get("tool"),
    )


def _dynamic_tool_response(content: str, *, success: bool) -> dict[str, Any]:
    """Build a ``DynamicToolCallResponse`` from a :class:`ToolResult`.

    ``is_error=True`` → ``success:false`` while the error text still rides in the
    ``inputText`` content item.
    """
    return {"success": success, "contentItems": [{"type": "inputText", "text": content}]}

# codexErrorInfo string-enum markers.
# Mapped to the core BackendError code vocabulary (transient|auth|timeout|protocol|fatal).
_AUTH_MARKERS = ("unauthorized", "forbidden", "authentication", "invalidApiKey")
_TRANSIENT_MARKERS = (
    "usageLimitExceeded", "serverOverloaded", "httpConnectionFailed",
    "rateLimit", "rate_limit", "overloaded", "capacity", "unavailable",
    "timeout", "timedOut", "temporarily", "tooManyRequests",
)


# -- RT-APP-USAGE: usage accounting ----------------
#
# Wire shape (the vendored protocol schema,
# field names live-exact): the notification carries top-level ``threadId`` + ``turnId`` +
# a ``tokenUsage`` ``ThreadTokenUsage`` = {``last``, ``total``, ``modelContextWindow``?},
# where ``last``/``total`` are ``TokenUsageBreakdown``s
# {cachedInputTokens, inputTokens, outputTokens, reasoningOutputTokens, totalTokens}:
#   * ``last``  = the usage of THIS turn (per-turn); multiple updates may arrive per turn
#                 and the LAST one is that turn's authoritative final ``last``.
#   * ``total`` = the SERVER's running cumulative across the whole thread.
#   * ``modelContextWindow`` = the model's context size (carried through, never summed).
#
# Budgeting rule: accumulate each turn's FINAL ``last`` keyed by
# (threadId,turnId). Last-write-wins per key both (a) makes multiple updates within a turn
# settle to that turn's final ``last`` (NOT a within-turn sum), and (b) dedups a turn whose
# usage is re-observed (a trailing duplicate update or a retry of the SAME turnId) so it is
# counted ONCE — never double-counted across resume/retry. The reported accumulated
# ``total`` is therefore Σ over distinct turns of each turn's final ``last`` (NOT the wire
# ``total``, which we also preserve as ``server_total`` so no information is lost).

#: The five ``TokenUsageBreakdown`` integer fields (schema-required) we sum element-wise.
_TOKEN_FIELDS = (
    "cachedInputTokens", "inputTokens", "outputTokens", "reasoningOutputTokens",
    "totalTokens",
)


def _sum_breakdowns(breakdowns: "list[Mapping[str, Any]]") -> dict[str, int]:
    """Element-wise sum of ``TokenUsageBreakdown``s over the five token fields.

    Missing fields default to 0 so a partial breakdown (a future build that drops a
    field) degrades gracefully rather than KeyError-ing the accounting.
    """
    return {f: sum(int(b.get(f, 0) or 0) for b in breakdowns) for f in _TOKEN_FIELDS}


class _UsageLedger:
    """Backend-scoped usage ledger: ``(thread_id, turn_id) -> that turn's final ``last``.

    Owned by the :class:`CodexAppServerBackend` (NOT the per-turn RunHandle) so that in
    the resident multi-round / resume case — where ONE backend instance drives several
    turns on one (or a resumed) thread — usage accumulates correctly across turns and a
    re-observed turn is never double-counted. Each :meth:`record` is last-write-wins for
    its (thread_id, turn_id) key:

      * multiple updates within ONE turn → the turn's row is overwritten, so the turn's
        contribution is its FINAL ``last`` (never the within-turn sum);
      * a trailing duplicate / a retry of the SAME (thread_id, turn_id) → overwrites the
        same row, so the turn stays counted ONCE (dedup; no double count across
        resume/retry).

    The accumulated ``total`` (:meth:`accumulated_total`) is the element-wise sum of every
    distinct turn's final ``last``.
    """

    def __init__(self) -> None:
        # (thread_id, turn_id) -> {"last": breakdown}. Insertion order is irrelevant to
        # the sum but kept (plain dict) for a stable turns_counted.
        self._rows: dict[tuple[str, str], Mapping[str, Any]] = {}

    def record(self, thread_id: str, turn_id: str, last: Mapping[str, Any]) -> None:
        """Record a turn's ``last`` breakdown (last-write-wins for its key)."""
        self._rows[(thread_id, turn_id)] = last

    def has_turn(self, thread_id: str, turn_id: str) -> bool:
        return (thread_id, turn_id) in self._rows

    def turn_last(self, thread_id: str, turn_id: str) -> Mapping[str, Any] | None:
        return self._rows.get((thread_id, turn_id))

    def accumulated_total(self) -> dict[str, int]:
        """Σ over distinct turns of each turn's final ``last`` (the dedup-safe total)."""
        return _sum_breakdowns(list(self._rows.values()))

    def turns_counted(self) -> int:
        return len(self._rows)


def _resolve_model(spec: AgentRunSpec) -> str | None:
    """Resolve ``spec.model`` through an EXPLICIT, caller-supplied ``spec.model_map``.

    RT-APP / model_map ("显式映射 … 不定义质量等价" — explicit mapping,
    does NOT define quality equivalence). ``spec.model_map`` is a concrete
    ``{alias → model_id}`` dict the CALLER supplies; if ``spec.model`` is one of its
    KEYS it is resolved to that concrete model_id, otherwise ``spec.model`` passes
    through UNCHANGED (``.get(model, model)``). The resolved value is what reaches the
    ``thread/start`` wire — the model_map's first half of "real-run proof" (the second
    half is observing what actually executed, incl. any RT-APP-REROUTE).

    This is DELIBERATELY a single-backend, explicit resolution — NOT the cross-backend
    tier equivalence that ``CodexExecBackend`` fails fast on (RT-CFG-1). v1 refuses to
    GUESS a tier's meaning across backends; here the caller states the concrete mapping
    for THIS backend, so there is nothing to guess. The two are orthogonal, not
    contradictory: app-server does not inherit a tier's meaning from another backend, it
    applies the caller's explicit map. When ``spec.model`` is ``None`` there is no key to
    resolve, so this returns ``None`` (the backend then sends no ``model`` field) — it
    never INVENTS a model out of the map.
    """
    if spec.model is None:
        return None
    if spec.model_map:
        return spec.model_map.get(spec.model, spec.model)
    return spec.model


def _prompt_text(prompt: Any) -> str:
    """Flatten the prompt to text (structured mode sends a single text input item)."""
    return "\n".join(i.text for i in normalize_prompt(prompt) if isinstance(i, TextInput))


def _runtime_workspace_roots(spec: AgentRunSpec) -> list[str]:
    """``spec.context_dirs`` → absolute paths for ``runtimeWorkspaceRoots``.

    The schema (ThreadStart/ResumeParams ``runtimeWorkspaceRoots``: ``AbsolutePathBuf``)
    REQUIRES absolute paths, so each ``ContextDirView.path`` is resolved to an absolute
    string. This makes the ``@ref``'d dirs explicit workspace roots rather than merely
    reachable via the full-access sandbox — aligning the app-server backend with the
    Claude (``add_dirs``) and CodexExec (``--add-dir``) backends (LaneD parity gap).
    Returns ``[]`` when no context dirs are declared (the field is then omitted).
    """
    return [os.path.abspath(str(cd.path)) for cd in spec.context_dirs]


def _classify_turn_error(error: Mapping[str, Any] | None) -> BackendError:
    """Classify a ``turn.error`` ({message, additionalDetails?, codexErrorInfo?}) into
    a core :class:`BackendError`, keying on the structured ``codexErrorInfo`` signal
    (string enum OR ``{variant:{httpStatusCode}}`` object) — not a stderr regex."""
    error = error or {}
    message = str(error.get("message") or error.get("additionalDetails") or "turn failed")
    info = error.get("codexErrorInfo")

    status: int | None = None
    token = ""
    if isinstance(info, str):
        token = info
    elif isinstance(info, Mapping):
        # e.g. {"variant": {"httpStatusCode": 503}} — find the first http status.
        status = _find_http_status(info)
        token = json.dumps(info)

    low = token.lower()
    if any(m.lower() in low for m in _AUTH_MARKERS):
        code = "auth"
    elif any(m.lower() in low for m in _TRANSIENT_MARKERS):
        code = "transient"
    elif status is not None and 500 <= status < 600:
        code = "transient"
    elif status is not None and 400 <= status < 500:
        code = "auth" if status in (401, 403) else "fatal"
    else:
        code = "fatal"
    return BackendError(code=code, message=message[:500], status=status, raw=error)


def _find_http_status(obj: Any) -> int | None:
    """Recursively locate an ``httpStatusCode`` int in a codexErrorInfo object."""
    if isinstance(obj, Mapping):
        v = obj.get("httpStatusCode")
        if isinstance(v, int):
            return v
        for child in obj.values():
            found = _find_http_status(child)
            if found is not None:
                return found
    return None


class CodexAppServerRunHandle:
    """Drives ONE ``turn/start`` to completion over a (possibly RESIDENT) transport:
    normalizes the transport's notification stream into :class:`AgentEvent`s and
    settles a :class:`RunOutcome`.

    The final ``agentMessage`` is **buffered during iteration** (it cannot be
    fetched later: ``turn/completed.items`` is empty and
    ``thread/turns/items/list`` is unimplemented — ), then strictly
    ``json.loads``-parsed into ``structured_output`` at outcome time.

    Lifecycle: the backend keeps the thread + transport
    RESIDENT across a ``run_task`` multi-round loop and reuses ONE transport for
    successive turns. So this handle **does NOT own / close the transport** — the
    backend owns it (closed by :meth:`CodexAppServerBackend.aclose`). It instead
    **filters the shared notification stream by its own ``turnId``**: a resident
    transport's single ``notifications()`` queue is read by each round's handle in
    turn, and round N's handle must ignore any trailing telemetry left by round
    N-1. Only the ``turn/completed`` for THIS handle's turnId ends THIS stream;
    events tagged with a *different* turnId are skipped. Thread-level notifications
    that carry no turnId (e.g. ``thread/started`` → RUN_STARTED) are always
    attributed to the current handle. The turnId is captured from the ``turn/start``
    response (``result.turn.id``) before any event is consumed.
    """

    def __init__(self, transport: AppServerTransport, *, thread_id: str,
                 turn_id: str | None = None, mode: Mode = "structured",
                 persisted: bool = False,
                 usage_ledger: "_UsageLedger | None" = None,
                 raw_events: bool = False,
                 on_turn_done: "Callable[[str], None] | None" = None) -> None:
        self._t = transport
        self._mode: Mode = mode
        # RT-APP-REASON: when False (default — ``experimentalRawEvents`` not on) reasoning
        # events keep their ``raw_only`` debug-channel tag (a public sink drops them from
        # the default feed). When True the handle re-stamps reasoning ``raw_only=False``
        # so an opted-in caller sees it in the public feed.
        self._raw_events = raw_events
        # RT-APP-CONC: callback to release this turn's per-thread active-turn slot when
        # the stream terminates / ends / is cancelled. Wrapped so it fires AT MOST ONCE
        # (a terminal then a later cancel must not double-release a slot a NEW turn may
        # already hold). None for a standalone handle (no backend, no slot to free).
        self._on_turn_done = on_turn_done
        self._turn_slot_released = False
        # The backend-scoped usage ledger ((threadId,turnId)->final last). Shared across
        # this backend's RunHandles so resident multi-round / resume accumulation +
        # dedup works (RT-APP-USAGE). A standalone handle (no backend) gets its own.
        self._usage_ledger = usage_ledger if usage_ledger is not None else _UsageLedger()
        # Whether THIS handle's thread was started NON-ephemeral (persist mode) and so
        # has a rollout on disk. Decides the ResumeHandle.token shape: a persisted
        # thread carries the NO-GATING posture so a cross-process resume can RE-SUPPLY
        # sandbox/approvalPolicy on thread/resume (the probe2 silent-downgrade fix).
        self._persisted = persisted
        self._thread_id: str | None = thread_id
        # The turnId this handle is bound to (from the turn/start response). Used to
        # filter the resident transport's shared notification stream so a prior
        # round's trailing frames never bleed into this round.
        self._turn_id: str | None = turn_id
        self._seq = 0
        # Final-answer buffering: prefer phase=="final_answer", else last agentMessage.
        self._final_text: str | None = None
        self._final_is_final_answer = False
        # Per-turn usage tracking (RT-APP-USAGE). The ledger holds the accumulated /
        # dedup'd cross-turn state; these three carry THIS turn's wire facts so
        # outcome() can report per-turn last + preserve the server's own total /
        # context-window without losing information. ``_usage_seen`` distinguishes
        # "this completed turn reported NO usage" (→ G5 fail-fast marker) from
        # "reported usage" — never a silent usage=None-as-zero.
        self._usage_seen = False
        self._usage_server_total: Mapping[str, Any] | None = None
        self._usage_ctx_window: Any = None
        # RT-APP-REROUTE: the LAST model/rerouted toModel observed on this turn (None
        # if the server never rerouted). A reroute is surfaced as an observable
        # BACKEND_NOTICE during iteration AND recorded here so outcome() reports the
        # FINAL model that actually executed — never a silent swallow.
        self._rerouted_to: str | None = None
        self._terminal_status: str | None = None
        self._turn_error: Mapping[str, Any] | None = None
        self._iterated = False

    # -- event stream -------------------------------------------------------

    @property
    async def events(self) -> AsyncIterator[AgentEvent]:
        """Yield normalized events for THIS handle's turn with a strictly-increasing seq.

        Emits EXACTLY ONE terminal event (RUN_COMPLETED / RUN_FAILED) as the LAST
        event — derived from ``turn/completed.turn.status`` for THIS turnId — then
        stops. The final agentMessage is captured here because it cannot be
        re-fetched later.

        turnId filtering (resident-transport multi-round): an event tagged with a
        *different* turnId than this handle's is a leftover from another round and
        is skipped (it does NOT advance seq, is NOT yielded, and a stray
        ``turn/completed`` for another turn does NOT terminate this stream). Events
        with no turnId (thread-level lifecycle / telemetry) are attributed here.
        """
        self._iterated = True
        try:
            async for note in self._t.notifications():
                ev = map_notification(note, seq=self._seq + 1)
                if ev is None:
                    continue
                if not self._belongs_to_this_turn(ev):
                    # A trailing frame from a previous (or concurrent) turn on the
                    # same resident transport — ignore it entirely so it neither
                    # pollutes this round's stream nor (if it is a turn/completed
                    # for another turn) prematurely terminates it.
                    continue
                ev = self._project_channel(ev)
                self._seq = ev.seq
                # Track correlation + buffers off the normalized event.
                self._observe(ev)
                yield ev
                if ev.kind in (EventKind.RUN_COMPLETED, EventKind.RUN_FAILED):
                    # turn/completed for THIS turn is the single terminal — stop. Free
                    # the per-thread turn slot so the NEXT turn on this thread may start
                    # (RT-APP-CONC); the finally is the backstop for an EOF-without-
                    # terminal stream.
                    self._free_turn_slot()
                    return
        finally:
            # The notifications iterator ended. If no terminal was seen (transport
            # closed / EOF mid-turn) the slot is still held — free it here so a dead
            # turn never leaks a permanent busy marker (RT-APP-CONC). Idempotent: a
            # normal terminal already freed it above. The transport's own lifecycle is
            # owned by the backend (aclose), not by this handle.
            self._free_turn_slot()

    def _project_channel(self, ev: AgentEvent) -> AgentEvent:
        """RT-APP-REASON: promote reasoning to the public feed iff raw events are opted in.

        events.py tags reasoning (item + delta) ``data["raw_only"]=True`` (debug-channel),
        so a public sink drops it from the default feed. When this handle was minted with
        ``raw_events=True`` (the ``experimentalRawEvents`` opt-in) we re-stamp reasoning
        ``raw_only=False`` so the opted-in caller sees it publicly. Non-reasoning events
        and the default (raw_events off) path pass through untouched.
        """
        if not self._raw_events or ev.kind != EventKind.REASONING:
            return ev
        if ev.data.get("raw_only") is not True:
            return ev
        # Re-stamp a copy public; AgentEvent is frozen so rebuild with patched data.
        new_data = dict(ev.data)
        new_data["raw_only"] = False
        return AgentEvent(
            ev.kind, ev.seq, data=new_data,
            session_id=ev.session_id, thread_id=ev.thread_id, turn_id=ev.turn_id,
            item_id=ev.item_id, call_id=ev.call_id, raw=ev.raw,
        )

    def _free_turn_slot(self) -> None:
        """RT-APP-CONC: release this turn's per-thread active-turn slot, AT MOST ONCE.

        Fires on terminal, EOF-without-terminal, or cancel. The once-guard is essential:
        after a terminal frees the slot, a *new* turn on the same thread may already hold
        it, so a later cancel of THIS (now-finished) handle must NOT free the new turn's
        slot. A standalone handle (no backend) has no callback → no-op.
        """
        if self._turn_slot_released:
            return
        self._turn_slot_released = True
        if self._on_turn_done is not None and self._thread_id:
            self._on_turn_done(str(self._thread_id))

    def _belongs_to_this_turn(self, ev: AgentEvent) -> bool:
        """True if ``ev`` should be attributed to THIS handle's turn.

        If this handle has a bound turnId (the normal case — captured from the
        turn/start response), an event carrying a *different* turnId is a leftover
        from another round on the shared resident stream and is filtered out.
        Events with no turnId (thread-level lifecycle/telemetry, e.g.
        ``thread/started``) are always attributed to the current turn. If the
        handle never learned its turnId (turn/start returned none), filtering is a
        no-op — every event is accepted (single-turn behavior, unchanged).
        """
        if self._turn_id is None:
            return True
        if ev.turn_id is None:
            return True
        return ev.turn_id == self._turn_id

    def _observe(self, ev: AgentEvent) -> None:
        if ev.thread_id:
            self._thread_id = ev.thread_id
        if ev.turn_id:
            self._turn_id = ev.turn_id
        if ev.kind == EventKind.USAGE:
            self._record_usage(ev)
        elif ev.kind == EventKind.BACKEND_NOTICE and ev.data.get("notice") == "model_rerouted":
            # RT-APP-REROUTE: the server rerouted to a different model. The notice is
            # ALREADY being yielded (observable); record the toModel so outcome() can
            # report the FINAL model that executed. Last-write-wins — if the turn is
            # rerouted more than once, the latest toModel is the final executed model.
            to_model = ev.data.get("to_model")
            if to_model:
                self._rerouted_to = str(to_model)
        elif ev.kind == EventKind.ASSISTANT_TEXT and not ev.data.get("streaming"):
            # A settled agentMessage (item/completed). Prefer phase=="final_answer";
            # never let a later non-final narrative clobber the final answer.
            text = ev.data.get("text", "")
            is_final = ev.data.get("phase") == "final_answer"
            if is_final:
                self._final_text = text
                self._final_is_final_answer = True
            elif not self._final_is_final_answer:
                self._final_text = text
        elif ev.kind in (EventKind.RUN_COMPLETED, EventKind.RUN_FAILED):
            self._terminal_status = ev.data.get("status") or (
                "completed" if ev.kind == EventKind.RUN_COMPLETED else "failed"
            )
            self._turn_error = ev.data.get("error")

    def _record_usage(self, ev: AgentEvent) -> None:
        """Record a ``thread/tokenUsage/updated`` into the backend ledger (RT-APP-USAGE).

        Keyed by (thread_id, turn_id). Within a turn this is last-write-wins (multiple
        updates settle to the turn's FINAL ``last``); a re-observed (thread_id, turn_id)
        — a trailing duplicate or a retry — overwrites the same ledger row, so the turn
        is never double-counted. The server's own running ``total`` and the
        ``modelContextWindow`` are carried on the handle (this turn's wire facts) so
        :meth:`_build_usage` can preserve them without ever summing them.

        A usage frame missing its correlation ids (thread_id / turn_id) or its ``last``
        breakdown cannot be ledgered (no dedup key / nothing to count); it is skipped
        rather than corrupting the accounting — the per-turn ``available`` flag then
        reflects only the ledger-able updates the turn actually produced.
        """
        usage = ev.data.get("usage")
        if not isinstance(usage, Mapping):
            return
        thread_id = ev.thread_id or self._thread_id
        turn_id = ev.turn_id or self._turn_id
        last = usage.get("last")
        if not thread_id or not turn_id or not isinstance(last, Mapping):
            return
        # The frame is ledger-able → this turn HAS usage. Last-write-wins per turn key.
        self._usage_seen = True
        self._usage_ledger.record(str(thread_id), str(turn_id), last)
        # Preserve the server's own cumulative total + context window (this turn's
        # latest wire view) — reported but never summed into the dedup'd accumulation.
        if isinstance(usage.get("total"), Mapping):
            self._usage_server_total = usage["total"]
        if usage.get("modelContextWindow") is not None:
            self._usage_ctx_window = usage["modelContextWindow"]

    def _build_usage(self) -> Mapping[str, Any]:
        """Settle ``outcome.usage`` for THIS turn (RT-APP-USAGE / G5).

        Returns an explicit, documented mapping rather than a raw passthrough or a
        silent ``None``:

        * **usage reported** → ``{available: True, last, total, server_total?,
          modelContextWindow?, turns_counted}`` where ``last`` is THIS turn's final
          per-turn ``last``, ``total`` is the dedup-safe accumulation (Σ each distinct
          turn's final ``last`` across the backend's lifetime), ``server_total`` is the
          server's own reported cumulative (preserved, not summed), and ``turns_counted``
          is how many distinct turns the accumulation covers.
        * **NO usage for a settled turn** → ``{available: False, reason: ...}`` — the G5
          fail-fast marker. A completed turn that could not report tokens must NOT read
          as zero cost; a budget layer keys on ``available is False`` to fail fast
          (missing-usage fail-fast). NEVER a bare ``usage=None``.

        RT-APP-REROUTE: in EITHER case, if the server rerouted this turn to a different
        model, ``rerouted: True`` + ``final_model: <toModel>`` are added (the reroute is
        independently observable as a BACKEND_NOTICE) so the run records the FINAL model
        that actually executed — the second half of model_map's real-run proof. A reroute
        is orthogonal to usage, so it is reported even on the usage-unavailable / failure
        paths (a rerouted turn still ran a model + still cost tokens).
        """
        if not self._usage_seen:
            usage: dict[str, Any] = {
                "available": False,
                "reason": (
                    "no thread/tokenUsage/updated received for this turn — the "
                    "app-server reported no token usage, so the cost is unknown "
                    "(G5: a budget layer must fail-fast, not treat this as zero cost)"
                ),
            }
            self._add_reroute_facts(usage)
            return usage
        usage = {
            "available": True,
            "total": self._usage_ledger.accumulated_total(),
            "turns_counted": self._usage_ledger.turns_counted(),
        }
        # THIS turn's per-turn final last (None only if ids were unknown — guarded above
        # via _usage_seen, but keep the lookup defensive).
        if self._thread_id and self._turn_id:
            turn_last = self._usage_ledger.turn_last(
                str(self._thread_id), str(self._turn_id))
            if turn_last is not None:
                usage["last"] = turn_last
        if self._usage_server_total is not None:
            usage["server_total"] = self._usage_server_total
        if self._usage_ctx_window is not None:
            usage["modelContextWindow"] = self._usage_ctx_window
        self._add_reroute_facts(usage)
        return usage

    def _add_reroute_facts(self, usage: dict[str, Any]) -> None:
        """Record RT-APP-REROUTE facts on the usage mapping (no-op if no reroute).

        When the server rerouted this turn (a ``model/rerouted`` BACKEND_NOTICE was
        observed), stamp ``rerouted: True`` + ``final_model`` (the rerouted-to model) so
        the run reports the model that ACTUALLY executed — never a silent swallow. With
        no reroute, nothing is added (so a caller never sees a phantom ``rerouted``).
        """
        if self._rerouted_to is not None:
            usage["rerouted"] = True
            usage["final_model"] = self._rerouted_to

    # -- outcome ------------------------------------------------------------

    async def outcome(self) -> RunOutcome:
        """Settle the run: status is always driven by ``turn.status``.

        ``structured_output`` differs by mode:

        * **tools mode** — ``None``. The contract was already satisfied by the
          ``item/tool/call`` dispatches that mutated ``ContractState`` host-side;
          core does NOT read ``structured_output`` (``host_tool_calls=True``), and
          the trailing ``agentMessage`` is narration, so it is NEVER strict-JSON
          parsed. ``final_text`` still carries that narration.
        * **structured mode** — the strict ``json.loads`` of the buffered final
          ``agentMessage`` (server-enforced schema-valid; non-JSON is a protocol
          error).

        In BOTH modes an *unexpected* server-request is a protocol error: structured
        mode registers no handler at all; tools mode handles only ``item/tool/call``,
        so any OTHER server-request still lands in
        :attr:`AppServerTransport.unexpected_server_requests`.
        """
        resume = self._resume_handle()
        final_text = self._final_text or ""
        # RT-APP-USAGE: settle usage once (the ledger is final by outcome time). Carries
        # per-turn last + dedup'd accumulated total, OR the explicit "usage unavailable"
        # G5 fail-fast marker — every RunOutcome below reports it (incl. failure paths,
        # since a failed turn still cost tokens / a budget layer reads the same signal).
        usage = self._build_usage()

        # An unexpected server-request means the wire issued a request this mode does
        # not model — structured mode models none; tools mode models only
        # item/tool/call (any OTHER method, e.g. an approval prompt, is unmodeled).
        # Either way it is a protocol error (NO GATING admits no approval/elicitation).
        unexpected = self._t.unexpected_server_requests
        if unexpected:
            methods = ", ".join(sorted({u.get("method", "?") for u in unexpected}))
            return RunOutcome(
                status="failed", resume=resume, final_text=final_text, usage=usage,
                error=BackendError(
                    code="protocol",
                    message=f"unexpected server-request(s) in {self._mode} mode: {methods}",
                    raw=unexpected),
            )

        # Turn-level failure (failed / interrupted) — never try to parse output.
        if self._terminal_status and self._terminal_status != "completed":
            if self._terminal_status == "interrupted":
                err = BackendError(code="fatal", message="turn interrupted", raw=self._turn_error)
            else:
                err = _classify_turn_error(self._turn_error)
            return RunOutcome(
                status="failed", resume=resume, final_text=final_text,
                usage=usage, error=err,
            )

        # The turn never reached a terminal (transport died mid-turn) → protocol error.
        if self._terminal_status is None:
            tail = self._t.stderr_tail(5)
            return RunOutcome(
                status="failed", resume=resume, final_text=final_text, usage=usage,
                error=BackendError(
                    code="protocol",
                    message=f"transport closed before turn/completed: {tail}"[:500]),
            )

        # Completed turn, TOOLS mode: the contract was driven entirely by the
        # item/tool/call dispatches (host-side ContractState mutation). The final
        # agentMessage is narration — return final_text but NO structured_output
        # (core ignores it under host_tool_calls=True).
        if self._mode == "tools":
            return RunOutcome(
                status="completed", resume=resume, final_text=final_text,
                usage=usage,
            )

        # Completed turn, STRUCTURED mode: the final agentMessage MUST be
        # strict-parseable JSON (outputSchema is server-enforced — non-JSON is a
        # protocol error, never a natural-language guess).
        try:
            structured = json.loads(final_text)
        except (json.JSONDecodeError, TypeError) as exc:
            return RunOutcome(
                status="failed", resume=resume, final_text=final_text, usage=usage,
                error=BackendError(
                    code="protocol",
                    message=f"final agentMessage is not valid JSON under outputSchema: {exc}",
                    raw=final_text[:500]),
            )
        return RunOutcome(
            status="completed", resume=resume, final_text=final_text,
            structured_output=structured, usage=usage,
        )

    def _resume_handle(self) -> ResumeHandle | None:
        """Mint the :class:`ResumeHandle` for this run's outcome (None if no thread_id).

        ``token`` always carries ``thread_id`` (the resident-reuse key). When the
        thread is PERSISTED (non-ephemeral), it ALSO carries the NO-GATING posture
        (``sandbox`` STRING form + ``approvalPolicy``) so a SECOND backend instance can
        RE-SUPPLY those fields on ``thread/resume`` and avoid the probe2 silent
        downgrade. Absence of the posture in the token is
        exactly how the resume path distinguishes a persisted thread (cross-process
        resumable) from an ephemeral one (resident-reuse only → fail-fast on a miss).
        """
        if not self._thread_id:
            return None
        token: dict[str, Any] = {"thread_id": self._thread_id}
        if self._persisted:
            token["sandbox"] = _SANDBOX_STRING
            token["approvalPolicy"] = _APPROVAL_POLICY
        return ResumeHandle(BACKEND_NAME, token)

    # -- cancel -------------------------------------------------------------

    #: Upper bound (seconds) the interrupt REQUEST is allowed to take before cancel
    #: gives up on it and falls through to aclose. Deliberately small — cancel must
    #: never block on a dead/slow peer, and aclose is the unconditional backstop, so
    #: the only cost of a too-short wait is "interrupt didn't land cleanly, child
    #: gets SIGKILLed instead" (still a correct cancel). This caps cancel regardless
    #: of the transport's (possibly 120s) configured request_timeout_s. A real
    #: ``turn/interrupt`` over stdio is answered as soon as app-server processes it
    #: (live: result {} arrives before turn/completed, INDEPENDENT of model speed —
    #: probe3 seq99-102), so 0.5s is ample for a healthy peer; a peer that can't ACK
    #: an interrupt in 0.5s is effectively dead and aclose's SIGKILL is the answer.
    _INTERRUPT_TIMEOUT_S = 0.5

    async def cancel(self) -> None:
        """Interrupt the in-flight turn (``turn/interrupt`` REQUEST), then close the
        transport. Correct + non-hanging in all RT-APP-CANCEL states; idempotent.

        ``turn/interrupt`` is a JSON-RPC **request** (``TurnInterruptParams`` →
        ``TurnInterruptResponse {}``, live-confirmed probe3 seq99-102): it makes the
        server transition the turn to ``status=interrupted`` rather than a bare kill.
        It is sent with a SHORT timeout (:attr:`_INTERRUPT_TIMEOUT_S`, capping the
        transport's own request_timeout_s) and the whole attempt is wrapped in
        ``suppress(Exception)`` — a timeout, a :class:`JsonRpcError` (e.g. -32600 on
        an already-completed turn), or a closed transport are ALL acceptable; the
        unconditional ``aclose`` below is the backstop.

        Three states (each correct + non-hanging):

        * **turnId unknown** — ``_turn_id`` is ``None`` (turn/start never yielded a
          ``turn.id``): there is nothing to target, so the interrupt request is
          SKIPPED and we go straight to aclose. No hang, no error.
        * **tool call pending** — an ``item/tool/call`` handler is mid-flight: the
          interrupt request + aclose do not deadlock; ``aclose`` fails pending
          request futures (so the interrupt's own future, if still waiting, unblocks)
          and cancels the in-flight handler task. No orphaned pending server-request.
        * **turn already completed** — the interrupt is a harmless no-op; even if the
          peer errors it, the error is suppressed and aclose still runs.

        Idempotent: a second cancel finds the transport already closed, so the
        interrupt request raises :class:`TransportClosed` (suppressed) and the
        idempotent ``aclose`` short-circuits — a pure no-op.
        """
        # State 1: no turnId to target → skip the interrupt request entirely (a
        # turn/interrupt with no turnId is meaningless); aclose is the only step.
        if self._thread_id and self._turn_id:
            with contextlib.suppress(Exception):
                # SHORT-timeout the request from the backend side so a slow/dead peer
                # (or the transport's large default request_timeout_s) can never make
                # cancel hang. asyncio.TimeoutError / JsonRpcError / TransportClosed
                # are all swallowed — aclose is the backstop.
                await asyncio.wait_for(
                    self._t.request(
                        "turn/interrupt",
                        {"threadId": self._thread_id, "turnId": self._turn_id},
                    ),
                    timeout=self._INTERRUPT_TIMEOUT_S,
                )
        # RT-APP-CONC: a cancelled turn is no longer active — free its per-thread slot
        # (at-most-once) so a subsequent turn on the thread is not falsely rejected.
        self._free_turn_slot()
        await self._t.aclose()


class CodexAppServerBackend:
    """Backend over ``codex app-server --stdio`` — tools OR structured channel.

    ``mode`` (default ``"tools"``) selects the contract channel for the lifetime
    of the instance (see the module docstring). ``"tools"`` is the MAIN path
    (``host_tool_calls=True`` + ``dynamicTools`` + ``item/tool/call`` dispatch to
    the host executor), aligning app-server with the Claude / CodexExec backends;
    ``"structured"`` is the M2 ``outputSchema`` path, kept as an option.

    Inject ``transport_factory`` to supply a pre-wired transport (the in-memory
    fake peer uses this); by default it constructs a real
    :class:`AppServerTransport` that spawns the child on ``start()``.

    CODEX_HOME isolation
    --------------------
    With no explicit home channel, each real transport gets its own project-local
    ephemeral ``CODEX_HOME`` under ``.tilldone/codex-home/runs/`` and the backend
    deletes it during :meth:`aclose`. ``spec.env["CODEX_HOME"]``, constructor
    ``codex_home=``, and ``TILLDONE_CODEX_HOME`` opt into a persistent/shared home;
    callers using the same persistent home concurrently should prewarm or serialize
    first use because that re-enters Codex's shared-home sqlite/skills startup path.

    Resident-transport lifecycle
    -------------------------------------------------------
    ``run_task`` runs a multi-round loop, feeding each round's ``outcome.resume``
    back into the NEXT round's ``backend.start()`` (loop.py — **0-core-diff**: core
    does not know about transports; the backend decides reuse-vs-reject). To make
    multi-round work WITHOUT leaking a child each round AND without the ephemeral
    ``thread/resume`` ``-32600`` footgun, this backend keeps the thread + transport
    **resident**: :attr:`_live` maps ``thread_id → AppServerTransport`` for every
    thread this backend instance currently owns. ``start()`` reuses the resident
    transport (a fresh ``turn/start`` on the same thread) when the resume handle
    points at one of them, and :meth:`aclose` truly closes them all (the leak fix —
    the Flow/harness layer calls ``aclose()`` in a ``finally``).

    Persisted cross-process resume (``persist_thread=True`` ruling #3)
    ----------------------------------------------------------------------------------
    With ``persist_thread=True`` (which requires an explicit persistent home on the
    real-transport path) ``thread/start`` opens a NON-ephemeral thread, so a rollout
    materializes on disk and the resume handle's ``token`` carries the NO-GATING
    posture. A SECOND backend instance fed that handle (the thread is NOT resident in
    it) opens a fresh transport and issues a native ``thread/resume`` **re-supplying**
    ``sandbox`` + ``approvalPolicy`` (omitting them silently downgrades the posture —
    the probe2 trap), asserts the resume result is STILL NO GATING (else fail-fast),
    and adds the resumed thread to its resident set. :meth:`aclose` best-effort
    ``thread/archive``'s every persisted thread before closing its transport so a
    persisted run leaves no thread leak (RT-APP-HOME-LEAK). The default
    (``persist_thread=False``) stays ephemeral + leak-safe — within-run resident
    multi-round already covers the common case.
    """

    name = BACKEND_NAME

    def __init__(
        self,
        *,
        codex_bin: str = "codex",
        codex_home: str | os.PathLike[str] | None = None,
        mode: Mode = "tools",
        persist_thread: bool = False,
        experimental_raw_events: bool = False,
        transport_factory: Callable[..., AppServerTransport] | None = None,
    ) -> None:
        if mode not in ("tools", "structured"):
            raise ValueError(f"unknown mode {mode!r}; expected 'tools' or 'structured'")
        self._codex_bin = codex_bin
        self._codex_home = (
            _normalize_codex_home(codex_home) if codex_home is not None else None
        )
        self._mode: Mode = mode
        # RT-APP-REASON: the ``experimentalRawEvents`` posture, NOT on by default. When
        # False (default) the backend's reasoning events stay RAW/DEBUG-only (tagged
        # ``raw_only`` by events.py → a public sink drops them from the default feed,
        # keeps their raw in the debug layer). When True an opted-in caller promotes
        # reasoning to the public feed (the RunHandle re-stamps ``raw_only=False``).
        self._experimental_raw_events = experimental_raw_events
        # persist_thread (default False — ephemeral, leak-safe). When True, thread/start
        # opens a NON-ephemeral thread so a rollout materializes on disk and the thread
        # is resumable cross-process via thread/resume ( — the
        # M4 same-backend cross-process path). Within-run resident multi-round already
        # covers the common case WITHOUT persistence (so the default stays ephemeral),
        # and a persisted thread leaves a rollout/daemon thread that aclose() must reap
        # (RT-APP-HOME-LEAK), so persistence is strictly opt-in.
        self._persist_thread = persist_thread
        self._transport_factory = transport_factory
        # thread_id -> resident transport. The backend OWNS these (the per-turn
        # RunHandle does not close them); aclose() closes + clears them all.
        self._live: dict[str, AppServerTransport] = {}
        # thread_ids that were started NON-ephemeral (persist mode) and therefore have a
        # persisted rollout. aclose() best-effort thread/archive's exactly these before
        # closing the transport so a persisted run does not leave a thread leak.
        self._persisted: set[str] = set()
        # Backend-owned default homes. Explicit homes are never registered here, so
        # ``aclose`` can delete only per-run ephemeral homes it created itself.
        self._owned_codex_homes: set[str] = set()
        self._transport_codex_homes: dict[AppServerTransport, str] = {}
        # RT-APP-USAGE: one usage ledger PER BACKEND INSTANCE, shared across every
        # RunHandle this backend mints. Keyed by (threadId,turnId), it accumulates each
        # turn's final ``last`` across the resident multi-round / resume lifetime and
        # dedups a re-observed turn — so usage is correct across rounds and never
        # double-counted (the per-turn handle alone could not see prior turns).
        self._usage = _UsageLedger()
        # RT-APP-CONC: per-thread active-turn registry + a backend lock guarding the
        # check-and-set. ``_active_turns`` holds the thread_ids that currently have an
        # IN-FLIGHT turn on this backend instance. app-server permits ONE active turn per
        # thread, so a SECOND turn/start on a thread whose turn has not yet terminated is
        # REJECTED (fail-fast) rather than silently interleaved on the shared resident
        # transport. The lock makes the "is this thread busy? if not, claim it" decision
        # atomic so two concurrent start()s racing the SAME thread cannot both win. The
        # slot is released when the owning RunHandle's event stream reaches its terminal
        # (turn/completed) or the handle is cancelled — see ``_TurnSlot`` / the
        # ``on_turn_done`` callback handed to each RunHandle. Routing for DIFFERENT
        # threads needs no lock: each thread lives on its OWN transport whose
        # ``item/tool/call`` handler closes over THAT run's executor, so a tool call is
        # routed by (threadId,turnId,callId) to the owning run by construction.
        self._active_turns: set[str] = set()
        self._turn_lock = asyncio.Lock()

    def capabilities(self) -> BackendCapabilities:
        # Two coherent, MUTUALLY-EXCLUSIVE capability surfaces; core branches only
        # on host_tool_calls. The shared event/resume/multimodal
        # flags are identical — only the contract channel differs.
        if self._mode == "tools":
            return BackendCapabilities(
                host_tool_calls=True,
                structured_final_output=False,
                dynamic_tools=True,
                session_resume=True,
                streaming_events=True,
                reasoning_events=True,
                command_events=True,
                file_change_events=True,
                multimodal_input=True,
            )
        return BackendCapabilities(
            host_tool_calls=False,
            structured_final_output=True,
            dynamic_tools=False,
            session_resume=True,
            streaming_events=True,
            reasoning_events=True,
            command_events=True,
            file_change_events=True,
            multimodal_input=True,
        )

    def probe_capabilities(self) -> dict[str, Any]:
        """RT-CAP-1 OFFLINE gate (layer 1) — akin to ``CodexExec.probe_host_tools``.

        Reports, WITHOUT spawning anything, whether the methods this mode drives are in
        the known app-server request method-set (``offline_ok``) and whether this mode
        needs the experimentalApi-gated ``dynamicTools`` (``needs_dynamic_tools``).

        Deliberately HONEST about the gate's limit: ``experimental_api_observable_offline``
        is always ``False`` because the ``initialize`` RESULT echoes NO capabilities
        (InitializeResponse schema = {codexHome, platformFamily, platformOs, userAgent}).
        experimentalApi acceptance is therefore a RUNTIME/canary signal — observable only
        as the ``thread/start`` ``-32600`` rejection, which :meth:`_open_thread` turns
        into a clear fail-fast. This method is the cheap pre-flight; the load-bearing
        behavior lives in ``start()``.
        """
        required = _MODE_REQUIRED_METHODS.get(self._mode, ())
        # G3 schema-drift (M7 Lane D): the offline gate is EXTENDED to validate that every
        # method the backend drives is present in the reference protocol schema — so a
        # codex upgrade that renamed/removed a method is caught here (deterministic, no
        # spawn), not at a confusing runtime -32601. Imported locally to avoid the import
        # cycle (schema_drift imports this module's constants). A drift RAISES rather than
        # returns False: a protocol the backend can no longer assume is a fail-fast, not a
        # soft "offline gate failed" the caller might paper over.
        from tilldone.backends.app_server.schema_drift import check_schema_drift

        drift = check_schema_drift()
        return {
            "mode": self._mode,
            "needs_dynamic_tools": self._mode == "tools",
            "offline_ok": all(m in _KNOWN_CLIENT_METHODS for m in required),
            "required_methods": list(required),
            # The experimentalApi handshake cannot be confirmed offline (no capability
            # echo) — it is a live-canary signal (the thread/start -32600). Stated so a
            # caller never mistakes a green offline gate for a negotiated experimentalApi.
            "experimental_api_observable_offline": False,
            # G3: the schema-drift verdict + the method-sets verified against the schema.
            "schema_drift_ok": drift["ok"],
            "schema_verified_client_methods": drift["client_present"],
            "schema_verified_server_methods": drift["server_present"],
        }

    def _make_transport(self, spec: AgentRunSpec) -> AppServerTransport:
        if self._transport_factory is not None:
            return self._transport_factory()
        timeout = spec.timeout_s if spec.timeout_s is not None else 120.0
        env = dict(spec.env) if spec.env else {}
        codex_home, owned = self._resolve_codex_home_with_ownership(
            spec, explicit=env.get("CODEX_HOME"))
        env["CODEX_HOME"] = codex_home
        _ensure_codex_home(codex_home)
        if owned:
            _mark_ephemeral_codex_home(codex_home)
            self._owned_codex_homes.add(codex_home)
        transport = AppServerTransport(
            codex_bin=self._codex_bin,
            env=env,
            cwd=str(spec.cwd),
            request_timeout_s=timeout,
        )
        if owned:
            self._transport_codex_homes[transport] = codex_home
        return transport

    def _resolve_codex_home(self, spec: AgentRunSpec, *, explicit: str | None = None) -> str:
        """Resolve child CODEX_HOME without inheriting ambient process CODEX_HOME.

        Precedence is:
        ``spec.env["CODEX_HOME"]`` > constructor ``codex_home=`` >
        ``TILLDONE_CODEX_HOME`` > project-local per-run ephemeral home.
        A process-level ambient ``CODEX_HOME`` is intentionally ignored unless the
        caller passes it through one of the explicit channels above.
        """

        return self._resolve_codex_home_with_ownership(spec, explicit=explicit)[0]

    def _resolve_codex_home_with_ownership(
        self,
        spec: AgentRunSpec,
        *,
        explicit: str | None = None,
    ) -> tuple[str, bool]:
        """Return ``(CODEX_HOME, backend_owns_home)`` for the child process."""

        if explicit:
            return _normalize_codex_home(explicit), False
        if self._codex_home is not None:
            return self._codex_home, False
        tilldone_env = os.environ.get(_TILLDONE_CODEX_HOME_ENV)
        if tilldone_env:
            return _normalize_codex_home(tilldone_env), False
        if self._requires_persistent_codex_home(spec):
            raise ValueError(
                "codex_app_server persist/resume requires an explicit persistent "
                "CODEX_HOME via spec.env['CODEX_HOME'], codex_home=, or "
                "TILLDONE_CODEX_HOME; the default CODEX_HOME is per-run ephemeral "
                "and is cleaned on backend.aclose()")
        return _default_codex_home(spec.cwd), True

    def _requires_persistent_codex_home(self, spec: AgentRunSpec) -> bool:
        if self._persist_thread:
            return True
        if spec.resume is None or spec.resume.backend != self.name:
            return False
        token = spec.resume.token
        return "sandbox" in token and "approvalPolicy" in token

    async def start(self, spec: AgentRunSpec, executor: ToolExecutor) -> CodexAppServerRunHandle:
        # RT-SESS-5 mirror: a resume handle minted by another backend must NOT be
        # used as an opaque string (cross-backend resume is a non-goal). Fail fast.
        if spec.resume is not None and spec.resume.backend != self.name:
            raise ValueError(
                f"resume handle is for backend {spec.resume.backend!r}, not {self.name!r} — "
                "cross-backend resume is a non-goal (RT-SESS-5)")

        # BRANCH B — resume present + backend matches. Resident-reuse comes FIRST
        #: if the thread_id is one this instance already
        # owns, the multi-round path issues another turn/start on the SAME thread +
        # connection — NO new spawn, NO native thread/resume. On a resident MISS the
        # token decides: a PERSISTED thread (posture in token) does a real cross-process
        # thread/resume (ruling #3); an ephemeral one fails fast (no rollout, M2).
        if spec.resume is not None:
            return await self._resume(spec, executor)

        # BRANCH A — no resume: build + start a fresh transport, handshake,
        # thread/start (ephemeral unless persist_thread), make it RESIDENT, turn/start.
        return await self._start_fresh(spec, executor)

    def _wire_tool_handler(self, transport: AppServerTransport,
                           executor: ToolExecutor) -> None:
        """TOOLS mode only: register the ``item/tool/call`` server-request handler.

        Registered BEFORE ``turn/start`` so a tool call arriving mid-turn is handled
        (the transport runs handlers concurrently with an in-flight ``request()`` —
        reentrancy-safe). Each call is parsed to a :class:`ToolCall`, dispatched to
        the host ``executor`` (the core ToolRegistry), and answered with a
        ``DynamicToolCallResponse``. A handler exception would surface as a JSON-RPC
        -32603 (transport backstop); we instead translate executor errors to
        ``success:false`` so the model gets the contract-level failure text.
        """
        if self._mode != "tools":
            return

        async def _on_tool_call(req: Mapping[str, Any]) -> dict[str, Any]:
            params = req.get("params") or {}
            call = _tool_call_from_params(params)
            result = await executor.invoke(call)
            return _dynamic_tool_response(result.content, success=not result.is_error)

        transport.on_server_request(_TOOL_CALL_METHOD, _on_tool_call)

    async def _start_fresh(self, spec: AgentRunSpec,
                           executor: ToolExecutor) -> CodexAppServerRunHandle:
        """No-resume path: spawn a new transport, handshake + thread/start, cache it
        as resident (the backend owns its lifecycle), then issue the first turn.

        TOOLS mode wires the ``item/tool/call`` handler BEFORE the turn; STRUCTURED
        mode registers NO handler — under NO GATING it expects zero server→client
        requests, so any that arrive land in ``transport.unexpected_server_requests``
        and are a protocol error at outcome time.
        """
        transport = self._make_transport(spec)
        self._wire_tool_handler(transport, executor)
        try:
            if self._transport_factory is None:
                # We built the transport → we own spawning the child + pumps. An
                # injected factory (tests/fake-peer) returns an already-started
                # transport, so we must NOT re-spawn it.
                await transport.start()
            await self._handshake(transport, spec)
            thread_id = await self._open_thread(transport, spec)
        except Exception:
            # Never leak the child/connection on a handshake failure.
            await transport.aclose()
            self._cleanup_transport_codex_home(transport)
            raise
        # The thread is now open; the backend OWNS the transport from here on (the
        # RunHandle does NOT close it). Register it as resident BEFORE the turn so a
        # turn/start failure still leaves it tracked for aclose() to reap.
        self._live[thread_id] = transport
        # Persist mode: this thread has a rollout on disk → remember it so aclose()
        # archives it (RT-APP-HOME-LEAK) and so a cross-process resume token can carry
        # the posture needed to re-supply policy on thread/resume.
        if self._persist_thread:
            self._persisted.add(thread_id)
        try:
            turn_id = await self._start_turn(transport, spec, thread_id)
        except Exception:
            await self._evict_and_close(thread_id)
            raise
        return CodexAppServerRunHandle(
            transport, thread_id=thread_id, turn_id=turn_id, mode=self._mode,
            persisted=self._persist_thread, usage_ledger=self._usage,
            raw_events=self._experimental_raw_events,
            on_turn_done=self._release_turn_slot)

    async def _resume(self, spec: AgentRunSpec,
                      executor: ToolExecutor) -> CodexAppServerRunHandle:
        """Resume dispatch (resident-reuse FIRST, then persisted cross-process resume).

        1. **Resident & alive** → REUSE the connection: just another ``turn/start`` on
           the existing thread (the within-run multi-round path ruling
           #1) — no spawn, no handshake, no ``thread/resume``. This is the common case
           and is UNCHANGED.
        2. **Not resident, token carries posture (PERSISTED thread)** → a real
           cross-process resume (ruling #3): open a FRESH transport, handshake,
           ``thread/resume`` RE-SUPPLYING ``sandbox`` + ``approvalPolicy`` (the probe2
           downgrade fix), assert NO GATING from the resume result top-level, make the
           resumed thread resident, then ``turn/start``.
        3. **Not resident, no posture (EPHEMERAL thread)** → the M2 fail-fast: an
           ``ephemeral:true`` thread has no rollout, so ``thread/resume`` would
           ``-32600 'no rollout found'``.
        """
        assert spec.resume is not None  # guarded by start()
        token = spec.resume.token
        thread_id = token.get("thread_id")
        if not thread_id:
            raise ValueError("resume handle for codex_app_server missing thread_id token")
        thread_id = str(thread_id)

        transport = self._live.get(thread_id)
        # transport._closed is the transport's own idempotent shutdown flag (set by
        # aclose). We read it directly rather than add a public accessor, to keep
        # the 0-core-diff boundary (transport.py is unchanged): a resident entry
        # whose transport was hard-closed (e.g. via a prior handle.cancel()) is a
        # miss, not a reusable connection.
        if transport is not None and not transport._closed:
            # (1) Resident & alive → reuse.
            #
            # RT-APP-CONC: claim the thread's single active-turn slot BEFORE touching the
            # shared transport. This is the load-bearing ordering: if a CONCURRENT start()
            # is already running a turn on this resident thread, the slot is taken and we
            # fail fast HERE — before re-wiring the shared item/tool/call handler — so the
            # loser never clobbers the in-flight winner's executor binding (which would
            # cross-wire the winner's tool calls to the loser's executor). Acquiring first
            # makes "reject the second same-thread turn" and "never corrupt the active
            # turn's routing" the same guarantee.
            await self._acquire_turn_slot(thread_id)
            try:
                # Re-wire the tool handler (idempotent overwrite) so THIS round's executor
                # handles the turn's tool calls — in practice run_task passes the same
                # registry, but rebinding keeps the closure honest if a caller supplies a
                # fresh one. Safe now: the slot guarantees no other turn is active here.
                self._wire_tool_handler(transport, executor)
                turn_id = await self._start_turn(
                    transport, spec, thread_id, acquire_slot=False)
            except BaseException:
                # We acquired the slot above; release it on any failure before the handle
                # (which would own the release) exists, so a failed reuse never leaks it.
                self._release_turn_slot(thread_id)
                raise
            return CodexAppServerRunHandle(
                transport, thread_id=thread_id, turn_id=turn_id, mode=self._mode,
                persisted=thread_id in self._persisted, usage_ledger=self._usage,
                raw_events=self._experimental_raw_events,
                on_turn_done=self._release_turn_slot)

        # Evict a stale (closed) entry so the registry never lies.
        self._live.pop(thread_id, None)

        # (2) PERSISTED thread (the token carries the posture we must re-supply) → a
        # real cross-process resume. Absence of the posture ⇒ ephemeral ⇒ fail-fast.
        if "sandbox" in token and "approvalPolicy" in token:
            return await self._resume_persisted(spec, executor, thread_id, token)

        # (3) Ephemeral / no posture → fail-fast (unchanged M2 behavior).
        raise RuntimeError(
            f"app_server thread {thread_id!r} is not resident in this backend "
            "instance; cross-run_task resume requires persist_thread=True (an "
            "ephemeral thread has no rollout — native thread/resume would -32600 "
            "'no rollout found')")

    async def _resume_persisted(self, spec: AgentRunSpec, executor: ToolExecutor,
                                thread_id: str,
                                token: Mapping[str, Any]) -> CodexAppServerRunHandle:
        """Cross-process resume of a PERSISTED thread on a FRESH transport.

        Opens + handshakes a new transport, then ``thread/resume`` RE-SUPPLYING the
        ``sandbox`` + ``approvalPolicy`` from the resume token AND the absolute
        ``runtimeWorkspaceRoots`` from ``spec.context_dirs`` (so an @ref'd dir survives
        the restart as an explicit workspace root). The resume result's TOP-LEVEL
        posture is then asserted to STILL be NO GATING — a downgrade/mismatch (the
        probe2 silent-downgrade trap) is a fail-fast, never a silently-gated run. On
        success the resumed thread becomes resident + persisted (so subsequent rounds
        reuse it and aclose() archives it), then the turn is issued.
        """
        transport = self._make_transport(spec)
        self._wire_tool_handler(transport, executor)
        try:
            if self._transport_factory is None:
                await transport.start()
            await self._handshake(transport, spec)
            await self._native_resume(transport, spec, thread_id, token)
        except Exception:
            # Never leak the child/connection on a resume/handshake failure.
            await transport.aclose()
            self._cleanup_transport_codex_home(transport)
            raise
        # Resumed thread is now live on this connection; the backend OWNS it. Register
        # it resident + persisted BEFORE the turn so a turn/start failure still leaves
        # it tracked for aclose() (which archives persisted threads).
        self._live[thread_id] = transport
        self._persisted.add(thread_id)
        try:
            turn_id = await self._start_turn(transport, spec, thread_id)
        except Exception:
            await self._evict_and_close(thread_id)
            raise
        return CodexAppServerRunHandle(
            transport, thread_id=thread_id, turn_id=turn_id, mode=self._mode,
            persisted=True, usage_ledger=self._usage,
            raw_events=self._experimental_raw_events,
            on_turn_done=self._release_turn_slot)

    async def _native_resume(self, t: AppServerTransport, spec: AgentRunSpec,
                             thread_id: str, token: Mapping[str, Any]) -> None:
        """Issue ``thread/resume`` (RE-SUPPLYING policy) + assert NO GATING preserved.

        The resume params carry ``threadId`` + the ``sandbox`` (STRING form) and
        ``approvalPolicy`` lifted from the resume TOKEN (re-supplying the posture is
        what keeps NO GATING across a process restart — probe verdict #3; omitting it
        is the probe2 silent downgrade). ``runtimeWorkspaceRoots`` (absolute) is
        re-supplied from ``spec.context_dirs`` per schema. The resume RESULT reports
        the live posture at its TOP LEVEL (``approvalPolicy`` string + ``sandbox``
        object ``{type}``); if either is not the NO-GATING value we FAIL FAST rather
        than run under a silently-downgraded (gated) posture.
        """
        params: dict[str, Any] = {
            "threadId": thread_id,
            "sandbox": str(token.get("sandbox") or _SANDBOX_STRING),
            "approvalPolicy": str(token.get("approvalPolicy") or _APPROVAL_POLICY),
        }
        roots = _runtime_workspace_roots(spec)
        if roots:
            params["runtimeWorkspaceRoots"] = roots
        result = await t.request("thread/resume", params)
        result = result or {}
        # Top-level posture (NOT under `thread`) — the probe's verdict #3 source.
        approval = result.get("approvalPolicy")
        sandbox = result.get("sandbox")
        sandbox_type = sandbox.get("type") if isinstance(sandbox, Mapping) else sandbox
        if approval != _APPROVAL_POLICY or sandbox_type != _SANDBOX_OBJECT_TYPE:
            # Silent-downgrade trap: the resume reported a gated posture
            # despite the re-supplied policy. Refuse to run gated — fail fast as an
            # EXCEPTION (the BackendError value type is reserved for RunOutcome.error;
            # this breach happens during start(), before any RunHandle exists, so the
            # caller gets a clear raise — same precedent as _open_thread's no-thread.id).
            raise RuntimeError(
                "codex_app_server thread/resume downgraded the NO-GATING posture despite "
                f"re-supplied policy: approvalPolicy={approval!r} (want {_APPROVAL_POLICY!r}), "
                f"sandbox={sandbox_type!r} (want {_SANDBOX_OBJECT_TYPE!r})")

    async def _handshake(self, t: AppServerTransport, spec: AgentRunSpec) -> None:
        await t.request(
            "initialize",
            {
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
                "capabilities": {"experimentalApi": True},
            },
        )
        await t.notify("initialized", {})

    async def _open_thread(self, t: AppServerTransport, spec: AgentRunSpec) -> str:
        # Only the no-resume (_start_fresh) path reaches here: it always opens a NEW
        # thread via thread/start. The resume path either reuses a resident thread
        # (another turn/start, NO thread/resume) or — in persist mode — does a fresh
        # transport + native thread/resume.
        params: dict[str, Any] = {
            "cwd": str(spec.cwd),
            "sandbox": _SANDBOX_STRING,  # STRING form on thread/start
            "approvalPolicy": _APPROVAL_POLICY,
            # ephemeral:False in persist mode → a rollout materializes on disk so the
            # thread is resumable cross-process. Default
            # True keeps the common path leak-safe (no rollout to reap).
            "ephemeral": not self._persist_thread,
        }
        # context_dirs → runtimeWorkspaceRoots (absolute paths, schema-required) so the
        # @ref'd dirs are explicit workspace roots, not only reachable via the
        # full-access sandbox — parity with Claude (add_dirs) / CodexExec (--add-dir).
        roots = _runtime_workspace_roots(spec)
        if roots:
            params["runtimeWorkspaceRoots"] = roots
        # RT-APP / model_map: send the EXPLICITLY-RESOLVED model (a caller-supplied
        # model_map alias → its concrete model_id; a non-alias passes through). The
        # resolved concrete model is what reaches the wire — never the bare alias — so
        # the run's real-run proof asserts the model that was actually requested. This
        # is single-backend explicit resolution, NOT the cross-backend tier equivalence
        # codex_exec fails fast on (RT-CFG-1) — see _resolve_model.
        resolved_model = _resolve_model(spec)
        if resolved_model:
            params["model"] = resolved_model
        if spec.system_instructions:
            params["baseInstructions"] = spec.system_instructions
        if spec.developer_instructions:
            params["developerInstructions"] = spec.developer_instructions
        # TOOLS mode: declare the contract (return/todo/finish) + custom tools that
        # core already placed in spec.tools as dynamicTools (REQUIRES experimentalApi,
        # sent on initialize — ). STRUCTURED mode wires none.
        sent_dynamic_tools = self._mode == "tools" and bool(spec.tools)
        if sent_dynamic_tools:
            params["dynamicTools"] = [_dynamic_tool_entry(td) for td in spec.tools]
        # RT-CAP-1 live canary: this thread/start (carrying dynamicTools) is the moment
        # the experimentalApi negotiation is actually exercised. If it was NOT negotiated
        # the server rejects with -32600 "requires experimentalApi capability" (findings
        # We MUST surface that as a CLEAR capability error — naming experimentalApi
        # — rather than leak the opaque -32600 OR (worse) retry tool-less: the contract
        # tools are unavailable, so a tool-less run could never report the contract and
        # would spin to max_rounds. NO text-protocol fallback. The
        # gate is scoped to a request that actually CARRIED dynamicTools, so an unrelated
        # -32600 (or a structured-mode thread/start) is never misattributed.
        try:
            result = await t.request("thread/start", params)
        except JsonRpcError as exc:
            if sent_dynamic_tools and _is_experimental_api_rejection(exc):
                raise RuntimeError(_CAPABILITY_FAILFAST_MSG) from exc
            raise
        thread = (result or {}).get("thread") or {}
        thread_id = thread.get("id")
        if not thread_id:
            # Handshake-time protocol breach → fail-fast as an exception (the
            # BackendError value type is reserved for RunOutcome.error). start()'s
            # caller gets a clear error, not a malformed handle.
            raise RuntimeError(
                f"codex_app_server thread/start returned no thread.id: {result!r}"[:500])
        return str(thread_id)

    async def _acquire_turn_slot(self, thread_id: str) -> None:
        """RT-APP-CONC: claim the single active-turn slot for ``thread_id`` or fail fast.

        app-server allows ONE active turn per thread. Under :attr:`_turn_lock` (so the
        check-and-set is atomic against a concurrent ``start()`` racing the same thread)
        this REJECTS a second turn on a thread whose turn is still in-flight, rather than
        letting two ``turn/start``s silently interleave on the shared resident transport.
        The slot is released by :meth:`_release_turn_slot` when the owning RunHandle's
        stream terminates (or it is cancelled), or here-adjacent on a ``turn/start``
        failure (so a failed start never leaks a permanent busy marker).
        """
        async with self._turn_lock:
            if thread_id in self._active_turns:
                raise RuntimeError(
                    f"app_server thread {thread_id!r} already has an active turn — a "
                    "second concurrent turn on the same thread is rejected (app-server "
                    "permits one active turn per thread; RT-APP-CONC). Wait for the "
                    "in-flight turn to complete before starting another.")
            self._active_turns.add(thread_id)

    def _release_turn_slot(self, thread_id: str) -> None:
        """RT-APP-CONC: release ``thread_id``'s active-turn slot (idempotent).

        Called when the owning RunHandle's event stream reaches its terminal / ends /
        is cancelled, freeing the thread for its NEXT turn (the sequential multi-round
        path: round N's terminal is consumed before round N+1's ``start``). Discarding
        (not removing) keeps a double-release a harmless no-op.
        """
        self._active_turns.discard(thread_id)

    async def _start_turn(self, t: AppServerTransport, spec: AgentRunSpec,
                          thread_id: str, *, acquire_slot: bool = True) -> str | None:
        # RT-APP-CONC: claim the thread's single active-turn slot BEFORE issuing
        # turn/start (a second concurrent turn on this thread fails fast in
        # _acquire_turn_slot). ``acquire_slot=False`` means the CALLER already acquired it
        # (the resident-reuse path, which must acquire before re-wiring the shared
        # handler) and owns its release-on-failure; in that case _start_turn must NOT
        # release on its own failure (the caller's except does). When _start_turn
        # acquires, it releases on its OWN turn/start failure so a failed start leaves no
        # permanent busy marker. On success the slot is held and the RunHandle owns the
        # release (via the on_turn_done callback).
        if acquire_slot:
            await self._acquire_turn_slot(thread_id)
        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": _prompt_text(spec.prompt)}],
            "sandboxPolicy": {"type": "dangerFullAccess"},  # OBJECT form on turn/start
        }
        if spec.output_schema is not None:
            params["outputSchema"] = dict(spec.output_schema)
        try:
            result = await t.request("turn/start", params)
        except BaseException:
            if acquire_slot:
                self._release_turn_slot(thread_id)
            raise
        # turn/start resp → result.turn.id. Captured eagerly so cancel()
        # can issue a real turn/interrupt even before any item notification arrives.
        turn = (result or {}).get("turn") or {}
        tid = turn.get("id")
        return str(tid) if tid else None

    async def _evict_and_close(self, thread_id: str) -> None:
        """Remove a thread from the resident registry and close its transport.

        If the thread was persisted (non-ephemeral), best-effort ``thread/archive`` it
        first so a turn/start failure on a persisted thread does not leak the rollout —
        same RT-APP-HOME-LEAK cleanup aclose() does, applied to the single evicted
        thread. Then drop it from the persisted set.
        """
        transport = self._live.pop(thread_id, None)
        was_persisted = thread_id in self._persisted
        self._persisted.discard(thread_id)
        if transport is not None:
            if was_persisted:
                await self._archive_quietly(transport, thread_id)
            await transport.aclose()
            self._cleanup_transport_codex_home(transport)

    #: Upper bound (seconds) a best-effort ``thread/archive`` during aclose is allowed
    #: to take. aclose MUST stay robust + non-hanging, so the archive is short-timeouted
    #: and fully error-suppressed — a slow/dead peer just falls through to the
    #: unconditional transport close (which SIGKILLs the child anyway).
    _ARCHIVE_TIMEOUT_S = 2.0

    async def aclose(self) -> None:
        """Close every RESIDENT transport, then clear the registry. Idempotent.

        This is the leak fix: the per-turn RunHandle no longer owns
        the transport — the backend keeps it resident across a ``run_task`` loop, so
        the persistent ``codex app-server`` child would survive a successful run
        unless something closes it. The Flow/harness layer calls ``aclose()`` in a
        ``finally`` exactly to reap it. ``AppServerTransport.aclose`` is itself
        idempotent, so a double ``aclose()`` (or one over an empty registry) is a
        harmless no-op.

        RT-APP-HOME-LEAK: a PERSISTED (non-ephemeral) thread leaves a rollout + a
        live daemon thread on the app-server. Before closing such a thread's transport
        we best-effort ``thread/archive`` it (suppressed + short-timeouted) so a
        persisted run does not leave a thread leak. An ephemeral thread has nothing to
        archive, so it is closed directly.
        """
        # Snapshot + clear first so re-entrancy / concurrent start() can't observe a
        # half-closed registry, then archive (persisted only) + close each transport.
        live = dict(self._live)
        persisted = set(self._persisted)
        self._live.clear()
        self._persisted.clear()
        for thread_id, transport in live.items():
            if thread_id in persisted:
                await self._archive_quietly(transport, thread_id)
            await transport.aclose()
            self._cleanup_transport_codex_home(transport)
        self._cleanup_all_owned_codex_homes()

    def _cleanup_transport_codex_home(self, transport: AppServerTransport) -> None:
        home = self._transport_codex_homes.pop(transport, None)
        if home is not None:
            self._cleanup_owned_codex_home(home)

    def _cleanup_all_owned_codex_homes(self) -> None:
        for home in list(self._owned_codex_homes):
            self._cleanup_owned_codex_home(home)

    def _cleanup_owned_codex_home(self, home: str) -> None:
        self._owned_codex_homes.discard(home)
        for transport, transport_home in list(self._transport_codex_homes.items()):
            if transport_home == home:
                self._transport_codex_homes.pop(transport, None)
        shutil.rmtree(home, ignore_errors=True)
        with contextlib.suppress(OSError):
            Path(home).parent.rmdir()

    async def _archive_quietly(self, transport: AppServerTransport,
                               thread_id: str) -> None:
        """Best-effort ``thread/archive`` of a persisted thread (errors suppressed).

        ``ThreadArchiveParams`` is just ``{threadId}``. The whole attempt is
        short-timeouted and wrapped in ``suppress(Exception)`` — a timeout, a
        ``JsonRpcError`` (e.g. archive unsupported on this build / already archived),
        or a closed transport are ALL acceptable; the unconditional ``aclose`` that
        follows is the backstop. aclose must never raise.
        """
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                transport.request("thread/archive", {"threadId": thread_id}),
                timeout=self._ARCHIVE_TIMEOUT_S,
            )

    # ====================================================================== #
    # Rich control: a capability-scoped CONTROL PLANE (steer / fork /         #
    # rollback). Reachable ONLY via the SteerCapable / BranchingCapable       #
    # optional protocols (core/backend.py) — NEVER from run_task. These       #
    # methods operate on a RESIDENT thread's transport (the same ``_live``    #
    # registry the multi-round contract loop reuses) but are an INDEPENDENT   #
    # surface: they neither read nor mutate ContractState, drive no           #
    # RunHandle, and emit no AgentEvent. run_task stays completely ignorant   #
    # of them. #
    # ====================================================================== #

    def _resident_transport(self, thread_id: str) -> AppServerTransport:
        """Return the live transport for a RESIDENT ``thread_id`` or fail fast.

        The control plane acts on a thread this backend instance currently owns (its
        ``_live`` registry — the same residency the multi-round loop reuses). A thread
        that is not resident (never started here, or already closed) has no transport to
        drive, so this raises rather than silently no-op'ing. We read the transport's own
        idempotent ``_closed`` flag (kept private — transport.py is unchanged) so a hard-
        closed entry is treated as a miss, exactly as :meth:`_resume` does.
        """
        transport = self._live.get(thread_id)
        if transport is None or transport._closed:
            raise RuntimeError(
                f"app_server thread {thread_id!r} is not resident in this backend "
                "instance — rich control (steer/fork/rollback) operates only on a live "
                "resident thread this backend owns")
        return transport

    async def steer(self, thread_id: str, input: Any, *, expected_turn_id: str) -> str:
        """Steer the in-flight turn on ``thread_id`` (``turn/steer``); return the new turn id.

        Implements :class:`~tilldone.core.backend.SteerCapable`. ``expected_turn_id`` is
        the REQUIRED active-turn precondition (``TurnSteerParams.expectedTurnId``; source =
        the turn id from ``turn/start``). The wire request is
        ``turn/steer {threadId, expectedTurnId, input}`` and the response is
        ``TurnSteerResponse {turnId}`` (the new active turn id), which is returned.

        Race fail-fast (the three race adversaries): a STALE expectedTurnId,
        steering an ALREADY-COMPLETED turn, or a CONSECUTIVE steer that re-uses the now-
        rotated turn id all cause the server to reject the request (the
        ``activeTurnNotSteerable`` codexErrorInfo, surfaced as a :class:`JsonRpcError`). We
        translate that into a CLEAR ``RuntimeError`` that names the steer race rather than
        leaking an opaque code OR silently dropping the input on a dead turn. After a
        successful steer the caller MUST adopt the returned turn id as the new
        ``expected_turn_id`` for any subsequent steer.

        Out-of-band relative to the contract loop: this does NOT touch any RunHandle's
        event stream — the steered input becomes part of the turn the resident transport is
        already driving, and run_task remains unaware steer exists.
        """
        transport = self._resident_transport(thread_id)
        params = {
            "threadId": thread_id,
            "expectedTurnId": expected_turn_id,
            "input": [{"type": "text", "text": _prompt_text(input)}],
        }
        try:
            result = await transport.request(_STEER_METHOD, params)
        except JsonRpcError as exc:
            blob = f"{exc.code} {exc.message} {exc.data}".lower()
            detail = "active turn is not steerable" if _STEER_RACE_MARKER in blob else \
                str(exc.message or exc.code)
            raise RuntimeError(
                f"turn/steer rejected for thread {thread_id!r} "
                f"(expectedTurnId={expected_turn_id!r}): {detail} — the expected turn is "
                "no longer the steerable active turn (stale id / already-completed / "
                "consecutive steer); re-read the active turn id and retry") from exc
        turn = result or {}
        new_turn = turn.get("turnId")
        if not new_turn:
            raise RuntimeError(
                f"turn/steer returned no turnId for thread {thread_id!r}: {result!r}"[:500])
        return str(new_turn)

    async def fork(self, thread_id: str) -> str:
        """Fork ``thread_id`` into a NEW thread (``thread/fork``); return the new thread id.

        Implements :class:`~tilldone.core.backend.BranchingCapable`. The wire request is
        ``thread/fork {threadId}`` (prefer thread_id over path per the schema) and the
        response is a ``ThreadForkResponse`` whose ``thread.id`` is the NEW forked thread
        (``forkedFromId`` points back at the source). That new id is returned; the forked
        thread is NOT made resident here (forking is a control-plane branch operation, not a
        run — a caller that wants to RUN the fork starts a turn on it via the normal path).
        """
        transport = self._resident_transport(thread_id)
        result = await transport.request(_FORK_METHOD, {"threadId": thread_id})
        thread = (result or {}).get("thread") or {}
        new_id = thread.get("id")
        if not new_id:
            raise RuntimeError(
                f"thread/fork returned no thread.id for source {thread_id!r}: "
                f"{result!r}"[:500])
        return str(new_id)

    async def rollback(self, thread_id: str, *, num_turns: int) -> str:
        """Roll back the last ``num_turns`` turns of ``thread_id``'s HISTORY
        (``thread/rollback``); return the thread id.

        Implements :class:`~tilldone.core.backend.BranchingCapable`.

        **Rolls back THREAD HISTORY ONLY — it does NOT roll back the filesystem.** Per the
        ``ThreadRollbackParams.numTurns`` schema: "This only modifies the thread's history
        and does not revert local file changes that have been made by the agent. Clients
        are responsible for reverting these changes." Accordingly this method performs ZERO
        filesystem operations — a turn's side-effect files survive a rollback untouched
        (the boundary is asserted by
        ``test_rollback_touches_no_filesystem`` and the 0-core-diff
        ``test_rollback_is_history_only_keeps_files``). File-level rollback, if ever wanted,
        is a SEPARATE artifact-snapshot design and must never be conflated with this.

        ``num_turns`` MUST be ``>= 1``. The schema's ``minimum`` is 0 but its description
        REQUIRES ``>= 1`` ("Must be >= 1"), so the client enforces it: ``num_turns < 1``
        raises :class:`ValueError` BEFORE any wire call — this defeats the ``numTurns=0``
        no-op false-green (a 0-turn rollback would silently change nothing yet "succeed").
        """
        if num_turns < 1:
            # Client-side fail-fast — NO wire call. A 0 (or negative) rollback is a no-op
            # the schema description forbids; surfacing it as an error prevents a false
            # green where "rollback succeeded" but nothing was rolled back.
            raise ValueError(
                f"thread/rollback num_turns must be >= 1 (got {num_turns}); a numTurns=0 "
                "rollback is a no-op the schema description forbids")
        transport = self._resident_transport(thread_id)
        # HISTORY-ONLY: the ONLY effect is the thread/rollback request. We deliberately do
        # NOT touch the filesystem (no snapshot restore, no file delete) — reverting local
        # file changes is the caller's responsibility per the schema.
        result = await transport.request(
            _ROLLBACK_METHOD, {"threadId": thread_id, "numTurns": num_turns})
        thread = (result or {}).get("thread") or {}
        rolled_id = thread.get("id")
        if not rolled_id:
            raise RuntimeError(
                f"thread/rollback returned no thread.id for {thread_id!r}: "
                f"{result!r}"[:500])
        return str(rolled_id)

    # ====================================================================== #
    # Skills: a capability-scoped CONTROL PLANE (discover + register roots).  #
    # Reachable ONLY via the SkillsCapable optional protocol (core/backend.py)#
    # — NEVER from run_task (skills discovery is not part of the contract      #
    # loop). Like steer/fork/rollback these act on a RESIDENT thread's         #
    # transport (the only connection the backend owns) and are an INDEPENDENT  #
    # surface: they touch no ContractState, drive no RunHandle, emit no event. #
    #                                                                          #
    # RT-APP-SKILLS-ROOT: ``skills/extraRoots/set`` is GLOBAL, not thread-      #
    # scoped. A caller that registers extra roots #
    # MUST reset them (clear_skills_extra_roots) on teardown, or they leak into #
    # the (shared) CODEX_HOME and pollute a later list/resume/skills view —     #
    # which is why skills live tests use a TEMP CODEX_HOME and clear extraRoots #
    # at the end.                                                              #
    # ====================================================================== #

    async def skills_list(self, thread_id: str, *, cwds: "list[str] | None" = None,
                          force_reload: bool = False) -> list[dict[str, Any]]:
        """Discover the skills available on a RESIDENT thread (``skills/list``).

        Implements part of :class:`~tilldone.core.backend.SkillsCapable`. The wire
        request is ``skills/list`` (``SkillsListParams`` {cwds?, forceReload?}); the
        response is a ``SkillsListResponse`` whose ``data`` is one ``SkillsListEntry``
        per cwd ({cwd, errors, skills:[SkillMetadata]}). The parsed ``data`` list is
        returned VERBATIM (each entry's ``skills`` carries name/description/enabled/
        path/scope) — this is a discovery surface, not a contract channel, so the rich
        metadata is passed through rather than projected onto core types.

        ``cwds`` is OMITTED when ``None``/empty (the server then defaults to the current
        session cwd, per the schema); ``forceReload`` is sent only when ``True`` (it
        bypasses the skills cache and re-scans from disk). Operates on the resident
        transport — an unknown thread fails fast (the skills surface owns no connection
        of its own).
        """
        transport = self._resident_transport(thread_id)
        params: dict[str, Any] = {}
        if cwds:
            params["cwds"] = list(cwds)
        if force_reload:
            params["forceReload"] = True
        result = await transport.request(
            _SKILLS_LIST_METHOD, params if params else None)
        data = (result or {}).get("data")
        # The schema makes ``data`` required; tolerate a missing/non-list value by
        # reporting an empty discovery rather than corrupting the caller's view.
        return list(data) if isinstance(data, list) else []

    async def set_skills_extra_roots(self, thread_id: str,
                                     roots: "list[str]") -> None:
        """Register extra skill-root directories (``skills/extraRoots/set``).

        Implements part of :class:`~tilldone.core.backend.SkillsCapable`. The wire
        request is ``skills/extraRoots/set`` (``SkillsExtraRootsSetParams``
        {extraRoots:[AbsolutePathBuf]}); each root is resolved to an ABSOLUTE path
        because the schema's ``AbsolutePathBuf`` requires it (a relative path would be
        rejected by the server). The response (``SkillsExtraRootsSetResponse``) is empty.

        RT-APP-SKILLS-ROOT: this mutates GLOBAL (NOT thread-scoped) state — the
        registered roots persist in the (shared) home until reset. A caller MUST
        :meth:`clear_skills_extra_roots` when done so the registration does not leak
        into a later list/resume/skills view of the same CODEX_HOME.
        """
        transport = self._resident_transport(thread_id)
        extra_roots = [os.path.abspath(str(r)) for r in roots]
        await transport.request(
            _SKILLS_EXTRA_ROOTS_SET_METHOD, {"extraRoots": extra_roots})

    async def clear_skills_extra_roots(self, thread_id: str) -> None:
        """Reset the GLOBAL extra skill-roots to ``[]`` (RT-APP-SKILLS-ROOT teardown).

        Implements part of :class:`~tilldone.core.backend.SkillsCapable`. Issues
        ``skills/extraRoots/set`` with an EMPTY ``extraRoots`` — the hygiene primitive
        that undoes a prior :meth:`set_skills_extra_roots`. Because extra roots are
        global, NOT thread-scoped, this MUST be called on teardown (or the registered
        roots pollute the shared home's subsequent list/resume/skills state). Sending an
        explicit empty list is the reset, distinct from "never set" (no-op).
        """
        transport = self._resident_transport(thread_id)
        await transport.request(
            _SKILLS_EXTRA_ROOTS_SET_METHOD, {"extraRoots": []})
