"""Pure ``codex app-server`` notification → :class:`AgentEvent` mapper (M2).

This is the normalization seam between the raw JSON-RPC *notification* dicts
that :class:`~tilldone.backends.app_server.transport.AppServerTransport`
surfaces (``{"method", "params"}``) and the cross-backend
:class:`~tilldone.core.events.AgentEvent` vocabulary
(:class:`~tilldone.core.enums.EventKind`).

Design
------
* **Pure**: ``map_notification(note, seq)`` is a deterministic dict→event (or
  ``None`` to drop) function with no I/O and no hidden state, so it is fully
  unit-testable in isolation.
* **Table-driven**: dispatch is keyed on ``method`` and, for ``item/*``, on
  ``item.type``. Anything not in the table is dropped (returns ``None``) — the
  app-server emits a lot of lifecycle/telemetry noise (``warning``,
  ``thread/status/changed``, ``account/rateLimits/updated``,
  ``remoteControl/status/changed``, …) that is not part of the stable view.
* **Correlation ids preserved**: thread / turn / item ids are lifted onto the
  event (``thread_id`` / ``turn_id`` / ``item_id`` / ``call_id``) so downstream
  consumers can correlate.

Wire facts (app-server notification shape):

* Item ``type`` values are **camelCase** on the app-server wire
  (``agentMessage`` / ``reasoning`` / ``commandExecution`` / ``fileChange`` /
  ``mcpToolCall`` / ``dynamicToolCall`` / ``userMessage``) — distinct from the
  ``codex exec`` snake_case stream. We accept the snake_case aliases too, for
  robustness, but the live notifications use camelCase.
* ``item/completed`` fires for **every** item type (incl. ``userMessage``); the
  final answer MUST be filtered to ``agentMessage`` + (preferably)
  ``phase=="final_answer"``. ``userMessage`` is dropped here.
* ``turn/completed`` is the sole terminal; ``turn.status`` ∈
  ``completed|interrupted|failed|inProgress`` maps the terminal kind.

This module imports only value types from :mod:`tilldone.core` — it never
imports the transport, and importing it must not pull the transport into
``tilldone.core`` (the 0-core-diff boundary runs the other direction).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from tilldone.core.enums import EventKind
from tilldone.core.events import AgentEvent

# Item ``type`` discriminators (camelCase live wire + snake_case exec aliases).
_AGENT_MESSAGE = ("agentMessage", "agent_message")
_REASONING = ("reasoning",)
_COMMAND = ("commandExecution", "command_execution")
_TOOL_CALL = ("mcpToolCall", "mcp_tool_call", "dynamicToolCall", "dynamic_tool_call")
_FILE_CHANGE = ("fileChange", "file_change", "patch", "file_update")
# Items that are part of the protocol but never enter the stable event view.
_DROPPED_ITEMS = ("userMessage", "user_message")

# Turn statuses that mean "the run ended unsuccessfully".
_TERMINAL_OK = "completed"

# -- RT-APP-REASON: reasoning is a DEBUG/RAW channel, not the default public feed ----
#
# Proposal  (RT-APP-REASON): "app-server reasoning delta/item 默认不进
# public sink,只进 raw/debug;experimentalRawEvents 不默认开". Reasoning (the model's
# chain-of-thought) is privacy-sensitive + high-volume, so it must NOT land in the
# DEFAULT public event feed — only in the raw/debug layer — unless a caller explicitly
# opts into raw events (the ``experimentalRawEvents`` posture, which is NOT on by
# default).
#
# The mapper stamps every reasoning event (item OR delta) with ``data[_RAW_ONLY]=True``,
# the DEBUG-CHANNEL marker. This is orthogonal to EventKind (the event is still a
# ``REASONING`` event on the handle's raw stream, so debug consumers + the conformance
# invariant see it), and a public sink keys on the marker to drop it from the public
# feed while still writing its ``raw`` to the debug/messages layer
# (``JsonlEventSink`` honours ``data["raw_only"]``). When the backend is run with
# ``experimental_raw_events=True`` it re-stamps these events public (``raw_only=False``)
# so an opted-in caller sees reasoning in the public feed too.
#: ``data`` key marking an event as RAW/DEBUG-only (excluded from the default public feed).
_RAW_ONLY = "raw_only"

# Reasoning streaming-delta notification methods (ReasoningTextDeltaNotification /
# ReasoningSummaryTextDeltaNotification + a defensive item/reasoning/delta alias). Like
# the reasoning ITEM, a reasoning DELTA is a debug/raw-only signal (RT-APP-REASON) —
# surfaced as a REASONING event tagged ``raw_only`` so it never pollutes the public feed
# by default, rather than silently dropped (so the debug layer stays faithful).
_REASONING_DELTA_METHODS = (
    "item/reasoning/delta",
    "item/reasoningText/delta",
    "item/reasoningSummaryText/delta",
)

# -- RT-APP-RETRY: a retryable ``error`` notification is NOT a terminal ---------------
#
# Proposal  (RT-APP-RETRY): "error notification 带 willRetry;willRetry=
# true 不先发 terminal,等最终 turn/completed/failed". The app-server emits an ``error``
# NOTIFICATION (ErrorNotification: {error:TurnError, threadId, turnId, willRetry}) when a
# turn hits a recoverable problem it will RETRY. That notification is NOT a terminal — the
# turn keeps running and its real terminal is the eventual ``turn/completed`` (status
# completed | failed | interrupted). Mapping it to a non-terminal BACKEND_NOTICE (carrying
# ``will_retry``) makes the retry OBSERVABLE while guaranteeing it can never end the
# stream early (only RUN_COMPLETED / RUN_FAILED do). The sole terminal stays
# ``turn/completed`` (events.py already maps failed/interrupted → RUN_FAILED).
_ERROR_NOTIFICATION_METHOD = "error"


def _correlate(params: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the standard correlation ids from a notification's params/item."""
    return {
        "thread_id": params.get("threadId"),
        "turn_id": params.get("turnId"),
        "item_id": item.get("id"),
    }


def map_notification(note: Mapping[str, Any], *, seq: int) -> Optional[AgentEvent]:
    """Map one app-server notification dict to a normalized :class:`AgentEvent`.

    ``note`` is ``{"method": str, "params": dict | None}`` as produced by
    :meth:`AppServerTransport.notifications`. Returns the mapped event, or
    ``None`` if the notification carries no stable-view signal (and should be
    dropped). ``seq`` is the monotonically-increasing sequence number the caller
    assigns — purity means the mapper never owns the counter.
    """
    method = note.get("method")
    params = note.get("params") or {}

    if method == "thread/started":
        thread = params.get("thread") or {}
        return AgentEvent(
            EventKind.RUN_STARTED, seq,
            data={"thread_id": thread.get("id")},
            thread_id=thread.get("id"), raw=note,
        )

    if method == "turn/started":
        # Lifecycle only — turn_id is tracked off the item/turn-completed frames.
        return None

    if method == "item/agentMessage/delta":
        # Optional streaming text. Surfaced as a streaming ASSISTANT_TEXT; the
        # authoritative final text is the item/completed agentMessage, so this is
        # purely additive (and tagged streaming=True so consumers can distinguish).
        return AgentEvent(
            EventKind.ASSISTANT_TEXT, seq,
            data={"delta": params.get("delta", ""), "streaming": True},
            thread_id=params.get("threadId"), turn_id=params.get("turnId"),
            item_id=params.get("itemId"), raw=note,
        )

    if method in _REASONING_DELTA_METHODS:
        # RT-APP-REASON: a reasoning streaming delta is a DEBUG/RAW-only signal — like the
        # reasoning ITEM, it must NOT enter the default public feed. Surfaced as a
        # streaming REASONING event tagged ``raw_only=True`` (NOT dropped, so the debug
        # layer stays faithful); a public sink keys on the marker to suppress it by default.
        return AgentEvent(
            EventKind.REASONING, seq,
            data={"delta": params.get("delta", ""), "streaming": True, _RAW_ONLY: True},
            thread_id=params.get("threadId"), turn_id=params.get("turnId"),
            item_id=params.get("itemId"), raw=note,
        )

    if method in ("item/started", "item/completed"):
        return _map_item(method, params, seq, note)

    if method == "turn/completed":
        turn = params.get("turn") or {}
        status = turn.get("status")
        kind = EventKind.RUN_COMPLETED if status == _TERMINAL_OK else EventKind.RUN_FAILED
        return AgentEvent(
            kind, seq,
            data={"status": status, "error": turn.get("error")},
            thread_id=params.get("threadId"), turn_id=turn.get("id"), raw=note,
        )

    if method == "thread/tokenUsage/updated":
        return AgentEvent(
            EventKind.USAGE, seq,
            data={"usage": params.get("tokenUsage")},
            thread_id=params.get("threadId"), turn_id=params.get("turnId"), raw=note,
        )

    if method == "model/rerouted":
        # RT-APP-REROUTE: the server rerouted this turn to a DIFFERENT model
        # (ModelReroutedNotification: {fromModel, toModel, reason, threadId, turnId}).
        # A reroute MUST be observable — surfaced as an explicit BACKEND_NOTICE, never
        # dropped as telemetry noise — and carries the from/to/reason so the run can
        # record the FINAL model that actually executed (the model_map real-run proof).
        return AgentEvent(
            EventKind.BACKEND_NOTICE, seq,
            data={
                "notice": "model_rerouted",
                "from_model": params.get("fromModel"),
                "to_model": params.get("toModel"),
                "reason": params.get("reason"),
            },
            thread_id=params.get("threadId"), turn_id=params.get("turnId"), raw=note,
        )

    if method == _ERROR_NOTIFICATION_METHOD:
        # RT-APP-RETRY: an ``error`` NOTIFICATION (ErrorNotification {error, threadId,
        # turnId, willRetry}). This is NOT a terminal — the turn keeps running and its
        # real terminal is the eventual ``turn/completed``. Surfaced as a NON-terminal
        # BACKEND_NOTICE carrying ``will_retry`` so the retry is observable, while it is
        # structurally impossible for it to end the stream early (only RUN_COMPLETED /
        # RUN_FAILED terminate). For a ``willRetry=true`` notice the stream waits for the
        # eventual terminal; for ``willRetry=false`` the accompanying turn/completed
        # (status failed/interrupted) is the terminal that maps to RUN_FAILED.
        error = params.get("error") or {}
        return AgentEvent(
            EventKind.BACKEND_NOTICE, seq,
            data={
                "notice": "turn_error",
                "will_retry": bool(params.get("willRetry")),
                "message": error.get("message"),
                "codex_error_info": error.get("codexErrorInfo"),
            },
            thread_id=params.get("threadId"), turn_id=params.get("turnId"), raw=note,
        )

    if method == "skills/changed":
        # RT-APP-SKILLS-ROOT-adjacent: the server watched local skill files and they
        # changed (SkillsChangedNotification — empty params). The schema says to treat
        # it as an INVALIDATION signal ("re-run skills/list with the client's current
        # parameters"). It MUST be observable — surfaced as an explicit BACKEND_NOTICE,
        # never dropped as telemetry noise — so a skills caller knows its last
        # skills/list is stale.
        return AgentEvent(
            EventKind.BACKEND_NOTICE, seq,
            data={"notice": "skills_changed"},
            raw=note,
        )

    # Everything else (warning, thread/status/changed, account/rateLimits/updated,
    # remoteControl/status/changed, thread/goal/cleared, …) is telemetry noise.
    return None


def _map_item(method: str, params: Mapping[str, Any], seq: int,
              note: Mapping[str, Any]) -> Optional[AgentEvent]:
    item = params.get("item") or {}
    it = item.get("type")
    cid = _correlate(params, item)
    completed = method == "item/completed"

    if it in _AGENT_MESSAGE:
        # Only the completed agentMessage carries the settled text; the started
        # frame has text="" and is not part of the stable view.
        if not completed:
            return None
        return AgentEvent(
            EventKind.ASSISTANT_TEXT, seq,
            data={"text": item.get("text", ""), "phase": item.get("phase")},
            thread_id=cid["thread_id"], turn_id=cid["turn_id"],
            item_id=cid["item_id"], raw=note,
        )

    if it in _REASONING:
        if not completed:
            return None
        # RT-APP-REASON: a reasoning ITEM is DEBUG/RAW-only — tagged ``raw_only`` so a
        # public sink suppresses it from the default feed (while its ``raw`` still reaches
        # the debug/messages layer). It stays a REASONING event on the raw stream.
        return AgentEvent(
            EventKind.REASONING, seq,
            data={"summary": item.get("summary"), "content": item.get("content"),
                  _RAW_ONLY: True},
            thread_id=cid["thread_id"], turn_id=cid["turn_id"],
            item_id=cid["item_id"], raw=note,
        )

    if it in _COMMAND:
        return AgentEvent(
            EventKind.COMMAND, seq,
            data={"command": item.get("command"), "exit_code": item.get("exit_code"),
                  "status": item.get("status"),
                  "phase": "end" if completed else "begin"},
            thread_id=cid["thread_id"], turn_id=cid["turn_id"],
            call_id=cid["item_id"], item_id=cid["item_id"], raw=note,
        )

    if it in _TOOL_CALL:
        name = item.get("tool") or item.get("name")
        if not completed:
            return AgentEvent(
                EventKind.TOOL_CALL, seq,
                data={"name": name, "server": item.get("server"),
                      "namespace": item.get("namespace"),
                      "arguments": item.get("arguments")},
                thread_id=cid["thread_id"], turn_id=cid["turn_id"],
                call_id=cid["item_id"], item_id=cid["item_id"], raw=note,
            )
        result = item.get("result") or {}
        is_error = item.get("status") == "failed" or bool(result.get("isError")) \
            or item.get("success") is False
        return AgentEvent(
            EventKind.TOOL_RESULT, seq,
            data={"name": name, "is_error": is_error},
            thread_id=cid["thread_id"], turn_id=cid["turn_id"],
            call_id=cid["item_id"], item_id=cid["item_id"], raw=note,
        )

    if it in _FILE_CHANGE:
        if not completed:
            return None
        return AgentEvent(
            EventKind.FILE_CHANGED, seq,
            data={"item_type": it},
            thread_id=cid["thread_id"], turn_id=cid["turn_id"],
            item_id=cid["item_id"], raw=note,
        )

    # userMessage and any unknown item type → not part of the stable view.
    return None
