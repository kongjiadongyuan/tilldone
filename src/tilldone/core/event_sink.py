"""AR-1 — JsonlEventSink: tail-friendly JSONL writer for AgentEvent streams.

Stdlib-only (json + pathlib). No backend or SDK imports.

Contract
--------
- sink(event) appends ONE JSON line to events_path containing:
    kind (string value), seq, data, and any non-None correlation id fields.
    The `raw` field is NEVER written to events_path.
- When messages_path is given and event.raw is not None, appends ONE JSON line
    {"seq": ..., "raw": <json-safe raw>} to messages_path.
    If raw is not JSON-serialisable, falls back to repr(raw).
- Each line is flushed immediately so a viewer can tail the file.
- close() / context-manager: files are closed; calling close() twice is safe.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Any

from tilldone.core.events import AgentEvent

# Correlation id field names declared on AgentEvent.
_CORR_FIELDS = ("session_id", "thread_id", "turn_id", "item_id", "call_id")


def _to_json_safe(raw: Any) -> Any:
    """Return a JSON-serialisable representation of *raw*, falling back to repr."""
    try:
        json.dumps(raw)
        return raw
    except (TypeError, ValueError):
        return repr(raw)


class JsonlEventSink:
    """Callable event sink that writes normalised events to a JSONL file.

    Parameters
    ----------
    events_path:
        Destination file for normalised events (no ``raw``).
    messages_path:
        Optional destination file for raw backend payloads.
        A line is written here only when ``event.raw is not None``.
    public_only:
        When ``True``, reasoning events (``kind.value == "reasoning"``) are
        NOT written to ``events_path`` (the public feed).  Their ``raw``
        payload, if any, is still written to ``messages_path`` so the debug
        layer remains faithful.  Default ``False`` preserves the original
        behaviour where all events including reasoning are written.

    Notes
    -----
    RT-APP-REASON (raw/debug channel): independently of ``public_only``, an event
    that carries ``data["raw_only"] is True`` is treated as a RAW/DEBUG-only event —
    it is EXCLUDED from the public ``events_path`` feed but its ``raw`` payload still
    goes to ``messages_path``.  This is the cross-backend hook a backend uses to mark
    a high-volume / privacy-sensitive signal (e.g. app-server reasoning, which is NOT
    exposed publicly unless ``experimentalRawEvents`` is opted in) as debug-only,
    without the sink needing to know any backend's event vocabulary.  Events that do
    NOT set the marker are unaffected (default behaviour preserved).
    """

    def __init__(
        self,
        events_path: Path,
        messages_path: Path | None = None,
        *,
        public_only: bool = False,
    ) -> None:
        self._events_path = Path(events_path)
        self._messages_path = Path(messages_path) if messages_path is not None else None
        self._public_only = public_only
        self._events_fh: IO[str] | None = None
        self._messages_fh: IO[str] | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _events_file(self) -> IO[str]:
        if self._events_fh is None:
            self._events_fh = self._events_path.open("a", encoding="utf-8")
        return self._events_fh

    def _messages_file(self) -> IO[str]:
        assert self._messages_path is not None
        if self._messages_fh is None:
            self._messages_fh = self._messages_path.open("a", encoding="utf-8")
        return self._messages_fh

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, event: AgentEvent) -> None:
        """Append the event to events_path (and raw to messages_path if set)."""
        if self._closed:
            raise RuntimeError("JsonlEventSink is closed")

        is_reasoning = event.kind.value == "reasoning"
        # RT-APP-REASON: a per-event RAW/DEBUG-only marker. An event flagged
        # data["raw_only"] is True is kept OUT of the public feed regardless of
        # public_only (its raw still reaches messages_path below). This is how a
        # backend marks reasoning (or any raw-only signal) debug-only by default
        # without the sink hard-coding a backend's event kinds.
        is_raw_only = event.data.get("raw_only") is True

        # An event is excluded from the public events feed when it is raw-only, OR
        # (legacy) when public_only suppresses reasoning. In both cases its raw payload
        # still goes to messages_path so the debug layer stays faithful.
        if not (is_raw_only or (self._public_only and is_reasoning)):
            # Build the normalised record — kind as string, no `raw`.
            record: dict[str, Any] = {
                "kind": event.kind.value,
                "seq": event.seq,
                "data": dict(event.data),
            }
            # Include only non-None correlation ids.
            for field in _CORR_FIELDS:
                val = getattr(event, field)
                if val is not None:
                    record[field] = val

            ef = self._events_file()
            ef.write(json.dumps(record, ensure_ascii=False))
            ef.write("\n")
            ef.flush()

        # Raw payload — only when messages_path provided and raw is present.
        # Always written regardless of public_only so the debug layer is faithful.
        if self._messages_path is not None and event.raw is not None:
            raw_record: dict[str, Any] = {
                "seq": event.seq,
                "raw": _to_json_safe(event.raw),
            }
            mf = self._messages_file()
            mf.write(json.dumps(raw_record, ensure_ascii=False))
            mf.write("\n")
            mf.flush()

    def close(self) -> None:
        """Close all open file handles. Safe to call more than once."""
        if self._events_fh is not None:
            self._events_fh.close()
            self._events_fh = None
        if self._messages_fh is not None:
            self._messages_fh.close()
            self._messages_fh = None
        self._closed = True

    # Context-manager support

    def __enter__(self) -> "JsonlEventSink":
        self._closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
