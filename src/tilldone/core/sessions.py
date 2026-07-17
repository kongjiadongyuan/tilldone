"""Session index for TillDone (AR-5).

Persists run records to a JSON-lines file so callers can list past runs
and resume one via ``from_session``.  Uses only stdlib (json, pathlib,
dataclasses) — no external dependencies.

Ordering guarantee
------------------
``list_sessions()`` returns records in **newest-first** order based on the
monotonically-increasing insertion index stored in each record.  On upsert
(same session_id recorded again) the existing record is replaced in-place and
its index is updated to the latest counter value, so it moves to the front of
the list.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tilldone.core.spec import ResumeHandle


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionInfo:
    """Read-only view of a persisted session record."""

    session_id: str
    backend: str
    aliases: tuple[str, ...]
    context_dir_ids: tuple[str, ...]
    label: str
    resumable: bool


class SessionRestoreError(Exception):
    """Raised when ``from_session`` cannot restore a session.

    Causes:
    - Unknown session_id.
    - One or more recorded aliases are absent from *available_aliases*.
    - One or more recorded context_dir_ids are absent from
      *available_context_dir_ids*.
    """


# ---------------------------------------------------------------------------
# Internal record schema (what is stored in the JSONL file)
# ---------------------------------------------------------------------------

# {
#   "session_id": str,
#   "backend": str,
#   "token": dict,          # ResumeHandle.token (JSON-serialisable)
#   "aliases": [str, ...],
#   "context_dir_ids": [str, ...],
#   "label": str,
#   "idx": int              # monotonic insertion counter
# }


# ---------------------------------------------------------------------------
# SessionIndex
# ---------------------------------------------------------------------------


class SessionIndex:
    """Append-backed session store; keyed by *session_id*."""

    def __init__(self, path: Path) -> None:
        self._path = path
        # In-memory dict keyed by session_id; preserves last-write state.
        self._records: dict[str, dict[str, Any]] = {}
        self._counter: int = 0
        self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Read all records from the JSONL file (if it exists)."""
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = rec.get("session_id")
                if not sid:
                    continue
                self._records[sid] = rec
                # Track highest seen index so future writes don't collide.
                idx = rec.get("idx", 0)
                if idx >= self._counter:
                    self._counter = idx + 1

    def _flush(self) -> None:
        """Rewrite the JSONL file from the in-memory dict.

        We always do a full rewrite so that upserts are reflected correctly.
        Files are small (one line per run), so this is fine for v1.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            # Write in insertion order (ascending idx) so _load rebuilds
            # the same state on the next open.
            for rec in sorted(self._records.values(), key=lambda r: r["idx"]):
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def _make_resume_handle(self, rec: dict[str, Any]) -> ResumeHandle:
        return ResumeHandle(backend=rec["backend"], token=rec["token"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        session_id: str,
        *,
        backend: str,
        resume: ResumeHandle,
        aliases: Sequence[str] = (),
        context_dir_ids: Sequence[str] = (),
        label: str = "",
    ) -> None:
        """Append or update a session record (upsert keyed by *session_id*).

        The token stored in *resume* must be JSON-serialisable.
        """
        token: Mapping[str, Any] = resume.token
        rec: dict[str, Any] = {
            "session_id": session_id,
            "backend": backend,
            "token": dict(token),
            "aliases": list(aliases),
            "context_dir_ids": list(context_dir_ids),
            "label": label,
            "idx": self._counter,
        }
        self._counter += 1
        self._records[session_id] = rec
        self._flush()

    def list_sessions(self) -> list[SessionInfo]:
        """Return all session records, newest-recorded first."""
        sorted_recs = sorted(
            self._records.values(), key=lambda r: r["idx"], reverse=True
        )
        result: list[SessionInfo] = []
        for rec in sorted_recs:
            token = rec.get("token", {})
            resumable = bool(token)
            result.append(
                SessionInfo(
                    session_id=rec["session_id"],
                    backend=rec["backend"],
                    aliases=tuple(rec.get("aliases", [])),
                    context_dir_ids=tuple(rec.get("context_dir_ids", [])),
                    label=rec.get("label", ""),
                    resumable=resumable,
                )
            )
        return result

    def from_session(
        self,
        session_id: str,
        *,
        available_aliases: Sequence[str] = (),
        available_context_dir_ids: Sequence[str] = (),
    ) -> ResumeHandle:
        """Return the stored ``ResumeHandle`` for *session_id*.

        Raises ``SessionRestoreError`` if:
        - *session_id* is not found.
        - Any recorded alias is absent from *available_aliases*.
        - Any recorded context_dir_id is absent from *available_context_dir_ids*.

        This implements RT-SESS-4: missing aliases/context_dir_ids fail fast
        with a message that names every missing item.
        """
        rec = self._records.get(session_id)
        if rec is None:
            raise SessionRestoreError(
                f"Session '{session_id}' not found in the index."
            )

        available_alias_set = set(available_aliases)
        available_ctx_set = set(available_context_dir_ids)

        missing_aliases = [a for a in rec.get("aliases", []) if a not in available_alias_set]
        missing_ctx_ids = [c for c in rec.get("context_dir_ids", []) if c not in available_ctx_set]

        if missing_aliases or missing_ctx_ids:
            parts: list[str] = []
            if missing_aliases:
                parts.append(
                    "missing aliases: " + ", ".join(repr(a) for a in missing_aliases)
                )
            if missing_ctx_ids:
                parts.append(
                    "missing context_dir_ids: " + ", ".join(repr(c) for c in missing_ctx_ids)
                )
            raise SessionRestoreError(
                f"Cannot restore session '{session_id}': " + "; ".join(parts) + "."
            )

        return self._make_resume_handle(rec)
