"""Tests for SessionIndex (AR-5 — session index).

TDD: these tests are written BEFORE the implementation.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from tilldone.core.spec import ResumeHandle
from tilldone.core.sessions import SessionIndex, SessionInfo, SessionRestoreError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _handle(backend: str, session_id: str) -> ResumeHandle:
    return ResumeHandle(backend=backend, token={"session_id": session_id})


# ---------------------------------------------------------------------------
# Test 1: record two runs, list_sessions returns both with correct fields
# ---------------------------------------------------------------------------

def test_list_sessions_basic(tmp_path: Path) -> None:
    idx = SessionIndex(tmp_path / "sessions.jsonl")

    h_claude = _handle("claude", "aaa-111")
    h_codex = _handle("codex", "bbb-222")

    idx.record("aaa-111", backend="claude", resume=h_claude, aliases=["my_tool"], label="first run")
    idx.record("bbb-222", backend="codex", resume=h_codex, context_dir_ids=["dir-A"])

    sessions = idx.list_sessions()
    assert len(sessions) == 2

    ids = {s.session_id for s in sessions}
    assert "aaa-111" in ids
    assert "bbb-222" in ids

    by_id = {s.session_id: s for s in sessions}

    s1 = by_id["aaa-111"]
    assert s1.backend == "claude"
    assert list(s1.aliases) == ["my_tool"]
    assert list(s1.context_dir_ids) == []
    assert s1.label == "first run"
    assert s1.resumable is True

    s2 = by_id["bbb-222"]
    assert s2.backend == "codex"
    assert list(s2.aliases) == []
    assert list(s2.context_dir_ids) == ["dir-A"]
    assert s2.resumable is True


def test_resumable_false_when_empty_token(tmp_path: Path) -> None:
    idx = SessionIndex(tmp_path / "sessions.jsonl")
    empty_handle = ResumeHandle(backend="claude", token={})
    idx.record("no-token", backend="claude", resume=empty_handle)

    sessions = idx.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].resumable is False


# ---------------------------------------------------------------------------
# Test 2: from_session returns the exact ResumeHandle when all deps present
# ---------------------------------------------------------------------------

def test_from_session_happy_path(tmp_path: Path) -> None:
    idx = SessionIndex(tmp_path / "sessions.jsonl")
    h = _handle("claude", "happy-123")
    idx.record(
        "happy-123",
        backend="claude",
        resume=h,
        aliases=["tool_a", "tool_b"],
        context_dir_ids=["dir-1"],
    )

    result = idx.from_session(
        "happy-123",
        available_aliases=["tool_a", "tool_b", "tool_c"],
        available_context_dir_ids=["dir-1", "dir-2"],
    )
    assert result == h


# ---------------------------------------------------------------------------
# Test 3: from_session raises SessionRestoreError on missing alias/context_id
# ---------------------------------------------------------------------------

def test_from_session_missing_alias(tmp_path: Path) -> None:
    idx = SessionIndex(tmp_path / "sessions.jsonl")
    h = _handle("claude", "alias-sess")
    idx.record("alias-sess", backend="claude", resume=h, aliases=["required_tool"])

    with pytest.raises(SessionRestoreError) as exc_info:
        idx.from_session("alias-sess", available_aliases=["other_tool"])

    assert "required_tool" in str(exc_info.value)


def test_from_session_missing_context_dir_id(tmp_path: Path) -> None:
    idx = SessionIndex(tmp_path / "sessions.jsonl")
    h = _handle("codex", "ctx-sess")
    idx.record("ctx-sess", backend="codex", resume=h, context_dir_ids=["needed-dir"])

    with pytest.raises(SessionRestoreError) as exc_info:
        idx.from_session("ctx-sess", available_context_dir_ids=["other-dir"])

    assert "needed-dir" in str(exc_info.value)


def test_from_session_multiple_missing_all_named(tmp_path: Path) -> None:
    """Error message should list ALL missing items, not just the first."""
    idx = SessionIndex(tmp_path / "sessions.jsonl")
    h = _handle("claude", "multi-sess")
    idx.record(
        "multi-sess",
        backend="claude",
        resume=h,
        aliases=["alpha", "beta"],
        context_dir_ids=["dir-x"],
    )

    with pytest.raises(SessionRestoreError) as exc_info:
        idx.from_session("multi-sess")  # nothing available

    msg = str(exc_info.value)
    assert "alpha" in msg
    assert "beta" in msg
    assert "dir-x" in msg


# ---------------------------------------------------------------------------
# Test 4: from_session on unknown id raises SessionRestoreError
# ---------------------------------------------------------------------------

def test_from_session_unknown_id(tmp_path: Path) -> None:
    idx = SessionIndex(tmp_path / "sessions.jsonl")

    with pytest.raises(SessionRestoreError):
        idx.from_session("nonexistent-id")


# ---------------------------------------------------------------------------
# Test 5: round-trip persistence
# ---------------------------------------------------------------------------

def test_persistence_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.jsonl"
    h = _handle("codex", "persist-999")

    # Write with first instance
    idx1 = SessionIndex(db_path)
    idx1.record(
        "persist-999",
        backend="codex",
        resume=h,
        aliases=["my_alias"],
        context_dir_ids=["my_dir"],
        label="persisted",
    )

    # Read with fresh instance
    idx2 = SessionIndex(db_path)
    sessions = idx2.list_sessions()
    assert len(sessions) == 1
    s = sessions[0]
    assert s.session_id == "persist-999"
    assert s.backend == "codex"
    assert list(s.aliases) == ["my_alias"]
    assert list(s.context_dir_ids) == ["my_dir"]
    assert s.label == "persisted"
    assert s.resumable is True

    # from_session also works on fresh instance
    result = idx2.from_session(
        "persist-999",
        available_aliases=["my_alias"],
        available_context_dir_ids=["my_dir"],
    )
    assert result == h


# ---------------------------------------------------------------------------
# Test 6: record with same session_id updates existing record (upsert)
# ---------------------------------------------------------------------------

def test_record_upsert(tmp_path: Path) -> None:
    idx = SessionIndex(tmp_path / "sessions.jsonl")
    h1 = _handle("claude", "upsert-sess")
    h2 = ResumeHandle(backend="claude", token={"session_id": "upsert-sess", "v": 2})

    idx.record("upsert-sess", backend="claude", resume=h1, label="v1")
    idx.record("upsert-sess", backend="claude", resume=h2, label="v2")

    sessions = idx.list_sessions()
    # Should have only one record after upsert
    assert len(sessions) == 1
    assert sessions[0].label == "v2"

    result = idx.from_session("upsert-sess", available_aliases=[], available_context_dir_ids=[])
    assert result == h2


# ---------------------------------------------------------------------------
# Test 7: SessionInfo is a frozen dataclass
# ---------------------------------------------------------------------------

def test_session_info_is_frozen(tmp_path: Path) -> None:
    idx = SessionIndex(tmp_path / "sessions.jsonl")
    h = _handle("claude", "frozen-sess")
    idx.record("frozen-sess", backend="claude", resume=h)

    sessions = idx.list_sessions()
    info = sessions[0]

    with pytest.raises((AttributeError, TypeError)):
        info.backend = "codex"  # type: ignore[misc]
