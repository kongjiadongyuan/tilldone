"""@ref expansion + loop wiring (RT-REF-1/4): real path, fail-fast, containment."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilldone.core.contract import CompletionContract
from tilldone.core.enums import EventKind
from tilldone.core.events import AgentEvent
from tilldone.core.loop import CorePolicy, run_task
from tilldone.core.refs import RefError, expand_refs
from tilldone.core.spec import BackendCapabilities, ContextDirView, ResumeHandle, RunOutcome


# ---------------- pure expand_refs ----------------

def test_expand_valid_to_absolute_path(tmp_path):
    (tmp_path / "a.txt").write_text("hi")
    out = expand_refs("read @ref:src/a.txt now", [ContextDirView(id="src", path=tmp_path)])
    assert str((tmp_path / "a.txt").resolve()) in out and "@ref" not in out


def test_unknown_id_fails_fast(tmp_path):
    with pytest.raises(RefError):
        expand_refs("@ref:nope/x.txt", [ContextDirView(id="src", path=tmp_path)])


def test_parent_traversal_rejected(tmp_path):
    (tmp_path / "sub").mkdir()
    cds = [ContextDirView(id="src", path=tmp_path / "sub")]
    with pytest.raises(RefError):
        expand_refs("@ref:src/../secret.txt", cds)


def test_absolute_relpath_rejected(tmp_path):
    with pytest.raises(RefError):
        expand_refs("@ref:src//etc/passwd", [ContextDirView(id="src", path=tmp_path)])


def test_unicode_filename_and_trailing_punctuation(tmp_path):
    (tmp_path / "文档.txt").write_text("x")
    out = expand_refs("see @ref:d/文档.txt.", [ContextDirView(id="d", path=tmp_path)])
    assert str((tmp_path / "文档.txt").resolve()) in out


def test_multiple_refs_expanded(tmp_path):
    (tmp_path / "a").write_text("1")
    (tmp_path / "b").write_text("2")
    cds = [ContextDirView(id="d", path=tmp_path)]
    out = expand_refs("@ref:d/a and @ref:d/b", cds)
    assert str((tmp_path / "a").resolve()) in out and str((tmp_path / "b").resolve()) in out


# ---------------- loop wiring ----------------

class _RecordingHandle:
    @property
    async def events(self):
        yield AgentEvent(EventKind.RUN_COMPLETED, 1, {})

    async def outcome(self):
        return RunOutcome(status="completed", resume=ResumeHandle("rec", {"id": "1"}))

    async def cancel(self):
        ...


class _RecordingBackend:
    name = "rec"

    def __init__(self):
        self.spec = None

    def capabilities(self):
        return BackendCapabilities(host_tool_calls=True)

    async def start(self, spec, executor):
        self.spec = spec
        return _RecordingHandle()

    async def aclose(self):
        ...


async def test_loop_expands_refs_and_forwards_context_dirs(tmp_path):
    (tmp_path / "in.txt").write_text("data")
    cd = ContextDirView(id="src", path=tmp_path)
    backend = _RecordingBackend()
    contract = CompletionContract(outputs=(), returns=(), todos=(), require_finish=False)
    await run_task(contract, backend, prompt="open @ref:src/in.txt please",
                   cwd=tmp_path, context_dirs=[cd], policy=CorePolicy(max_rounds=1))
    assert backend.spec is not None
    assert str((tmp_path / "in.txt").resolve()) in backend.spec.prompt
    assert "@ref" not in backend.spec.prompt
    assert list(backend.spec.context_dirs) == [cd]


async def test_loop_unknown_ref_fails_fast_before_backend(tmp_path):
    backend = _RecordingBackend()
    contract = CompletionContract(outputs=(), returns=(), todos=(), require_finish=False)
    with pytest.raises(RefError):
        await run_task(contract, backend, prompt="@ref:ghost/x.txt", cwd=tmp_path,
                       context_dirs=[], policy=CorePolicy(max_rounds=1))
    assert backend.spec is None  # failed before starting the backend
