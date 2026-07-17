"""Tests for tilldone.recipes.translate — TDD (AR-7).

All tests use FakeBackend; no live model is called.
"""
from __future__ import annotations

import pytest

from tilldone.backends.fake import FakeBackend, Script, WriteFile, Finish
from tilldone.core.loop import MaxRoundsExceeded, CorePolicy
from tilldone.recipes.translate import translate


async def test_translate_completes_and_writes_dest(tmp_path):
    """FakeBackend writes dest + calls Finish -> contract satisfied, result returned."""
    source = tmp_path / "hello.txt"
    source.write_text("Hello, world!")
    dest = tmp_path / "hello_fr.txt"

    backend = FakeBackend(Script([
        WriteFile(dest, "Bonjour, le monde!"),
        Finish(),
    ]))

    result = await translate(
        backend,
        source=source,
        dest=dest,
        target_lang="French",
        cwd=tmp_path,
    )

    assert dest.exists(), "dest file must be created by the backend"
    assert dest.read_text() == "Bonjour, le monde!"
    # translate returns run_task result; contract has 0 returns -> None
    assert result is None


async def test_translate_prompt_references_source_via_ref(tmp_path):
    """The @ref pipeline is exercised end-to-end: loop expands @ref and the
    FakeBackend (which ignores the prompt entirely) still satisfies the contract
    by writing dest.  If @ref expansion failed the call would raise RefError
    before the backend is ever contacted, so a clean completion proves the
    @ref was valid."""
    source = tmp_path / "document.txt"
    source.write_text("The quick brown fox.")
    dest = tmp_path / "document_es.txt"

    # FakeBackend ignores prompt; it just writes dest + finishes.
    backend = FakeBackend(Script([
        WriteFile(dest, "El veloz zorro marrón."),
        Finish(),
    ]))

    result = await translate(
        backend,
        source=source,
        dest=dest,
        target_lang="Spanish",
        cwd=tmp_path,
    )

    assert dest.exists()
    assert result is None


async def test_translate_unsatisfied_raises(tmp_path):
    """Backend does nothing -> dest never written -> MaxRoundsExceeded."""
    source = tmp_path / "src.txt"
    source.write_text("Some content.")
    dest = tmp_path / "out.txt"

    backend = FakeBackend(Script([]))  # does nothing

    with pytest.raises(MaxRoundsExceeded):
        await translate(
            backend,
            source=source,
            dest=dest,
            target_lang="German",
            cwd=tmp_path,
            policy=CorePolicy(max_rounds=1),
        )
