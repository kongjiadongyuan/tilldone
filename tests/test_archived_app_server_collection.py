"""Default pytest collection keeps archived app-server tests out of the mainline."""

from __future__ import annotations

from pathlib import Path

from conftest import _is_archived_app_server_path


def test_archived_app_server_paths_are_identified() -> None:
    root = Path(__file__).resolve().parent.parent
    assert _is_archived_app_server_path(root / "tests/backends/app_server/test_backend.py")
    assert _is_archived_app_server_path(root / "tests/recipes/test_recipe_app_server.py")
    assert not _is_archived_app_server_path(root / "tests/backends/test_codex_unit.py")
