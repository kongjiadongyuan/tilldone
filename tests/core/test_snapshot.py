from pathlib import Path

from tilldone.core.snapshot import snapshot_paths, changed_since


def test_detects_create_and_update(tmp_path: Path):
    p = tmp_path / "out.txt"
    before = snapshot_paths([p])
    assert before[p] is None              # absent at snapshot time
    p.write_text("v1")
    snap1 = snapshot_paths([p])
    assert changed_since(p, before) is True       # created (absent -> present)
    assert changed_since(p, snap1) is False       # unchanged vs snap1
    p.write_text("v2")
    assert changed_since(p, snap1) is True         # content changed


def test_directory_is_not_a_file(tmp_path: Path):
    d = tmp_path / "adir"
    d.mkdir()
    snap = snapshot_paths([d])
    assert snap[d] is None                # a directory has no file digest
    assert changed_since(d, snap) is False
