from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path


def _digest(p: Path) -> str | None:
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def snapshot_paths(paths: Iterable[Path]) -> dict[Path, str | None]:
    return {Path(p): _digest(Path(p)) for p in paths}


def changed_since(path: Path, snap: Mapping[Path, str | None]) -> bool:
    prior = snap.get(Path(path), None)
    return _digest(Path(path)) != prior
