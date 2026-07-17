"""@ref expansion — resolve ``@ref:<id>/<relpath>`` tokens to absolute paths.

A Task prompt may reference files inside a declared context_dir via ``@ref:<id>/<relpath>``.
Core expands each token to the absolute path of the file (the backend makes the context_dir
readable via add_dirs / --add-dir), so the agent can read real content. Safety (RT-REF-1/4):
unknown id -> fail-fast; the relpath must stay inside the context_dir (canonical containment;
``..`` / absolute / symlink escape rejected).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

# @ref:<id>/<relpath> — id is [A-Za-z0-9_.-], relpath is the following non-space run.
_REF_RE = re.compile(r"@ref:([A-Za-z0-9_.-]+)/(\S+)")
_TRAILING_PUNCT = ".,;:!?)\"']}"


class RefError(ValueError):
    """A @ref could not be expanded (unknown id, traversal, or absolute path)."""


def expand_refs(prompt: str, context_dirs: Sequence) -> str:
    """Replace every ``@ref:<id>/<relpath>`` with the absolute resolved path.

    Raises RefError on an unknown context_dir id or a relpath that escapes it.
    """
    by_id = {cd.id: cd for cd in context_dirs}

    def _replace(match: re.Match) -> str:
        cid, rel = match.group(1), match.group(2)
        rel = rel.rstrip(_TRAILING_PUNCT)  # tolerate trailing prose punctuation
        cd = by_id.get(cid)
        if cd is None:
            raise RefError(f"unknown context_dir id in @ref: {cid!r}")
        if rel.startswith("/") or rel.startswith("~") or rel == "":
            raise RefError(f"@ref relpath must be a non-empty relative path: {rel!r}")
        base = Path(cd.path).resolve()
        target = (base / rel).resolve()
        if target != base and base not in target.parents:
            raise RefError(f"@ref {cid!r}/{rel!r} escapes its context_dir {str(base)!r}")
        return str(target)

    return _REF_RE.sub(_replace, prompt)
