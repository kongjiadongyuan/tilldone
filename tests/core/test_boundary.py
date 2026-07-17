import ast
import pathlib

FORBIDDEN = {"claude_agent_sdk", "subprocess", "codex"}


def _core_dir() -> pathlib.Path:
    # tests/core/test_boundary.py -> parents[2] == repo root
    return pathlib.Path(__file__).resolve().parents[2] / "src" / "tilldone" / "core"


def test_core_has_no_backend_or_subprocess_imports():
    core = _core_dir()
    assert core.is_dir(), f"core dir not found: {core}"
    offenders: list[tuple[str, str]] = []
    for f in core.rglob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    if n.name.split(".")[0] in FORBIDDEN:
                        offenders.append((str(f), n.name))
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in FORBIDDEN:
                    offenders.append((str(f), node.module))
    assert offenders == [], f"core leaked forbidden imports: {offenders}"
