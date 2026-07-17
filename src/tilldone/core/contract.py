from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tilldone.core.snapshot import changed_since
from tilldone.core.validators import OutputValidator


@dataclass(frozen=True)
class OutputExpectation:
    label: str
    abs_path: Path
    expect: Literal["created", "updated"]
    validator: OutputValidator | None = None


@dataclass(frozen=True)
class ReturnExpectation:
    name: str
    type_hint: str | None
    required: bool = True


@dataclass(frozen=True)
class TodoExpectation:
    name: str


@dataclass(frozen=True)
class CompletionContract:
    outputs: tuple[OutputExpectation, ...]
    returns: tuple[ReturnExpectation, ...]
    todos: tuple[TodoExpectation, ...]
    require_finish: bool


@dataclass
class ContractState:
    returns: dict[str, Any] = field(default_factory=dict)
    todos_done: dict[str, str] = field(default_factory=dict)
    finished: bool = False


class ContractEvaluator:
    """The done-ness judge. Authority = filesystem (sha256 vs pre-run snapshot)
    + accumulated ContractState. It NEVER trusts backend 'file written' signals.
    """

    def __init__(self, contract: CompletionContract):
        self.c = contract

    def evaluate(self, state: ContractState, output_snaps) -> list[str]:
        missing: list[str] = []
        for o in self.c.outputs:
            if not self._output_ok(o, output_snaps):
                missing.append(f"output:{o.label}")
        for r in self.c.returns:
            if r.required and r.name not in state.returns:
                missing.append(f"return:{r.name}")
        for t in self.c.todos:
            if t.name not in state.todos_done:
                missing.append(f"todo:{t.name}")
        if self.c.require_finish and not state.finished:
            missing.append("finish")
        return missing

    def _output_ok(self, o: OutputExpectation, output_snaps) -> bool:
        p = o.abs_path
        try:
            if not p.is_file():
                return False
            content = p.read_bytes()
        except OSError:
            return False
        if not content:  # non-empty: empty file is never a satisfied output
            return False
        # Must differ from the pre-run snapshot: defeats stale artifacts (RT-OUT-3)
        # and same-byte rewrites (RT-OUT-2). Applies to both created and updated.
        if not changed_since(p, output_snaps):
            return False
        # "updated" additionally requires the file to have existed pre-run.
        if o.expect == "updated" and output_snaps.get(p) is None:
            return False
        # Semantic validity (G1): a declared validator must pass on the bytes.
        if o.validator is not None and not o.validator.validate(content).ok:
            return False
        return True

    def _return_properties(self) -> dict[str, Any]:
        # Advertise each declared return's JSON type so the model sends the right
        # type (e.g. integer, not "4"). Without this the model guesses and a strict
        # type_hint check (correctly, per RT-RET-2) rejects mistyped values forever.
        return {r.name: _json_schema_for_hint(r.type_hint) for r in self.c.returns}

    def tool_defs(self, *, host_tool_calls: bool):
        from tilldone.core.tools import ToolDef
        if not host_tool_calls:
            return []
        return_schema: dict[str, Any] = {
            "type": "object",
            "properties": self._return_properties(),
            "additionalProperties": False,
        }
        required = [r.name for r in self.c.returns if r.required]
        if required:
            return_schema["required"] = required
        return [
            ToolDef("return", "Submit the declared return values using their exact names and types.",
                    return_schema, kind="contract", namespace="contract"),
            ToolDef("todo", "Mark a declared todo complete.",
                    {"type": "object",
                     "properties": {"name": {"type": "string"}, "note": {"type": "string"}},
                     "required": ["name"]},
                    kind="contract", namespace="contract"),
            ToolDef("finish", "Declare the task complete.",
                    {"type": "object", "properties": {"note": {"type": "string"}}},
                    kind="contract", namespace="contract"),
        ]

    def output_schema(self):
        return {
            "type": "object",
            "properties": {
                "returns": {
                    "type": "object",
                    "properties": self._return_properties(),
                    "required": [r.name for r in self.c.returns if r.required],
                },
                "todos_done": {"type": "array", "items": {"type": "string"}},
                "finished": {"type": "boolean"},
            },
            "required": ["returns", "todos_done", "finished"],
            "additionalProperties": False,
        }

    def apply_structured(self, state: ContractState, output) -> None:
        """Ingest the structured-output channel into the SAME ContractState.

        A return is recorded ONLY if its name is declared AND its value matches
        the declared type_hint (defeats RT-RET-3 / RT-STR-1 via the structured
        path). Undeclared todo names are ignored (evaluate only checks declared
        names). This mirrors the host-tool path's contract handlers.
        """
        if not output:
            return
        declared_returns = {r.name: r for r in self.c.returns}
        for name, value in (output.get("returns") or {}).items():
            r = declared_returns.get(name)
            if r is None:
                continue
            if not matches_type_hint(value, r.type_hint):
                continue
            state.returns[name] = value
        declared_todos = {t.name for t in self.c.todos}
        for name in (output.get("todos_done") or []):
            if name in declared_todos:
                state.todos_done.setdefault(name, "")
        if output.get("finished"):
            state.finished = True


_KNOWN_HINTS = {"bool", "int", "float", "str", "json", "list", "array", "dict", "object"}


def is_known_type_hint(hint: str | None) -> bool:
    if hint is None:
        return True
    h = hint.strip().lower()
    if h in _KNOWN_HINTS:
        return True
    if h.startswith("list[") and h.endswith("]"):
        return is_known_type_hint(h[5:-1].strip())
    return False


def matches_type_hint(value: Any, hint: str | None) -> bool:
    """True iff value conforms to the declared hint. An unknown/garbled hint
    returns False (never silently True) — defeats RT-RET-1."""
    if hint is None:
        return True
    h = hint.strip().lower()
    if h == "bool":
        return isinstance(value, bool)
    if h == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if h == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if h == "str":
        return isinstance(value, str)
    if h in ("list", "array"):
        return isinstance(value, list)
    if h in ("dict", "object"):
        return isinstance(value, dict)
    if h == "json":
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False
    if h.startswith("list[") and h.endswith("]"):
        if not isinstance(value, list):
            return False
        inner = h[5:-1].strip()
        if not is_known_type_hint(inner):
            return False
        return all(matches_type_hint(v, inner) for v in value)
    return False  # unknown hint -> reject


def _json_schema_for_hint(hint: str | None) -> dict[str, Any]:
    """Map a return type_hint to a JSON-Schema fragment used to *guide* the model
    when emitting return values. ``json`` / unknown / None -> unconstrained ({})."""
    h = (hint or "").strip().lower()
    if h == "bool":
        return {"type": "boolean"}
    if h == "int":
        return {"type": "integer"}
    if h == "float":
        return {"type": "number"}
    if h == "str":
        return {"type": "string"}
    if h in ("dict", "object"):
        return {"type": "object"}
    if h in ("list", "array") or (h.startswith("list[") and h.endswith("]")):
        return {"type": "array"}
    return {}
