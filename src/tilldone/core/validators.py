from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str = ""


@runtime_checkable
class OutputValidator(Protocol):
    """Validates output-file bytes.

    Contract: validators are PURE and MUST NOT raise. On any failure or
    internal error they return ValidationResult(ok=False, reason=...).
    They operate on raw bytes (never assume a text encoding).
    """

    def validate(self, content: bytes) -> ValidationResult: ...


@dataclass(frozen=True)
class JsonValidator:
    """Content parses as JSON. Half-written / empty / garbage -> not ok."""

    def validate(self, content: bytes) -> ValidationResult:
        try:
            json.loads(content)
        except Exception as e:  # JSONDecodeError, UnicodeDecodeError, ...
            return ValidationResult(False, f"invalid JSON: {e}")
        return ValidationResult(True)


_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


@dataclass(frozen=True)
class JsonSchemaValidator:
    """Minimal, dependency-free JSON-Schema check: top-level ``type``,
    object ``required`` keys, and one-level property ``type``s.

    For full JSON-Schema, wrap ``jsonschema.validate`` in a CallableValidator.
    """

    schema: Mapping[str, Any]

    def validate(self, content: bytes) -> ValidationResult:
        try:
            value = json.loads(content)
        except Exception as e:
            return ValidationResult(False, f"invalid JSON: {e}")
        return self._check(value, self.schema, "$")

    def _check(self, value: Any, schema: Mapping[str, Any], path: str) -> ValidationResult:
        t = schema.get("type")
        if t is not None:
            py = _JSON_TYPES.get(t)
            if py is None:
                return ValidationResult(False, f"{path}: unknown schema type {t!r}")
            # bool is a subclass of int; integer/number must reject booleans
            if t in ("integer", "number") and isinstance(value, bool):
                return ValidationResult(False, f"{path}: expected {t}, got boolean")
            if not isinstance(value, py):
                return ValidationResult(False, f"{path}: expected {t}")
        for key in schema.get("required", []):
            if not isinstance(value, dict) or key not in value:
                return ValidationResult(False, f"{path}: missing required key {key!r}")
        if isinstance(value, dict):
            for key, subschema in schema.get("properties", {}).items():
                if key in value:
                    r = self._check(value[key], subschema, f"{path}.{key}")
                    if not r.ok:
                        return r
        return ValidationResult(True)


@dataclass(frozen=True)
class LineCountValidator:
    """Line count (by newline; a single trailing newline is not an extra line)."""

    min: int | None = None
    max: int | None = None
    exact: int | None = None

    def validate(self, content: bytes) -> ValidationResult:
        n = content.count(b"\n")
        if content and not content.endswith(b"\n"):
            n += 1
        if self.exact is not None and n != self.exact:
            return ValidationResult(False, f"expected exactly {self.exact} lines, got {n}")
        if self.min is not None and n < self.min:
            return ValidationResult(False, f"expected >= {self.min} lines, got {n}")
        if self.max is not None and n > self.max:
            return ValidationResult(False, f"expected <= {self.max} lines, got {n}")
        return ValidationResult(True)


@dataclass(frozen=True)
class MagicBytesValidator:
    """Content starts with a fixed byte prefix (e.g. a file magic number)."""

    prefix: bytes

    def validate(self, content: bytes) -> ValidationResult:
        if content.startswith(self.prefix):
            return ValidationResult(True)
        return ValidationResult(False, f"missing magic-byte prefix {self.prefix!r}")


@dataclass(frozen=True)
class RegexBytesValidator:
    """Regex match on raw bytes. search=True matches anywhere; False = fullmatch."""

    pattern: bytes
    search: bool = True

    def validate(self, content: bytes) -> ValidationResult:
        try:
            rx = re.compile(self.pattern)
        except re.error as e:
            return ValidationResult(False, f"bad regex: {e}")
        m = rx.search(content) if self.search else rx.fullmatch(content)
        if m is None:
            return ValidationResult(False, f"regex {self.pattern!r} did not match")
        return ValidationResult(True)


@dataclass(frozen=True)
class CallableValidator:
    """Wrap an arbitrary ``Callable[[bytes], bool]`` (the universal escape hatch)."""

    fn: Callable[[bytes], bool]
    description: str = ""

    def validate(self, content: bytes) -> ValidationResult:
        try:
            ok = bool(self.fn(content))
        except Exception as e:
            return ValidationResult(False, f"validator raised: {e}")
        return ValidationResult(ok, "" if ok else (self.description or "custom validator failed"))


@dataclass(frozen=True)
class AllOf:
    """AND-combine validators; returns the first failure (short-circuits)."""

    validators: Sequence[OutputValidator]

    def validate(self, content: bytes) -> ValidationResult:
        for v in self.validators:
            r = v.validate(content)
            if not r.ok:
                return r
        return ValidationResult(True)
