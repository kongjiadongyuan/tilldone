"""Version/protocol floor + schema-drift fail-fast for the archived app-server backend.

The ``codex app-server`` protocol is experimental. A codex upgrade that **renames or
removes a method** the backend drives would, with no gate, fail at a confusing runtime
``-32601`` (or silently mis-behave) instead of failing fast with a clear "protocol
drifted" signal. This module is the layered gate:

* **(a) CLI version floor** — :func:`assert_codex_version_floor` parses
  ``codex --version`` and rejects anything below :data:`CODEX_VERSION_FLOOR`
  (``0.139.0``, the floor this backend was developed against).

* **(b) schema-drift fail-fast** — :func:`check_schema_drift` validates that every
  JSON-RPC method the backend relies on is present in a vendored protocol schema's
  method enums (``ClientRequest.json`` for client requests, ``ServerRequest.json`` for
  the ``item/tool/call`` server-request tools mode depends on). A backend-driven method
  ABSENT from the schema is drift → :class:`SchemaDriftError`.

Grounding: :data:`BACKEND_DRIVEN_CLIENT_METHODS` and
:data:`BACKEND_DRIVEN_SERVER_METHODS` are assembled from the backend's own method
constants (``backend._KNOWN_CLIENT_METHODS`` + rich-control / skills / tool-call
names), so the check tracks the surface the backend code actually uses.

This module does I/O only on vendored schema files + (in the version layer) string
parsing. It imports only sibling ``backend`` constants, never ``tilldone.core``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tilldone.backends.app_server import backend as _be

#: codex-cli version floor this backend was developed against. A codex below this
#: floor predates the protocol surface the backend drives and is rejected.
CODEX_VERSION_FLOOR: tuple[int, int, int] = (0, 139, 0)


class SchemaDriftError(RuntimeError):
    """Raised when the codex protocol drifted out from under the backend.

    Two triggers: a backend-driven JSON-RPC method is ABSENT from the reference schema
    (the method was renamed/removed), or ``codex --version`` is below the verified floor.
    Either is a fail-fast — the backend must not run against a protocol it can no longer
    assume (G3).
    """


# -- (b) schema-drift: backend-driven method-set, grounded in the backend ------------
#
# Assembled from the backend's OWN constants so the canary tracks the real wire surface:
#   * client requests: the offline-gate's known set (_KNOWN_CLIENT_METHODS — handshake +
#     thread/turn lifecycle + archive) PLUS the rich-control + skills methods the backend
#     issues from its capability-scoped control plane (_STEER/_FORK/_ROLLBACK/_SKILLS_*).
#   * server requests: the sole server→client request the backend depends on in tools
#     mode (_TOOL_CALL_METHOD == "item/tool/call").
#: Every client-request method the backend issues (grounded in backend constants).
BACKEND_DRIVEN_CLIENT_METHODS: tuple[str, ...] = tuple(sorted(
    set(_be._KNOWN_CLIENT_METHODS)
    | {
        _be._STEER_METHOD,
        _be._FORK_METHOD,
        _be._ROLLBACK_METHOD,
        _be._SKILLS_LIST_METHOD,
        _be._SKILLS_EXTRA_ROOTS_SET_METHOD,
    }
))
#: Every server→client request method the backend depends on (tools mode: item/tool/call).
BACKEND_DRIVEN_SERVER_METHODS: tuple[str, ...] = (_be._TOOL_CALL_METHOD,)


def reference_schema_dir() -> Path:
    """Absolute path to the vendored protocol schema directory under the repo root.

    Resolved relative to the repo root (this file lives at
    ``src/tilldone/backends/app_server/schema_drift.py`` → repo root is ``parents[4]``),
    so it works regardless of the process CWD.
    """
    return Path(__file__).resolve().parents[4] / "reference" / "codex-app-server-schema"


def _method_enum(schema_doc: dict) -> set[str]:
    """Collect every ``method.enum`` value across a ``oneOf`` request schema.

    ``ClientRequest.json`` / ``ServerRequest.json`` are ``{"oneOf": [variant, ...]}``
    where each variant pins ``properties.method.enum = ["<method>"]``. We union all the
    enums into the authoritative set of methods the protocol defines.
    """
    methods: set[str] = set()
    for variant in schema_doc.get("oneOf", []):
        enum = variant.get("properties", {}).get("method", {}).get("enum")
        if isinstance(enum, list):
            methods.update(str(m) for m in enum)
    return methods


def load_schema_methods(schema_dir: Path | None = None) -> tuple[set[str], set[str]]:
    """Load ``(client_methods, server_methods)`` from the reference protocol schema.

    ``client_methods`` is every method enumerated by ``ClientRequest.json``;
    ``server_methods`` is every method enumerated by ``ServerRequest.json``. These are
    the authoritative protocol method-sets the drift check measures the backend against.
    """
    schema_dir = schema_dir or reference_schema_dir()
    client_doc = json.loads((schema_dir / "ClientRequest.json").read_text(encoding="utf-8"))
    server_doc = json.loads((schema_dir / "ServerRequest.json").read_text(encoding="utf-8"))
    return _method_enum(client_doc), _method_enum(server_doc)


def check_schema_drift(schema_dir: Path | None = None) -> dict:
    """Validate the backend's method-set against the reference schema. Fail fast on drift.

    Every method in :data:`BACKEND_DRIVEN_CLIENT_METHODS` must be present in the schema's
    ClientRequest enum, and every method in :data:`BACKEND_DRIVEN_SERVER_METHODS` in its
    ServerRequest enum. Any backend-driven method that is ABSENT (a rename/removal) raises
    :class:`SchemaDriftError` naming the drifted method(s). On success returns a structured
    report (``ok=True``, what was checked, what was present) for logging.
    """
    client_schema, server_schema = load_schema_methods(schema_dir)

    client_missing = [m for m in BACKEND_DRIVEN_CLIENT_METHODS if m not in client_schema]
    server_missing = [m for m in BACKEND_DRIVEN_SERVER_METHODS if m not in server_schema]
    drifted = client_missing + server_missing

    report = {
        "ok": not drifted,
        "drifted": drifted,
        "client_checked": list(BACKEND_DRIVEN_CLIENT_METHODS),
        "server_checked": list(BACKEND_DRIVEN_SERVER_METHODS),
        "client_present": [m for m in BACKEND_DRIVEN_CLIENT_METHODS if m in client_schema],
        "server_present": [m for m in BACKEND_DRIVEN_SERVER_METHODS if m in server_schema],
    }
    if drifted:
        raise SchemaDriftError(
            "codex app-server protocol DRIFT detected — method(s) the backend drives are "
            f"ABSENT from the reference schema (renamed/removed?): {drifted}. The backend "
            "would issue a request the server no longer understands; refusing to run "
            "(G3 schema-drift fail-fast). Refresh the vendored protocol schema and "
            "reconcile the backend's method constants.")
    return report


# -- (a) CLI version floor -----------------------------------------------------------

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def parse_codex_version(text: str) -> tuple[int, int, int]:
    """Parse a ``major.minor.patch`` tuple out of ``codex --version`` output.

    Accepts ``"codex-cli 0.139.0"``, a bare ``"0.139.0"``, trailing newline, etc. Raises
    ``ValueError`` if no ``N.N.N`` triple is present (so garbage is never mistaken for a
    version that meets the floor).
    """
    m = _VERSION_RE.search(text or "")
    if not m:
        raise ValueError(f"could not parse a codex version from {text!r}")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def assert_codex_version_floor(version_text: str,
                               floor: tuple[int, int, int] = CODEX_VERSION_FLOOR) -> bool:
    """Assert the parsed codex version is ≥ ``floor`` (semantic, not string, compare).

    Returns ``True`` when the floor is met; raises :class:`SchemaDriftError` (a fail-fast,
    same family as a method drift — the toolchain is below the verified protocol) when it
    is not.
    """
    version = parse_codex_version(version_text)
    if version < floor:
        raise SchemaDriftError(
            f"codex version {'.'.join(map(str, version))} is below the verified floor "
            f"{'.'.join(map(str, floor))} — the app-server wire facts were validated "
            "against the floor; refusing to run on an older protocol (G3 version floor)")
    return True
