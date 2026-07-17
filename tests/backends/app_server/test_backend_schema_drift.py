"""Version/protocol floor + schema-drift fail-fast for the archived app-server backend.

Without a pin + drift check, a codex upgrade that **renames or removes a method**
the backend drives would silently mis-behave instead of failing fast.

Layered gate:

* **(a) CLI version floor** — ``codex --version`` must report a version ≥ the verified
  floor. Runs when ``codex`` is on PATH, else SKIPs (SKIP != PASS).

* **(b) schema-drift fail-fast** — every JSON-RPC method the backend relies on
  (client requests it issues + the ``item/tool/call`` server-request) must be present
  in the vendored protocol schema method enums (``ClientRequest.json`` +
  ``ServerRequest.json``). Absence → :class:`SchemaDriftError`.

These tests drive the drift checker against the current vendored schema (backend
method-set fully covered today) and against a synthetically-mutated schema to prove
the fail-fast fires on a removed method.
"""

from __future__ import annotations

import shutil

import pytest

from tilldone.backends.app_server.backend import CodexAppServerBackend
from tilldone.backends.app_server.schema_drift import (
    BACKEND_DRIVEN_CLIENT_METHODS,
    BACKEND_DRIVEN_SERVER_METHODS,
    CODEX_VERSION_FLOOR,
    SchemaDriftError,
    assert_codex_version_floor,
    check_schema_drift,
    load_schema_methods,
    parse_codex_version,
    reference_schema_dir,
)

# Live gate for the (a) CLI version-floor layer: a real skipif (counted by conftest's T1
# run-count gate), and it actually RUNS when codex is on PATH.
_CODEX_BIN = shutil.which("codex")
_skip_no_codex = pytest.mark.skipif(
    _CODEX_BIN is None, reason="codex binary not on PATH (CLI version-floor canary)")


# --------------------------------------------------------------------------- #
# (b) schema-drift: the offline, deterministic core                            #
# --------------------------------------------------------------------------- #

def test_reference_schema_dir_exists_and_has_method_enums():
    """The vendored reference schema is present and its ClientRequest/ServerRequest
    carry machine-readable method enums (the protocol ground truth the gate stands on)."""
    schema_dir = reference_schema_dir()
    assert schema_dir.is_dir(), f"reference schema dir missing: {schema_dir}"
    client, server = load_schema_methods(schema_dir)
    # The protocol is large (>100 client requests live); just sanity-floor it.
    assert len(client) > 50
    assert "initialize" in client
    assert "thread/start" in client
    assert "turn/start" in client
    # item/tool/call is the sole server→client request the backend depends on.
    assert "item/tool/call" in server


def test_backend_driven_methods_are_all_present_in_current_schema():
    """CANARY: every method the backend RELIES ON is present in the CURRENT vendored
    schema → no drift today. (If a future schema refresh removes/renames one of these,
    this test goes red — exactly the G3 trip wire.)"""
    report = check_schema_drift()  # defaults to the vendored reference dir
    assert report["drifted"] == [], f"unexpected schema drift: {report['drifted']}"
    assert report["ok"] is True
    # The report enumerates what it checked (transparency).
    assert set(report["client_checked"]) == set(BACKEND_DRIVEN_CLIENT_METHODS)
    assert set(report["server_checked"]) == set(BACKEND_DRIVEN_SERVER_METHODS)


def test_backend_driven_client_methods_cover_the_real_wire_methods():
    """The backend-driven client set must include every method the backend's own code
    issues — handshake + thread/turn lifecycle + rich control + skills (so the canary
    actually guards the surface the backend uses, not a stale subset)."""
    expected = {
        "initialize", "thread/start", "thread/resume", "turn/start", "turn/interrupt",
        "thread/archive", "turn/steer", "thread/fork", "thread/rollback",
        "skills/list", "skills/extraRoots/set",
    }
    assert expected <= set(BACKEND_DRIVEN_CLIENT_METHODS)


def test_check_schema_drift_passes_for_current_schema_returns_report():
    """check_schema_drift returns a structured report (does NOT raise) when there is no
    drift, so a caller can log what was verified."""
    report = check_schema_drift()
    assert report["ok"] is True
    assert isinstance(report["client_present"], list)
    assert isinstance(report["server_present"], list)


def test_schema_drift_fail_fast_on_removed_client_method(tmp_path):
    """THE fail-fast: a schema in which a backend-driven client method is ABSENT (a codex
    upgrade removed/renamed it) → SchemaDriftError naming the missing method. We build a
    minimal mutated schema dir with ``thread/start`` deleted from the enum."""
    import json

    src = reference_schema_dir()
    # Copy the two method-bearing files, then DELETE thread/start from ClientRequest.
    (tmp_path / "ClientRequest.json").write_text(
        _drop_method((src / "ClientRequest.json").read_text(), "thread/start"),
        encoding="utf-8")
    shutil.copy(src / "ServerRequest.json", tmp_path / "ServerRequest.json")

    with pytest.raises(SchemaDriftError) as ei:
        check_schema_drift(tmp_path)
    msg = str(ei.value)
    assert "thread/start" in msg  # names the drifted method
    assert "drift" in msg.lower()


def test_schema_drift_fail_fast_on_removed_server_request(tmp_path):
    """The server-request half: if ``item/tool/call`` vanishes from ServerRequest (the
    backend's tools mode depends on it), that is drift too."""
    import json

    src = reference_schema_dir()
    shutil.copy(src / "ClientRequest.json", tmp_path / "ClientRequest.json")
    (tmp_path / "ServerRequest.json").write_text(
        _drop_method((src / "ServerRequest.json").read_text(), "item/tool/call"),
        encoding="utf-8")

    with pytest.raises(SchemaDriftError) as ei:
        check_schema_drift(tmp_path)
    assert "item/tool/call" in str(ei.value)


def _drop_method(schema_text: str, method: str) -> str:
    """Return ``schema_text`` (a ClientRequest/ServerRequest JSON) with ``method`` removed
    from whatever ``method.enum`` it lives in — simulating a renamed/removed protocol method."""
    import json

    doc = json.loads(schema_text)
    for variant in doc.get("oneOf", []):
        enum = variant.get("properties", {}).get("method", {}).get("enum")
        if enum and method in enum:
            enum.remove(method)
    return json.dumps(doc)


# --------------------------------------------------------------------------- #
# (a) CLI version-floor: parsing (offline) + live assertion                    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,expected", [
    ("codex-cli 0.139.0", (0, 139, 0)),
    ("codex-cli 0.140.2", (0, 140, 2)),
    ("0.139.0", (0, 139, 0)),
    ("codex 1.0.0\n", (1, 0, 0)),
])
def test_parse_codex_version_extracts_tuple(text, expected):
    assert parse_codex_version(text) == expected


def test_parse_codex_version_rejects_garbage():
    with pytest.raises(ValueError):
        parse_codex_version("no version here")


def test_version_floor_comparison_is_semantic():
    """The floor check is a semantic version compare, not a string compare (so 0.140 > 0.139
    and 0.139.0 == floor passes, while 0.138.x fails)."""
    floor = CODEX_VERSION_FLOOR
    assert assert_codex_version_floor(f"codex-cli {floor[0]}.{floor[1]}.{floor[2]}") is True
    assert assert_codex_version_floor("codex-cli 0.140.0") is True
    with pytest.raises(SchemaDriftError):
        assert_codex_version_floor("codex-cli 0.138.99")


@_skip_no_codex
def test_live_codex_version_meets_floor():
    """LIVE (a): the codex on PATH reports a version ≥ the verified floor. Runs for real
    when codex is installed (this box: codex-cli 0.139.0); SKIPs otherwise (SKIP != PASS)."""
    import subprocess

    out = subprocess.run([_CODEX_BIN, "--version"], capture_output=True, text=True,
                         check=True).stdout
    assert assert_codex_version_floor(out) is True


# --------------------------------------------------------------------------- #
# build-on probe_capabilities(): the offline gate now reports schema-drift      #
# --------------------------------------------------------------------------- #

def test_probe_capabilities_reports_no_schema_drift_for_tools_mode():
    """RT-CAP-1 offline gate (probe_capabilities) is EXTENDED with the G3 drift check:
    it reports schema_drift_ok=True against the current vendored schema, and lists the
    methods this mode drives that were schema-verified."""
    probe = CodexAppServerBackend(mode="tools").probe_capabilities()
    assert probe["schema_drift_ok"] is True
    # tools mode drives item/tool/call (the server-request); it must be in the verified set.
    assert "item/tool/call" in probe["schema_verified_server_methods"]
    assert "thread/start" in probe["schema_verified_client_methods"]


def test_probe_capabilities_structured_mode_also_drift_clean():
    probe = CodexAppServerBackend(mode="structured").probe_capabilities()
    assert probe["schema_drift_ok"] is True
