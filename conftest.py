"""Repo-root conftest.py — T1 gate: "skip != pass" enforcement.

When a gate env var (TILLDONE_CLAUDE_E2E or TILLDONE_CODEX_E2E) is set to "1"
AND integration-marked tests were collected in the selected set
AND every single one of them was skipped (zero actually ran),
pytest exits with a failing status code — because CI green from 100% skips is theater.

When the gate is off, or when integration was explicitly deselected (e.g. -m "not
integration"), this conftest is a complete no-op and must not affect normal unit runs.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Pure helper — importable and unit-testable independently of pytest machinery
# ---------------------------------------------------------------------------

def gated_run_is_theater(
    *,
    gate_on: bool,
    integration_collected: int,
    integration_ran: int,
) -> bool:
    """Return True iff a gated integration run produced zero actual executions.

    Parameters
    ----------
    gate_on:
        True when at least one of the gating env vars is "1".
    integration_collected:
        Number of integration-marked items that were *selected* for this run
        (i.e. present in the collection after marker deselection).
    integration_ran:
        Number of those items that produced a non-skipped ``call`` phase report.

    Returns True only when all three conditions hold simultaneously:
      - the gate is active (caller wanted live tests),
      - at least one integration test was collected (not deselected),
      - and none of them actually ran (all were skipped).
    """
    return gate_on and integration_collected > 0 and integration_ran == 0


# ---------------------------------------------------------------------------
# Gate detector
# ---------------------------------------------------------------------------

def gate_on() -> bool:
    """Return True if any live-backend gate env is set to '1'."""
    return (
        os.environ.get("TILLDONE_CLAUDE_E2E") == "1"
        or os.environ.get("TILLDONE_CODEX_E2E") == "1"
    )


# ---------------------------------------------------------------------------
# Pytest hook state — module-level counters (one conftest instance per session)
# ---------------------------------------------------------------------------

_integration_collected: int = 0
_integration_ran: int = 0

_REPO_ROOT = Path(__file__).resolve().parent
_ARCHIVED_APP_SERVER_RECIPE = Path("tests") / "recipes" / "test_recipe_app_server.py"


def _is_archived_app_server_path(path: Path) -> bool:
    """True for archived v2 app-server test assets excluded from default collection."""
    try:
        rel = path.resolve().relative_to(_REPO_ROOT)
    except ValueError:
        rel = path
    parts = rel.parts
    return (
        parts[:3] == ("tests", "backends", "app_server")
        or rel == _ARCHIVED_APP_SERVER_RECIPE
    )


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--include-archived-app-server",
        action="store_true",
        default=False,
        help="include archived v2 codex app-server tests in collection",
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    if config.getoption("--include-archived-app-server", default=False):
        return False
    return _is_archived_app_server_path(collection_path)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Count selected integration items after collection + deselection."""
    global _integration_collected
    count = 0
    for item in session.items:
        if item.get_closest_marker("integration") is not None:
            count += 1
    _integration_collected = count


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Count integration items that completed the *call* phase without being skipped."""
    global _integration_ran
    if (
        report.when == "call"
        and not report.skipped
        and report.nodeid  # safety: must be a real node
    ):
        # We only have the report here; check the item's markers via the stored
        # nodeid — but we don't have a direct handle to the item.  Instead we
        # rely on the marker being present on the report via keywords, which
        # pytest populates from the item's markers and pytestmark.
        if "integration" in report.keywords:
            _integration_ran += 1


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """T3 flaky governance: auto-rerun live (integration) tests up to 2x. LLM non-determinism
    and transient gateway model-capacity make even deterministic fixtures occasionally flaky;
    rerun-on-failure institutionalizes the M-of-N policy instead of hand-stabilizing. Unit tests
    are NEVER retried — they must be deterministic (a rerun would mask a real bug)."""
    for item in items:
        if item.get_closest_marker("integration") is not None:
            item.add_marker(pytest.mark.flaky(reruns=2, reruns_delay=3))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail the session when a gated run produced only skips (T1 theater check)."""
    if gated_run_is_theater(
        gate_on=gate_on(),
        integration_collected=_integration_collected,
        integration_ran=_integration_ran,
    ):
        print(
            "\nT1: gated suite requested but all integration tests SKIPPED — "
            f"missing key? run-count=0 collected={_integration_collected}"
        )
        # Exit code 1 = tests failed; overrides any passing status.
        session.exitstatus = 1
