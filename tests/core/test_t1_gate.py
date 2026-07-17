"""Unit tests for the T1 gate pure helper (AR-10).

These tests exercise ``gated_run_is_theater`` in isolation from pytest machinery.
The function lives in the repo-root conftest so we import it directly — the repo
root is on sys.path because pytest adds rootdir to sys.path automatically.
"""
from __future__ import annotations

import conftest  # repo-root conftest is importable when rootdir is on sys.path


def test_gate_off_always_false() -> None:
    """When the gate is off, the theater check is always False regardless of counts."""
    assert conftest.gated_run_is_theater(
        gate_on=False,
        integration_collected=5,
        integration_ran=0,
    ) is False


def test_gate_on_collected_nonzero_ran_zero_is_theater() -> None:
    """Gate on + tests collected + none ran → True (this IS theater)."""
    assert conftest.gated_run_is_theater(
        gate_on=True,
        integration_collected=3,
        integration_ran=0,
    ) is True


def test_gate_on_some_ran_not_theater() -> None:
    """Gate on but at least one integration test ran → not theater."""
    assert conftest.gated_run_is_theater(
        gate_on=True,
        integration_collected=3,
        integration_ran=2,
    ) is False


def test_gate_on_collected_zero_not_theater() -> None:
    """Gate on but zero integration tests collected (deselected) → not theater."""
    assert conftest.gated_run_is_theater(
        gate_on=True,
        integration_collected=0,
        integration_ran=0,
    ) is False
