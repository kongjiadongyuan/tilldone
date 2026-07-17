"""Codex error-code classification (RT-CFG-4): capacity/5xx/rate-limit -> transient (retry)."""

from __future__ import annotations

from tilldone.backends.codex_exec import _classify_codex_error


def test_capacity_and_unavailable_are_transient():
    assert _classify_codex_error("Selected model is at capacity. Please try a different model.") == "transient"
    assert _classify_codex_error("service_unavailable_error") == "transient"
    assert _classify_codex_error("HTTP 503 from upstream") == "transient"
    assert _classify_codex_error("429 rate limit exceeded") == "transient"
    assert _classify_codex_error("model is overloaded, try again") == "transient"
    assert _classify_codex_error("request timed out") == "transient"


def test_genuine_errors_are_fatal():
    assert _classify_codex_error("invalid request: schema mismatch") == "fatal"
    assert _classify_codex_error("authentication failed") == "fatal"
    assert _classify_codex_error("model error") == "fatal"
