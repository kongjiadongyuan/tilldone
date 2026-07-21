"""Unit tests for Codex exec's unbounded newline framing."""

from __future__ import annotations

import asyncio

from tilldone.backends.codex_exec import _iter_lines


async def test_iter_lines_reads_one_megabyte_line() -> None:
    # Direct StreamReader iteration uses readline(), whose default 64 KiB limit rejects this.
    payload = b"x" * (1024 * 1024)
    stream = asyncio.StreamReader()
    stream.feed_data(payload + b"\n")
    stream.feed_eof()

    lines = [line async for line in _iter_lines(stream)]

    assert lines == [payload]
    assert len(lines[0]) == 1024 * 1024


async def test_iter_lines_handles_newline_at_read_block_boundary() -> None:
    first = b"a" * 65536
    stream = asyncio.StreamReader()
    stream.feed_data(first + b"\nsecond\n")
    stream.feed_eof()

    lines = [line async for line in _iter_lines(stream)]

    assert lines == [first, b"second"]


async def test_iter_lines_yields_eof_residual_without_newline() -> None:
    stream = asyncio.StreamReader()
    stream.feed_data(b"complete\nunterminated")
    stream.feed_eof()

    lines = [line async for line in _iter_lines(stream)]

    assert lines == [b"complete", b"unterminated"]
