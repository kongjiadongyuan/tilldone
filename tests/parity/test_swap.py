"""Backend-swap parity: the same task must satisfy Claude == CodexBackend.

Current product surface exposes two callers' choices:

* ``ClaudeBackend`` over ``claude-agent-sdk``.
* ``CodexBackend`` over ``codex exec --json`` plus TillDone MCP injection.

The archived app-server driver is intentionally out of this mainline parity suite.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from tilldone.backends.claude import ClaudeBackend
from tilldone.backends.codex import CodexBackend
from tilldone.core.contract import (
    CompletionContract,
    OutputExpectation,
    ReturnExpectation,
    TodoExpectation,
)
from tilldone.core.loop import CorePolicy, run_task
from tilldone.core.spec import ContextDirView
from tilldone.core.validators import CallableValidator
from tilldone.testing.parity import ParityVerdict, assert_parity

pytestmark = pytest.mark.integration

_CLAUDE = os.environ.get("TILLDONE_CLAUDE_E2E") == "1"
_CODEX = os.environ.get("TILLDONE_CODEX_E2E") == "1"
_BOTH = _CLAUDE and _CODEX

_skip = pytest.mark.skipif(
    not _BOTH,
    reason="parity needs BOTH TILLDONE_CLAUDE_E2E=1 and TILLDONE_CODEX_E2E=1",
)

_VALUE = 4
_MARKER = "PEGASUS-7Q"


def _claude():
    return ClaudeBackend()


def _codex():
    return CodexBackend(extra_config=['model_reasoning_effort="low"'])


def _build_value_and_output(d: Path):
    report = d / "report.txt"
    contract = CompletionContract(
        outputs=(OutputExpectation("r", report, "created",
                                   validator=CallableValidator(lambda b: b.strip() == b"DONE")),),
        returns=(ReturnExpectation("answer", "int", True),),
        todos=(),
        require_finish=True,
    )
    prompt = (
        f"Complete ALL of these steps; the task is NOT done until every tool call is made:\n"
        f"1. Create a file at {report} whose contents are exactly: DONE\n"
        f"2. Call the `return` tool with argument answer = 4 — it MUST be the JSON integer 4, "
        f"not the string \"4\".\n"
        f"3. Call the `finish` tool.\n"
        f"Writing the file alone does NOT complete the task — you must also call `return` and `finish`."
    )
    return contract, prompt, report


def _build_ref_real_read(d: Path):
    ctx = d / "ctx"
    ctx.mkdir(parents=True)
    (ctx / "data.txt").write_text(f"intro line\nMARKER={_MARKER}\ntail line\n")
    contract = CompletionContract(
        outputs=(),
        returns=(ReturnExpectation("marker", "str", True),),
        todos=(),
        require_finish=True,
    )
    prompt = (
        "Read the file at @ref:data/data.txt. It contains a line of the form "
        "MARKER=<value>. Call return with marker set to exactly that <value> "
        "string. Then call finish."
    )
    return contract, prompt, [ContextDirView(id="data", path=ctx)]


def _build_multi_todo(_d: Path):
    contract = CompletionContract(
        outputs=(),
        returns=(),
        todos=(TodoExpectation("alpha"), TodoExpectation("beta")),
        require_finish=True,
    )
    prompt = (
        "Call the todo tool twice — once with name set to alpha, once with name set "
        "to beta — then call the finish tool."
    )
    return contract, prompt


def _run(contract, prompt, backend, cwd, context_dirs=()):
    async def go():
        return await run_task(
            contract,
            backend,
            prompt=prompt,
            cwd=cwd,
            policy=CorePolicy(max_rounds=6),
            context_dirs=context_dirs,
        )

    return asyncio.run(go())


def _verdict(
    backend: str,
    *,
    return_name: str | None,
    value,
    output_ok: bool = True,
    todos_done=frozenset(),
) -> ParityVerdict:
    return ParityVerdict(
        backend=backend,
        returns={return_name: value} if return_name is not None else {},
        contract_satisfied=True,
        todos_done=frozenset(todos_done),
        finished=True,
        status="completed",
        error_code=None,
        output_ok=output_ok,
    )


@_skip
def test_parity_value_and_output(tmp_path):
    cd = tmp_path / "c"
    xd = tmp_path / "x"
    cd.mkdir()
    xd.mkdir()
    cc, cp, crep = _build_value_and_output(cd)
    xc, xp, xrep = _build_value_and_output(xd)

    c_res = _run(cc, cp, _claude(), cd)
    x_res = _run(xc, xp, _codex(), xd)

    assert_parity([
        _verdict("claude", return_name="answer", value=c_res,
                 output_ok=crep.read_bytes().strip() == b"DONE"),
        _verdict("codex", return_name="answer", value=x_res,
                 output_ok=xrep.read_bytes().strip() == b"DONE"),
    ])
    assert c_res == x_res == _VALUE
    assert crep.read_bytes().strip() == xrep.read_bytes().strip() == b"DONE"


@_skip
def test_parity_ref_real_read(tmp_path):
    cd = tmp_path / "c"
    xd = tmp_path / "x"
    cd.mkdir()
    xd.mkdir()
    cc, cp, ccd = _build_ref_real_read(cd)
    xc, xp, xcd = _build_ref_real_read(xd)

    c_res = _run(cc, cp, _claude(), cd, ccd)
    x_res = _run(xc, xp, _codex(), xd, xcd)

    assert_parity([
        _verdict("claude", return_name="marker", value=c_res),
        _verdict("codex", return_name="marker", value=x_res),
    ])
    assert c_res == x_res == _MARKER


@_skip
def test_parity_multi_todo_and_finish(tmp_path):
    cd = tmp_path / "c"
    xd = tmp_path / "x"
    cd.mkdir()
    xd.mkdir()
    cc, cp = _build_multi_todo(cd)
    xc, xp = _build_multi_todo(xd)

    c_res = _run(cc, cp, _claude(), cd)
    x_res = _run(xc, xp, _codex(), xd)

    assert_parity([
        _verdict("claude", return_name=None, value=None,
                 todos_done=frozenset({"alpha", "beta"})),
        _verdict("codex", return_name=None, value=None,
                 todos_done=frozenset({"alpha", "beta"})),
    ])
    assert c_res is None and x_res is None
