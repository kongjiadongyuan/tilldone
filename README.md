# TillDone

English | [简体中文](README.zh-CN.md)

> **Status: alpha** (`1.0.0a0`). The API may change before a stable release.

TillDone is a task-oriented agent runner and backend compatibility layer for Python. Define completion once, then run the same task through Claude Code or Codex.

A task is complete only when its declared outputs, return values, todos, and finish signal are satisfied. TillDone keeps this contract in a backend-independent core while each backend translates it to its native agent runtime. Same contract, different backend.

## Why TillDone

```text
CompletionContract + prompt + tools
                 |
           Flow / run_task
                 |
        ClaudeBackend | CodexBackend
```

Three boundaries matter:

- **Contract lives in core.** A backend cannot declare files complete on its own.
- **Backends only run a round.** They stream standard events and return a standard outcome.
- **Core decides whether to continue.** It re-evaluates the filesystem and contract-tool results after each round.

Core has zero runtime dependencies. Backend SDKs come in via extras.

## Installation

Requires **Python ≥ 3.11**.

```bash
# Claude Code backend
uv add "tilldone[claude]"

# Codex backend
uv add "tilldone[codex]"
```

Install both extras if you want either backend. Core stays dependency-free; `claude` pulls in `claude-agent-sdk`, `codex` pulls in `mcp`. `CodexBackend` also needs the `codex` CLI on `PATH` and authenticated per Codex’s own setup.

## Quickstart

A real completion contract: write a file **and** return a typed value.

```python
import asyncio
from pathlib import Path

from tilldone import (
    CompletionContract,
    Flow,
    OutputExpectation,
    ReturnExpectation,
)
from tilldone.backends.codex import CodexBackend


async def main() -> None:
    cwd = Path.cwd()
    report = cwd / "report.txt"

    contract = CompletionContract(
        outputs=(OutputExpectation("report", report, "created"),),
        returns=(ReturnExpectation("summary", "str", required=True),),
        todos=(),
        require_finish=True,
    )

    async with Flow(CodexBackend()) as flow:
        summary = await flow.run(
            contract,
            prompt=(
                f"Write a short report to {report}. "
                "Then call return with a string named summary, and call finish."
            ),
            cwd=cwd,
        )

    print(summary)


asyncio.run(main())
```

`Flow` is the high-level execution boundary around one backend and one active run at a time. On context exit it calls the backend’s `aclose()`.

Swap backends by changing the import and constructor only:

```python
from tilldone.backends.claude import ClaudeBackend

backend = ClaudeBackend()
```

or:

```python
from tilldone.backends.codex import CodexBackend

backend = CodexBackend()
```

Same contract, different backend.

## Completion contracts

Completion is evaluated from declared state, not inferred from the model’s final prose. Four kinds of conditions:

| Condition | API | Satisfied when |
| --- | --- | --- |
| File output | `OutputExpectation` | File exists, is non-empty, and changed relative to the pre-run snapshot; optional validator passes |
| Return value | `ReturnExpectation` | Agent calls built-in `return` with a declared name and a value matching `type_hint` |
| Todo | `TodoExpectation` | Agent calls built-in `todo` with a declared name |
| Finish | `require_finish=True` | Agent calls built-in `finish` |

Details that matter in practice:

- `expect="created"` — after the run, the file must be non-empty and changed vs. the pre-run snapshot.
- `expect="updated"` — same as created, and the file must already exist before the run.
- The evaluator does **not** trust backend `file_changed` events; filesystem content and a SHA-256 pre-snapshot are authoritative.
- Zero returns → `None`; one return → scalar; multiple returns → `dict`.
- Known `type_hint` values: `bool`, `int`, `float`, `str`, `json`, `list`/`array`, `dict`/`object`, and recursive `list[...]`.
- Unsatisfied contracts keep running (default max 6 contract rounds). Tune with `CorePolicy`.

Todos:

```python
from tilldone import CompletionContract, TodoExpectation

contract = CompletionContract(
    outputs=(),
    returns=(),
    todos=(TodoExpectation("check-tests"), TodoExpectation("write-summary")),
    require_finish=True,
)
```

Optional validators (import from `tilldone.core.validators`):

```python
from tilldone import OutputExpectation
from tilldone.core.validators import JsonValidator

output = OutputExpectation("result", Path("result.json").resolve(), "created", JsonValidator())
```

Also available: `JsonSchemaValidator`, `LineCountValidator`, `MagicBytesValidator`, `RegexBytesValidator`, `CallableValidator`, `AllOf`.

## Custom tools & MCP

Custom tools are async handlers in the host process. They take a `ToolCall` and return a `ToolResult`. Pass them to `Flow(custom_tools=...)` or `run_task(custom_tools=...)`. Claude exposes them via in-process MCP; Codex uses a TillDone-managed localhost MCP bridge.

```python
from tilldone import Flow, ToolCall, ToolResult


async def get_build_id(call: ToolCall) -> ToolResult:
    return ToolResult(content="build-42")


flow = Flow(
    backend,
    custom_tools={"get_build_id": get_build_id},
)
```

Tell the prompt which tool to call and how to use the result. The public map is name → handler only (no description/schema kwargs on this surface).

For Codex, you can also attach your own MCP servers:

```python
from tilldone.backends.codex import CodexBackend

backend = CodexBackend(
    extra_config=[
        'mcp_servers.files.url="http://127.0.0.1:7777/mcp"',
        'mcp_servers.files.bearer_token_env_var="FILES_MCP_TOKEN"',
    ]
)
```

`mcp_servers.tilldone.*` is reserved for the built-in bridge — overriding it fails fast. Prefer env vars for tokens; do not embed secrets in config strings.

## Context dirs & `@ref`

Named directories decouple prompts from machine-absolute paths. Syntax: `@ref:<id>/<relative-path>`.

```python
from pathlib import Path

from tilldone import ContextDirView, Flow

docs = ContextDirView(
    id="docs",
    path=Path("external-docs").resolve(),
    description="Reference documents",
)

flow = Flow(backend, context_dirs=(docs,))
result = await flow.run(
    contract,
    prompt="Read @ref:docs/source.md and produce the declared output.",
    cwd=Path.cwd(),
)
```

- Core expands refs to resolved absolute paths before the backend starts.
- Unknown id, absolute relpath, or `..` / symlink escape fails fast.
- `context_dirs` are also passed to the backend (Claude add-dirs / Codex `--add-dir`).

## Event streams

If you only care about the final result, use `Flow.run()`. If you need live progress, use `Flow.run_iter()`. For disk feeds (tail / external UI), use `JsonlEventSink`.

```python
run = flow.run_iter(contract, prompt=prompt, cwd=Path.cwd())

async for event in run:
    print(event.kind.value, event.data)

result = await run.result()
```

```python
from tilldone import JsonlEventSink

with JsonlEventSink(Path("events.jsonl"), Path("messages.jsonl")) as sink:
    result = await flow.run(
        contract,
        prompt=prompt,
        cwd=Path.cwd(),
        event_sink=sink,
    )
```

- `events.jsonl` holds normalized events (no `raw`); `messages.jsonl` optionally stores JSON-safe raw payloads.
- Each line is flushed immediately — suitable for `tail` or an external viewer.
- Eleven standard `EventKind` values: `run_started`, `assistant_text`, `reasoning`, `tool_call`, `tool_result`, `command`, `file_changed`, `usage`, `backend_notice`, `run_completed`, `run_failed`.
- `AgentEvent` can carry session/thread/turn/item/call correlation IDs.

There is no built-in viewer CLI; the public surface is the event stream and JSONL feed.

## Resume & sessions

Two different behaviors:

- **Within one contract task**, `run_task` automatically threads each round’s `RunOutcome.resume` into the next.
- **Across separate calls**, resume is a backend-level `AgentRunSpec.resume` capability. `Flow.run()` has no `resume=` parameter.

`ResumeHandle` is an opaque, backend-tagged token. Same-backend only — Claude → Codex (or the reverse) fails fast. There is no cross-backend resume.

`SessionIndex` can persist, list, and restore handles, and checks that required tool aliases / context dir IDs are still available. It is **not** wired into `Flow` automatically.

## Recipes

Recipes are ordinary async convenience functions that assemble a `CompletionContract`, prompt, and `run_task` call — not a plugin system. The only public recipe today:

```python
from pathlib import Path

from tilldone.recipes.translate import translate

await translate(
    backend,
    source=Path("notes.md"),
    dest=Path("notes.zh.md"),
    target_lang="Chinese",
    cwd=Path.cwd(),
)
```

`translate` registers the source parent as `ContextDirView(id="src_dir", ...)`, asks the agent to read via `@ref`, requires a non-empty destination file, and requires `finish`. With no return values it returns `None`.

## Public API map

Top-level `from tilldone import ...`:

| Category | Names |
| --- | --- |
| Driver | `run_task`, `CorePolicy`, `MaxRoundsExceeded`, `BackendFailure` |
| Contract | `CompletionContract`, `OutputExpectation`, `ReturnExpectation`, `TodoExpectation`, `ContractState`, `ContractEvaluator` |
| Tools | `ToolDef`, `ToolCall`, `ToolResult`, `ToolExecutor` |
| Run model | `AgentRunSpec`, `RunOutcome`, `ResumeHandle`, `BackendError`, `BackendCapabilities`, `ContextDirView` |
| Events | `AgentEvent`, `EventKind`, `JsonlEventSink` |
| Backend protocols | `Backend`, `RunHandle`, `SessionCapable` |
| High-level SDK | `Flow`, `ActiveRunError` |
| Session index | `SessionIndex`, `SessionInfo`, `SessionRestoreError` |

Backends stay on optional modules so `import tilldone` never loads an SDK:

```python
from tilldone.backends.claude import ClaudeBackend
from tilldone.backends.codex import CodexBackend
```

## Development

```bash
uv sync --all-extras
uv run pytest
```

Pytest uses `asyncio_mode = "auto"`.
