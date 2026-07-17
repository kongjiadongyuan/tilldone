# TillDone

[English](README.md) | 简体中文

> **状态:alpha**(`1.0.0a0`)。稳定版发布前 API 可能变动。

TillDone 是面向 Python 的任务型 agent 执行器与后端兼容层。完成条件只声明一次，同一任务可在 Claude Code 或 Codex 上运行。

任务是否完成，只看声明的 outputs、return values、todos 与 finish 信号是否全部满足。契约落在与后端无关的 core 里，各 backend 再翻译到各自的原生 agent runtime。Same contract, different backend.

## Why TillDone

```text
CompletionContract + prompt + tools
                 |
           Flow / run_task
                 |
        ClaudeBackend | CodexBackend
```

三条边界：

- **契约在 core。** Backend 不能自行宣布文件已完成。
- **Backend 只负责跑一轮。** 流出标准事件，返回标准 outcome。
- **是否继续由 core 决定。** 每轮结束后重新评估文件系统与契约工具结果。

Core 无运行时第三方依赖；backend SDK 通过 extras 安装。

## Installation

需要 **Python ≥ 3.11**。

```bash
# Claude Code backend
uv add "tilldone[claude]"

# Codex backend
uv add "tilldone[codex]"
```

两个后端都要用时，装上两个 extras 即可。Core 本身不带依赖；`claude` extra 安装 `claude-agent-sdk`，`codex` extra 安装 `mcp`。`CodexBackend` 还要求 `codex` CLI 在 `PATH` 中，并按 Codex 自身方式完成认证与配置。

## Quickstart

一份真实的完成契约：写文件，并返回类型化值。

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

`Flow` 是高层执行边界：绑定一个 backend，同一时刻最多一个 active run。退出 async context 时会调用 backend 的 `aclose()`。

切换 backend 只需改 import 与构造：

```python
from tilldone.backends.claude import ClaudeBackend

backend = ClaudeBackend()
```

或：

```python
from tilldone.backends.codex import CodexBackend

backend = CodexBackend()
```

Same contract, different backend.

## Completion contracts

完成与否按声明状态评估，不靠模型最后一段自然语言推断。四类条件：

| Condition | API | Satisfied when |
| --- | --- | --- |
| File output | `OutputExpectation` | 文件存在、非空，且相对运行前快照发生变化；可选 validator 通过 |
| Return value | `ReturnExpectation` | Agent 调用内置 `return`，名称已声明，值符合 `type_hint` |
| Todo | `TodoExpectation` | Agent 调用内置 `todo`，名称必须已声明 |
| Finish | `require_finish=True` | Agent 调用内置 `finish` |

实践中需要记住的细节：

- `expect="created"` — 运行后文件必须非空，且相对 pre-run snapshot 有变化。
- `expect="updated"` — 同上，且文件在运行前就必须已存在。
- Evaluator **不信任** backend 的 `file_changed` 事件；以文件系统内容与 SHA-256 前快照为准。
- 0 个 return → `None`；1 个 → 标量；多个 → `dict`。
- 已知 `type_hint`：`bool`、`int`、`float`、`str`、`json`、`list`/`array`、`dict`/`object`，以及递归的 `list[...]`。
- 契约未满足会继续跑（默认最多 6 个 contract rounds），可用 `CorePolicy` 调整。

Todos：

```python
from tilldone import CompletionContract, TodoExpectation

contract = CompletionContract(
    outputs=(),
    returns=(),
    todos=(TodoExpectation("check-tests"), TodoExpectation("write-summary")),
    require_finish=True,
)
```

可选 validator（从 `tilldone.core.validators` 导入）：

```python
from tilldone import OutputExpectation
from tilldone.core.validators import JsonValidator

output = OutputExpectation("result", Path("result.json").resolve(), "created", JsonValidator())
```

另有：`JsonSchemaValidator`、`LineCountValidator`、`MagicBytesValidator`、`RegexBytesValidator`、`CallableValidator`、`AllOf`。

## Custom tools & MCP

Custom tools 是宿主进程里的 async handler，接收 `ToolCall`，返回 `ToolResult`。传给 `Flow(custom_tools=...)` 或 `run_task(custom_tools=...)`。Claude 经进程内 MCP 暴露；Codex 经 TillDone 管理的 localhost MCP bridge 暴露。

```python
from tilldone import Flow, ToolCall, ToolResult


async def get_build_id(call: ToolCall) -> ToolResult:
    return ToolResult(content="build-42")


flow = Flow(
    backend,
    custom_tools={"get_build_id": get_build_id},
)
```

Prompt 里要写清调用哪个工具、如何使用结果。公开映射只有 name → handler（此层不提供 description/schema 参数）。

Codex 还可挂接自有 MCP server：

```python
from tilldone.backends.codex import CodexBackend

backend = CodexBackend(
    extra_config=[
        'mcp_servers.files.url="http://127.0.0.1:7777/mcp"',
        'mcp_servers.files.bearer_token_env_var="FILES_MCP_TOKEN"',
    ]
)
```

`mcp_servers.tilldone.*` 是内置 bridge 的保留命名空间，覆盖会 fail fast。Token 走环境变量，不要写进配置字符串。

## Context dirs & `@ref`

用命名目录把 prompt 与机器绝对路径解耦。语法：`@ref:<id>/<relative-path>`。

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

- Core 在启动 backend 前把 ref 展开为解析后的绝对路径。
- 未知 id、绝对 relpath、或 `..` / 符号链接逃逸会 fail fast。
- `context_dirs` 同时传给 backend（Claude add-dirs / Codex `--add-dir`）。

## Event streams

只关心最终结果用 `Flow.run()`；要实时进度用 `Flow.run_iter()`；要落盘供 `tail` / 外部 UI 用 `JsonlEventSink`。

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

- `events.jsonl` 是归一事件（不含 `raw`）；`messages.jsonl` 可选保存 JSON-safe 的 raw payload。
- 每行立即 flush，适合 `tail` 或外部 viewer。
- 标准 `EventKind` 共 11 类：`run_started`、`assistant_text`、`reasoning`、`tool_call`、`tool_result`、`command`、`file_changed`、`usage`、`backend_notice`、`run_completed`、`run_failed`。
- `AgentEvent` 还可携带 session/thread/turn/item/call 等 correlation ID。

没有内置 viewer CLI；公开面是 event stream 与 JSONL feed。

## Resume & sessions

两种行为不要混：

- **同一次契约任务内**，`run_task` 会自动把每轮的 `RunOutcome.resume` 传给下一轮。
- **跨独立调用** 的 resume 是 backend 级 `AgentRunSpec.resume` 能力；`Flow.run()` 没有 `resume=` 参数。

`ResumeHandle` 是带 backend 标记的不透明 token，只能同后端续接——Claude 的 handle 传给 Codex（或反过来）会 fail fast。不支持 cross-backend resume。

`SessionIndex` 可持久化、列出并恢复 handle，并校验原会话所需的 tool aliases 与 context dir IDs 是否仍可用。它**不会**自动接到 `Flow` 上。

## Recipes

Recipe 是普通 async 便利函数：拼好 `CompletionContract`、prompt 和 `run_task` 调用，不是插件系统。当前唯一公开 recipe：

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

`translate` 会把 source 父目录注册为 `ContextDirView(id="src_dir", ...)`，通过 `@ref` 要求 agent 读真实文件，要求目标文件创建且非空，并要求 `finish`。无返回值时返回 `None`。

## Public API map

顶层 `from tilldone import ...`：

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

Backend 从可选模块显式 import，保证 `import tilldone` 不会加载 SDK：

```python
from tilldone.backends.claude import ClaudeBackend
from tilldone.backends.codex import CodexBackend
```

## Development

```bash
uv sync --all-extras
uv run pytest
```

Pytest 使用 `asyncio_mode = "auto"`。
