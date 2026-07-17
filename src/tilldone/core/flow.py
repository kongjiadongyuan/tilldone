"""tilldone.core.flow — SDK wrapper over run_task (RT-SDK-1 / AR-6).

Provides:
    ActiveRunError  — raised when a second concurrent run is attempted on the same Flow.
    Flow            — high-level wrapper that exposes run() and run_iter().
    RunIter         — async-iterable of AgentEvents returned by Flow.run_iter(); exposes
                      an awaitable .result() that resolves to the task result.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from tilldone.core.backend import Backend
from tilldone.core.contract import CompletionContract
from tilldone.core.events import AgentEvent
from tilldone.core.loop import CorePolicy, run_task


class ActiveRunError(RuntimeError):
    """Raised when a second concurrent run is attempted on the same Flow instance."""


_SENTINEL = object()  # marks end-of-events in the internal queue


class RunIter:
    """Async-iterable of AgentEvents produced by a Flow.run_iter() call.

    Usage::

        it = flow.run_iter(contract, prompt="…", cwd=…)
        async for event in it:
            process(event)
        result = await it.result()

    The iterator drains the event queue until the background run_task finishes.
    Calling .result() before the loop is exhausted will wait for completion.
    If the background task raised, .result() re-raises that exception.
    """

    def __init__(
        self,
        task: asyncio.Task,
        queue: asyncio.Queue,
        release: asyncio.Event,
    ) -> None:
        self._task = task
        self._queue = queue
        self._release = release
        self._drained = False

    def __aiter__(self) -> AsyncIterator[AgentEvent]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[AgentEvent]:
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                self._drained = True
                break
            yield item

    async def result(self) -> Any:
        """Await the completion of the background task and return its result.

        If the task raised an exception it is re-raised here.
        Can be called even if the async-for loop was not fully consumed first;
        drains (and discards) any remaining events in that case.
        """
        if not self._drained:
            # Drain remaining events so the producer is unblocked.
            while True:
                item = await self._queue.get()
                if item is _SENTINEL:
                    self._drained = True
                    break
        # At this point the background task must already be done (the sentinel
        # is only placed after run_task returns or raises).
        return await self._task


class Flow:
    """High-level SDK wrapper over run_task.

    A Flow is bound to a single backend at construction time.  At most ONE run
    (via run() or run_iter()) may be active on a given Flow instance at a time;
    a second concurrent call raises ActiveRunError.

    Args:
        backend:       The Backend implementation to use.
        policy:        Optional CorePolicy overrides.
        custom_tools:  Optional mapping of extra tool names to handlers.
        context_dirs:  Optional sequence of ContextDirView objects.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        policy: CorePolicy | None = None,
        custom_tools: dict | None = None,
        context_dirs: tuple = (),
    ) -> None:
        self._backend = backend
        self._policy = policy or CorePolicy()
        self._custom_tools = custom_tools
        self._context_dirs = context_dirs
        self._busy = False
        self._lock = asyncio.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _acquire(self) -> None:
        """Mark the flow as busy, raising ActiveRunError if already active."""
        if self._busy:
            raise ActiveRunError(
                "A run is already in progress on this Flow instance. "
                "Await the previous run to complete before starting a new one."
            )
        self._busy = True

    def _release(self) -> None:
        self._busy = False

    async def aclose(self) -> None:
        """Close the underlying backend lifecycle resources, if any.

        This is backend-agnostic lifecycle cleanup: resident transports,
        subprocesses, SDK clients, or other resources stay owned by the backend,
        and Flow simply forwards the close once.
        """

        if self._closed:
            return
        self._closed = True
        await self._backend.aclose()

    async def __aenter__(self) -> "Flow":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        contract: CompletionContract,
        *,
        prompt: str,
        cwd: Path,
        event_sink=None,
    ) -> Any:
        """Run run_task and return its result.

        Raises ActiveRunError if another run is already in progress on this Flow.
        """
        self._acquire()
        try:
            return await run_task(
                contract,
                self._backend,
                prompt=prompt,
                cwd=cwd,
                policy=self._policy,
                custom_tools=self._custom_tools,
                context_dirs=self._context_dirs,
                event_sink=event_sink,
            )
        finally:
            self._release()

    def run_iter(
        self,
        contract: CompletionContract,
        *,
        prompt: str,
        cwd: Path,
    ) -> RunIter:
        """Return a RunIter that yields AgentEvents as they are produced.

        Calling this method marks the Flow as busy immediately (before any event
        is yielded), so a subsequent run() call will raise ActiveRunError.

        The run executes as a background asyncio.Task.  Events are buffered in an
        asyncio.Queue and drained by the returned RunIter.  After iteration, call
        ``await it.result()`` to obtain the task result (or surface an exception).

        Raises ActiveRunError synchronously if another run is already in progress.
        """
        self._acquire()

        queue: asyncio.Queue = asyncio.Queue()

        def _sink(event: AgentEvent) -> None:
            queue.put_nowait(event)

        async def _run_and_signal() -> Any:
            try:
                result = await run_task(
                    contract,
                    self._backend,
                    prompt=prompt,
                    cwd=cwd,
                    policy=self._policy,
                    custom_tools=self._custom_tools,
                    context_dirs=self._context_dirs,
                    event_sink=_sink,
                )
                return result
            finally:
                # Always place sentinel so the RunIter can exit its loop.
                queue.put_nowait(_SENTINEL)
                self._release()

        loop = asyncio.get_event_loop()
        task = loop.create_task(_run_and_signal())

        release_event = asyncio.Event()
        return RunIter(task, queue, release_event)
