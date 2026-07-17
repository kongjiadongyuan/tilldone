"""Pure JSON-RPC stdio transport over ``codex app-server --stdio``.

This is the protocol-only foundation for all later app-server work. It owns:

* spawning ``codex app-server --stdio`` and wiring its stdio,
* **newline-delimited JSON** framing (one object per line) — robust against
  split-mid-line reads, multiple objects in one read, and garbage lines,
* request/response multiplexing keyed by the **client's own** id counter,
* server→client request routing keyed by ``method`` (the server uses an
  independent id space — never correlate by direction),
* a notification pump exposing an async iterator for upper layers,
* per-request timeouts, and
* deterministic shutdown that never leaves an awaiter hanging.

Hard scope boundary (M1): this module is ignorant of any Backend protocol,
event normalization, or thread/turn/contract semantics. It returns raw
JSON-RPC results / notifications as plain ``dict`` / ``Any``; normalization is
M2's job. It MUST NOT import :mod:`tilldone.core`.

Protocol facts (observed against ``codex app-server --stdio``):

* Transport ``codex app-server --stdio``; newline-delimited JSON, **not**
  Content-Length framing.
* Envelopes carry **no** ``jsonrpc:"2.0"`` field — we never send it and tolerate
  it if present.
* Two independent id spaces: the client uses its own monotonic counter; the
  server uses its own (observed ``id:0`` for ``item/tool/call``). Responses to
  client requests are correlated by client id; server→client requests are
  routed by ``method``.
* The only server→client REQUEST under NO GATING is ``item/tool/call``.
  Notifications carry a ``method`` and no ``id``.

Testability seam
----------------
The reader / stderr / writer pumps operate on injected stream objects, so the
in-memory ``fake_peer`` test harness can drive a fully real transport WITHOUT
spawning codex. Production code uses :meth:`AppServerTransport.start` (which
spawns the child and wires its real stdio); tests use the documented
:meth:`AppServerTransport._attach` + :meth:`AppServerTransport._spawn_pumps`
seam. See ``tests/backends/app_server/fake_peer.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Optional, Protocol

# JSON-RPC error code used when an inbound server→client request targets a
# method with no registered handler. -32601 = "Method not found".
_METHOD_NOT_FOUND = -32601
# Used when a registered handler raises while producing a result.
_INTERNAL_ERROR = -32603

# Bounded tail kept from the child's stderr for error detail (lines).
_STDERR_TAIL_MAX = 200


class JsonRpcError(Exception):
    """A JSON-RPC ``error`` object returned by the peer for a client request.

    Carries the structured ``code`` / ``message`` / ``data`` fields verbatim so
    upper layers (M2) can classify transient vs fatal without re-parsing.
    """

    def __init__(self, code: Any, message: Any, data: Any = None) -> None:
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"JSON-RPC error {code}: {message}")


class TransportClosed(Exception):
    """Raised when the transport is closed (EOF, peer death, or ``aclose``).

    Any request future still awaiting a response is failed with this so callers
    never hang on a dead connection.
    """


class _ByteWriter(Protocol):
    """Minimal outbound-byte sink (satisfied by ``asyncio.StreamWriter``)."""

    def write(self, data: bytes) -> Any: ...


class _ByteReader(Protocol):
    """Minimal inbound-byte source (satisfied by ``asyncio.StreamReader``).

    We only need ``read(n)`` — enough to do our own newline framing and stay
    robust to partial/packed reads regardless of the peer's flushing.
    """

    async def read(self, n: int = ...) -> bytes: ...


_ServerRequestHandler = Callable[[dict], Awaitable[Any]]


class AppServerTransport:
    """asyncio JSON-RPC stdio client for ``codex app-server``.

    Lifecycle::

        t = AppServerTransport(codex_bin=..., env=..., cwd=...)
        t.on_server_request("item/tool/call", handler)   # optional, allowlisted
        await t.start()
        result = await t.request("initialize", {...})     # id-correlated
        await t.notify("initialized", {})
        async for note in t.notifications():               # server notifications
            ...
        await t.aclose()                                   # idempotent

    All public coroutines are safe to call concurrently; a single reader task
    multiplexes responses, server requests, and notifications, so a server
    ``item/tool/call`` arriving while another ``request()`` awaits its response
    is dispatched rather than deadlocked.
    """

    #: Bytes to pull per stdout read. Framing is done by us, so the value only
    #: affects how aggressively we coalesce — correctness is independent of it.
    _READ_CHUNK = 65536

    def __init__(
        self,
        *,
        codex_bin: str = "codex",
        args: tuple[str, ...] = ("app-server", "--stdio"),
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        request_timeout_s: float = 120.0,
    ) -> None:
        self._codex_bin = codex_bin
        self._args = tuple(args)
        self._env = dict(env) if env is not None else None
        self._cwd = cwd
        self._request_timeout_s = request_timeout_s

        # Wire I/O — set by start() (real subprocess) or _attach() (tests).
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._writer: Optional[_ByteWriter] = None
        self._reader: Optional[_ByteReader] = None
        self._stderr_reader: Optional[_ByteReader] = None

        # Client request multiplexing — our own monotonic id counter.
        self._next_id = 0
        self._pending: dict[Any, asyncio.Future[Any]] = {}

        # Server→client request handlers (allowlist by method).
        self._handlers: dict[str, _ServerRequestHandler] = {}
        self._unexpected_server_requests: list[dict] = []

        # Notification fan-out to upper layers.
        self._notifications: asyncio.Queue[Optional[dict]] = asyncio.Queue()

        # Pump tasks + lifecycle.
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._handler_tasks: set[asyncio.Task] = set()
        self._stderr_tail: list[str] = []
        self._started = False
        self._closed = False
        self._closing = False

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Spawn ``codex app-server --stdio`` and launch the reader + stderr pumps.

        Idempotent-guarded: a second call raises ``RuntimeError`` rather than
        leaking a second child.
        """
        if self._started:
            raise RuntimeError("AppServerTransport already started")
        env = dict(os.environ)
        if self._env is not None:
            env.update(self._env)
        proc = await asyncio.create_subprocess_exec(
            self._codex_bin,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=self._cwd,
        )
        # mypy: PIPE guarantees these are not None.
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        self._attach(
            reader=proc.stdout,
            writer=proc.stdin,
            stderr_reader=proc.stderr,
            proc=proc,
        )
        self._spawn_pumps()

    def _attach(
        self,
        *,
        reader: _ByteReader,
        writer: _ByteWriter,
        stderr_reader: _ByteReader | None = None,
        proc: asyncio.subprocess.Process | None = None,
    ) -> None:
        """Test seam: wire injected byte streams instead of a real subprocess.

        ``reader``/``stderr_reader`` need only ``async read(n)``; ``writer`` needs
        only ``write(bytes)``. Used by the in-memory fake peer. Call
        :meth:`_spawn_pumps` afterwards to start the reader/stderr tasks.
        """
        self._reader = reader
        self._writer = writer
        self._stderr_reader = stderr_reader
        self._proc = proc

    def _spawn_pumps(self) -> None:
        """Launch the reader pump (and stderr drain if a stderr stream exists)."""
        self._started = True
        self._reader_task = asyncio.create_task(self._read_loop())
        if self._stderr_reader is not None:
            self._stderr_task = asyncio.create_task(self._drain_stderr())

    # -- outbound -----------------------------------------------------------

    def _send_raw(self, msg: dict) -> None:
        if self._closed or self._writer is None:
            raise TransportClosed("transport is closed")
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        self._writer.write(data)

    async def request(self, method: str, params: dict | None = None) -> Any:
        """Send a client request and await its id-correlated response.

        Returns the JSON-RPC ``result``. Raises:
          * :class:`JsonRpcError` if the peer returns an ``error`` result,
          * :class:`TimeoutError` after ``request_timeout_s`` (the transport
            stays usable — only this request is abandoned),
          * :class:`TransportClosed` if the connection dies (or is closed)
            before a response arrives.
        """
        if self._closed:
            raise TransportClosed("transport is closed")
        self._next_id += 1
        rid = self._next_id
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self._pending[rid] = fut
        msg: dict[str, Any] = {"id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        try:
            self._send_raw(msg)
        except TransportClosed:
            self._pending.pop(rid, None)
            raise
        try:
            return await asyncio.wait_for(fut, timeout=self._request_timeout_s)
        except asyncio.TimeoutError as exc:
            # Per-request timeout: abandon this id, keep the transport alive.
            raise TimeoutError(
                f"request {method!r} (id={rid}) timed out after {self._request_timeout_s}s"
            ) from exc
        finally:
            self._pending.pop(rid, None)

    async def notify(self, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        msg: dict[str, Any] = {"method": method}
        if params is not None:
            msg["params"] = params
        self._send_raw(msg)

    def _respond_result(self, rid: Any, result: Any) -> None:
        self._send_raw({"id": rid, "result": result})

    def _respond_error(self, rid: Any, code: int, message: str, data: Any = None) -> None:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        self._send_raw({"id": rid, "error": err})

    # -- server-request handlers --------------------------------------------

    def on_server_request(self, method: str, handler: _ServerRequestHandler) -> None:
        """Register an allowlisted handler for a server→client request ``method``.

        ``handler(params)`` is awaited; its return value is sent back as the
        JSON-RPC ``result``. If it raises, a JSON-RPC error (-32603) is sent
        with the exception text. Methods with no registered handler are
        auto-answered with -32601 and recorded in
        :attr:`unexpected_server_requests` — the pump never hangs.
        """
        self._handlers[method] = handler

    @property
    def unexpected_server_requests(self) -> list[dict]:
        """Server requests whose ``method`` had no registered handler.

        Each was auto-answered with a JSON-RPC -32601 error AND appended here as
        ``{"id", "method", "params"}``. Returns a shallow copy so callers cannot
        mutate internal state.
        """
        return list(self._unexpected_server_requests)

    # -- notifications ------------------------------------------------------

    async def notifications(self) -> AsyncIterator[dict]:
        """Yield server→client notifications as ``{"method", "params"}`` dicts.

        Terminates (the iterator ends) when the transport closes. Intended for a
        single consumer; multiple concurrent iterators will compete for items.
        """
        while True:
            item = await self._notifications.get()
            if item is None:  # sentinel: transport closed
                return
            yield item

    # -- inbound pump -------------------------------------------------------

    async def _read_loop(self) -> None:
        """Single reader task: frame newline-delimited JSON and dispatch.

        Robust to: a JSON object split across reads, several objects in one
        read, and non-JSON garbage lines (skipped + tail-recorded, never fatal).
        """
        assert self._reader is not None
        reader = self._reader
        buf = b""
        try:
            while True:
                chunk = await reader.read(self._READ_CHUNK)
                if not chunk:  # EOF: peer closed stdout / exited
                    break
                buf += chunk
                # Split complete lines; keep the trailing partial line in buf.
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._handle_line(line)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive; pump must never crash silently
            pass
        finally:
            # Flush a trailing complete object that arrived without a newline
            # (defensive — the wire is newline-terminated, but EOF-without-\n
            # should not silently drop a final object).
            tail = buf.strip()
            if tail:
                self._handle_line(tail)
            self._fail_all_pending(TransportClosed("app-server stdout closed (EOF)"))
            # Wake any notification consumers so their iterator terminates.
            self._notifications.put_nowait(None)

    def _handle_line(self, raw: bytes) -> None:
        text = raw.decode("utf-8", "replace").strip()
        if not text:
            return
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            # Garbage / non-JSON line: record a bounded tail, keep pumping.
            self._stderr_tail.append(f"[non-json stdout] {text[:200]}")
            del self._stderr_tail[:-_STDERR_TAIL_MAX]
            return
        if not isinstance(msg, dict):
            self._stderr_tail.append(f"[non-object stdout] {text[:200]}")
            del self._stderr_tail[:-_STDERR_TAIL_MAX]
            return
        self._route(msg)

    def _route(self, msg: dict) -> None:
        has_id = "id" in msg
        has_method = "method" in msg

        # Response to one of our client requests: id present, no method,
        # result-or-error present. Correlate strictly by our id.
        if has_id and not has_method and ("result" in msg or "error" in msg):
            fut = self._pending.get(msg["id"])
            if fut is not None and not fut.done():
                if "error" in msg:
                    err = msg["error"] or {}
                    fut.set_exception(
                        JsonRpcError(err.get("code"), err.get("message"), err.get("data"))
                    )
                else:
                    fut.set_result(msg.get("result"))
            # Unknown / already-resolved id: ignore (late duplicate, timed-out).
            return

        # Server→client request: id AND method (server's own id space).
        if has_id and has_method:
            self._dispatch_server_request(msg["id"], msg["method"], msg.get("params"))
            return

        # Notification: method, no id.
        if has_method and not has_id:
            self._notifications.put_nowait(
                {"method": msg["method"], "params": msg.get("params")}
            )
            return

        # Unknown envelope shape: record, don't crash.
        self._stderr_tail.append(f"[unrouted] {json.dumps(msg, ensure_ascii=False)[:200]}")
        del self._stderr_tail[:-_STDERR_TAIL_MAX]

    def _dispatch_server_request(self, rid: Any, method: str, params: Any) -> None:
        handler = self._handlers.get(method)
        if handler is None:
            # No allowlisted handler → answer with an error AND record it.
            self._unexpected_server_requests.append(
                {"id": rid, "method": method, "params": params}
            )
            self._respond_error(
                rid,
                _METHOD_NOT_FOUND,
                f"tilldone app-server transport: no handler for server-request {method!r}",
            )
            return
        # Run the handler concurrently so a slow handler cannot stall the pump
        # (and so a server request arriving mid-request() is non-blocking).
        task = asyncio.create_task(self._run_handler(handler, rid, method, params))
        self._handler_tasks.add(task)
        task.add_done_callback(self._handler_tasks.discard)

    async def _run_handler(
        self, handler: _ServerRequestHandler, rid: Any, method: str, params: Any
    ) -> None:
        try:
            result = await handler({"method": method, "params": params})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # handler blew up → send a JSON-RPC error back
            try:
                self._respond_error(rid, _INTERNAL_ERROR, f"handler error: {exc!r}")
            except TransportClosed:
                pass
            return
        try:
            self._respond_result(rid, result)
        except TransportClosed:
            pass

    async def _drain_stderr(self) -> None:
        """Continuously read child stderr so its PIPE never fills (P0#1).

        An unread stderr PIPE on the child fills its OS buffer and deadlocks the
        child — this is exactly the v1 P0#1 bug. We keep a bounded tail for error
        detail and otherwise discard.
        """
        assert self._stderr_reader is not None
        reader = self._stderr_reader
        buf = b""
        try:
            while True:
                chunk = await reader.read(self._READ_CHUNK)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    txt = line.decode("utf-8", "replace").rstrip("\r")
                    if txt:
                        self._stderr_tail.append(txt)
                        del self._stderr_tail[:-_STDERR_TAIL_MAX]
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            pass

    def stderr_tail(self, n: int = 40) -> str:
        """Return the last ``n`` recorded stderr (and unrouted-stdout) lines."""
        return "\n".join(self._stderr_tail[-n:])

    # -- teardown -----------------------------------------------------------

    def _fail_all_pending(self, exc: BaseException) -> None:
        """Resolve every pending request future with ``exc`` so no awaiter hangs."""
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    async def aclose(self) -> None:
        """Kill + reap the child, cancel pumps, and fail all pending requests.

        Idempotent. After this, awaiters of in-flight :meth:`request` calls
        receive :class:`TransportClosed` (they never hang), the notifications
        iterator terminates, and the child process is reaped (no zombie).
        """
        if self._closed:
            return
        self._closed = True
        self._closing = True

        # 1) Fail pending request futures up-front so awaiters unblock promptly.
        self._fail_all_pending(TransportClosed("transport closed"))

        # 2) Close stdin so the child sees EOF on its input (best-effort).
        writer = self._writer
        if writer is not None:
            close = getattr(writer, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        # 3) Kill + reap the child (only this child; tracked by handle).
        proc = self._proc
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except Exception:
                pass

        # 4) Cancel + await pump tasks and any in-flight handler tasks.
        tasks: list[asyncio.Task] = []
        for t in (self._reader_task, self._stderr_task):
            if t is not None:
                t.cancel()
                tasks.append(t)
        for t in list(self._handler_tasks):
            t.cancel()
            tasks.append(t)
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        # 5) Terminate the notifications iterator for any consumer.
        self._notifications.put_nowait(None)
