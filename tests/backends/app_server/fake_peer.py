"""In-memory fake ``codex app-server`` peer for transport unit tests.

NO real codex spawn. This drives a *real* :class:`AppServerTransport` over a
pair of in-memory byte pipes, so protocol behavior is exercised end-to-end and
deterministically (no gateway, no subprocess, no sleeps tied to wall-clock).

Why it exists
-------------
``AppServerTransport.start()`` spawns a child and wires its real stdio. For unit
tests we instead inject in-memory streams through the transport's documented
test seam (``_attach`` + ``_spawn_pumps``). ``FakePeer`` owns the *server* end
of those pipes and lets a test script arbitrary server behavior.

Wiring (two unidirectional byte pipes)::

    client (transport)  --writes-->  c2s pipe  --reads-->  FakePeer.read_client_message()
    FakePeer.send_*()   --writes-->  s2c pipe  --reads-->  client (transport) reader pump

Public API (LaneB scripts server behavior with these)
------------------------------------------------------
Construction / lifecycle::

    peer = FakePeer()
    transport = peer.make_transport(request_timeout_s=...)   # attaches + starts pumps
    ...
    await peer.aclose()            # closes both pipes (drives transport EOF); idempotent

Observe what the client sent (one parsed JSON object per call, in order)::

    msg = await peer.read_client_message()        # {"id"/"method"/"params"/...}
    msg = await peer.read_client_message(timeout=0.5)

Drive the server→client direction:

    peer.respond(client_id, result)               # success response to a client request id
    peer.respond_error(client_id, code, message, data=None)
    peer.notify(method, params=None)              # server notification (no id)
    peer.server_request(method, params=None, id=0) # server→client REQUEST (server's own id)

Adversarial framing (each writes raw bytes onto the s2c pipe verbatim)::

    peer.send_raw_bytes(b'...')                   # arbitrary bytes (no auto newline)
    peer.send_line(obj)                           # one object + '\n' (normal framing)
    peer.send_split(obj, at=N)                    # one object across two writes, split at byte N
    peer.send_packed(obj_a, obj_b, ...)           # several objects in ONE write (all newline-joined)
    peer.send_garbage_line(text)                  # a non-JSON line + '\n'

All ``send_*`` / ``respond*`` / ``notify`` / ``server_request`` are synchronous
(they enqueue bytes; the transport's reader pump consumes them asynchronously).
``read_client_message`` is async because it awaits bytes from the client.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from tilldone.backends.app_server.transport import AppServerTransport


class _Pipe:
    """A one-way in-memory byte pipe.

    Satisfies the transport's two stream protocols simultaneously:
      * writer side: synchronous ``write(bytes)`` (like ``StreamWriter.write``),
      * reader side: ``await read(n)`` returning up to ``n`` buffered bytes, or
        ``b""`` once the pipe is closed AND drained (EOF).

    A backing :class:`asyncio.StreamReader` gives us correct buffering + EOF +
    wakeups for free.
    """

    def __init__(self) -> None:
        self._reader = asyncio.StreamReader()

    # writer side -----------------------------------------------------------
    def write(self, data: bytes) -> None:
        self._reader.feed_data(data)

    def close(self) -> None:
        if not self._reader.at_eof():
            self._reader.feed_eof()

    # reader side -----------------------------------------------------------
    async def read(self, n: int = 65536) -> bytes:
        # StreamReader.read(n) returns b"" at EOF, else up to n buffered bytes.
        return await self._reader.read(n)


class FakePeer:
    """The server end of an in-memory ``codex app-server`` connection.

    See module docstring for the full scripting API.
    """

    def __init__(self) -> None:
        # c2s: client (transport) -> server (we read it to inspect client msgs).
        self._c2s = _Pipe()
        # s2c: server (we write) -> client (transport reader pump consumes it).
        self._s2c = _Pipe()
        self._client_buf = b""
        self._transport: AppServerTransport | None = None
        self._closed = False

    # -- construction -------------------------------------------------------

    def make_transport(self, *, request_timeout_s: float = 120.0, **kwargs: Any) -> AppServerTransport:
        """Build a real transport wired to this peer and start its pumps.

        ``kwargs`` are forwarded to :class:`AppServerTransport` (e.g. ``env``),
        but ``start()`` is NOT called — no subprocess is spawned. Instead the
        documented ``_attach`` + ``_spawn_pumps`` seam injects the in-memory
        pipes. The transport behaves exactly as in production from here on.
        """
        t = AppServerTransport(request_timeout_s=request_timeout_s, **kwargs)
        t._attach(reader=self._s2c, writer=self._c2s, stderr_reader=None, proc=None)
        t._spawn_pumps()
        self._transport = t
        return t

    # -- observe the client -------------------------------------------------

    async def read_client_message(self, *, timeout: float = 5.0) -> dict:
        """Read and return the next JSON object the client (transport) sent.

        Performs newline framing over the c2s pipe, returning one parsed object
        per call in send order. Raises ``asyncio.TimeoutError`` if nothing
        arrives within ``timeout`` (keeps tests from hanging), or
        ``EOFError`` if the client side closed without a complete message.
        """
        return await asyncio.wait_for(self._read_client_message(), timeout=timeout)

    async def _read_client_message(self) -> dict:
        while b"\n" not in self._client_buf:
            chunk = await self._c2s.read(65536)
            if not chunk:
                raise EOFError("client closed c2s pipe with no complete message buffered")
            self._client_buf += chunk
        line, self._client_buf = self._client_buf.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    # -- drive the server→client direction ----------------------------------

    def respond(self, client_id: Any, result: Any) -> None:
        """Send a success response correlated to a client request ``id``."""
        self.send_line({"id": client_id, "result": result})

    def respond_error(self, client_id: Any, code: int, message: str, data: Any = None) -> None:
        """Send an error response correlated to a client request ``id``."""
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        self.send_line({"id": client_id, "error": err})

    def notify(self, method: str, params: Any = None) -> None:
        """Send a server notification (method, no id)."""
        msg: dict[str, Any] = {"method": method}
        if params is not None:
            msg["params"] = params
        self.send_line(msg)

    def server_request(self, method: str, params: Any = None, *, id: Any = 0) -> None:
        """Send a server→client REQUEST using the SERVER's own id space.

        Defaults to ``id=0`` to mirror the live ``item/tool/call`` observation —
        deliberately overlapping the client's id space to prove correlation is
        by direction/shape (id+method ⇒ request), never by id value.
        """
        msg: dict[str, Any] = {"id": id, "method": method}
        if params is not None:
            msg["params"] = params
        self.send_line(msg)

    # -- raw / adversarial framing ------------------------------------------

    def send_raw_bytes(self, data: bytes) -> None:
        """Write arbitrary bytes onto the s2c pipe verbatim (no auto newline)."""
        self._s2c.write(data)

    def send_line(self, obj: Any) -> None:
        """Write one JSON object followed by a single ``\\n`` (normal framing)."""
        self._s2c.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))

    def send_split(self, obj: Any, *, at: int) -> None:
        """Write one newline-terminated object across TWO writes, split at byte ``at``.

        Exercises the reader's mid-line accumulation: the first write contains a
        partial object; the second completes it (including the trailing newline).
        """
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self._s2c.write(data[:at])
        self._s2c.write(data[at:])

    def send_packed(self, *objs: Any) -> None:
        """Write SEVERAL newline-delimited objects in a SINGLE write.

        Exercises the reader's multiple-objects-per-read path.
        """
        blob = "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in objs)
        self._s2c.write(blob.encode("utf-8"))

    def send_garbage_line(self, text: str) -> None:
        """Write a non-JSON line (+ newline) the reader must skip without crashing."""
        self._s2c.write((text + "\n").encode("utf-8"))

    # -- teardown -----------------------------------------------------------

    async def aclose(self) -> None:
        """Close both pipes (drives the transport to EOF) and the transport. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # Closing s2c gives the transport reader pump EOF.
        self._s2c.close()
        self._c2s.close()
        if self._transport is not None:
            await self._transport.aclose()

    def close_server_stream(self) -> None:
        """Close ONLY the s2c (server→client) pipe: simulates the peer dying.

        The transport's reader pump sees EOF and should fail all pending
        requests with :class:`TransportClosed` WITHOUT an explicit ``aclose``.
        """
        self._s2c.close()
