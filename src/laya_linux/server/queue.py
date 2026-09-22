"""Bounded request queue and serialized execution (architecture §5.10, §8.4).

One execution owner per device: predictions for an alias run through a fixed
number of worker tasks, so an HTTP server never duplicates model copies for
concurrency. The queue has a hard capacity — overflow is rejected immediately
with ``QUEUE_FULL`` instead of consuming unbounded memory — and each job has a
deadline after which the caller receives ``SERVER_BUSY``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..errors import QueueFullError, ServerBusyError


@dataclass
class _Job:
    fn: Callable[..., Any]
    args: tuple
    future: asyncio.Future
    discarded: bool = field(default=False)


class BoundedExecutor:
    """FIFO queue + fixed workers; blocking work runs in threads."""

    def __init__(self, capacity: int, concurrency: int = 1):
        if capacity < 1 or concurrency < 1:
            raise ValueError("capacity and concurrency must be positive integers")
        self.capacity = capacity
        self.concurrency = concurrency
        self._queue: asyncio.Queue[_Job] = asyncio.Queue(maxsize=capacity)
        self._workers: list[asyncio.Task] = []

    def start(self) -> None:
        if not self._workers:
            self._workers = [asyncio.create_task(self._worker()) for _ in range(self.concurrency)]

    async def stop(self) -> None:
        for worker in self._workers:
            worker.cancel()
        self._workers = []

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    async def submit(self, fn: Callable[..., Any], *args: Any, timeout: float | None = None) -> Any:
        """Enqueue ``fn(*args)`` and await its result.

        Raises ``QueueFullError`` immediately when the queue is at capacity and
        ``ServerBusyError`` when the job does not finish within ``timeout``.
        A timed-out job keeps running to completion in its worker (the heavy
        call cannot be safely aborted) but its result is discarded.
        """
        self.start()
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        job = _Job(fn=fn, args=args, future=future)
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            raise QueueFullError("Server queue is full; retry later") from None
        try:
            return await asyncio.wait_for(future, timeout)
        except TimeoutError:
            job.discarded = True
            raise ServerBusyError("Server is busy; the request exceeded its execution deadline") from None

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job.discarded or job.future.cancelled():
                    continue
                result = await asyncio.to_thread(job.fn, *job.args)
                if not job.future.done():
                    job.future.set_result(result)
            except BaseException as exc:  # noqa: BLE001 - forwarded to the awaiting caller
                if not job.future.done() and not job.future.cancelled():
                    job.future.set_exception(exc)
            finally:
                self._queue.task_done()


class AgentRegistry:
    """Lazily loaded, shared Agent per model alias (one model copy per alias).

    Loading happens on first use inside the caller's worker thread, so the
    asyncio loop never blocks on model load. ``threading.Lock`` guards
    concurrent first loads; afterwards every worker shares the same instance,
    so there is never more than one model copy per alias per process.
    """

    def __init__(self, model_paths: dict[str, str], device: str | None = None, dtype: str = "auto"):
        self._paths = dict(model_paths)
        self._device = device
        self._dtype = dtype
        self._agents: dict[str, Any] = {}
        self._lock = threading.Lock()

    def __contains__(self, alias: str) -> bool:
        return alias in self._paths

    def aliases(self) -> list[str]:
        return sorted(self._paths)

    def describe_loaded(self) -> list[dict[str, Any]]:
        return [{"alias": a, "loaded": a in self._agents} for a in sorted(self._paths)]

    def get(self, alias: str):
        """Return the (possibly not yet loaded) Agent for ``alias``.

        Raises ``KeyError`` for unknown aliases; callers translate that into a
        stable client-facing error. Actual loading runs here — call from a
        worker thread, never from the event loop.
        """
        if alias not in self._paths:
            raise KeyError(alias)
        with self._lock:
            if alias not in self._agents:
                from ..agent import Agent

                self._agents[alias] = Agent(self._paths[alias], device=self._device, dtype=self._dtype)
            return self._agents[alias]

    def preload(self) -> None:
        for alias in self._paths:
            self.get(alias)
