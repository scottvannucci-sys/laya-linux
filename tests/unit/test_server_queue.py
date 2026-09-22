"""Bounded executor: overload rejection and deadline behavior (unit)."""

import asyncio

import pytest

from laya_linux.errors import QueueFullError, ServerBusyError
from laya_linux.server.queue import BoundedExecutor


def test_submit_runs_and_returns():
    async def main():
        ex = BoundedExecutor(capacity=2, concurrency=1)
        return await ex.submit(lambda a, b: a + b, 2, 3)

    assert asyncio.run(main()) == 5


def test_queue_full_rejected_immediately():
    import threading

    async def main():
        ex = BoundedExecutor(capacity=1, concurrency=1)
        gate = threading.Event()

        def blocker():
            gate.wait(2.0)  # truly blocks the worker thread

        task0 = asyncio.create_task(ex.submit(blocker))
        await asyncio.sleep(0.1)  # worker now busy inside blocker
        blocked = asyncio.create_task(ex.submit(lambda: "queued"))
        await asyncio.sleep(0.1)  # queued (queue at capacity 1)
        try:
            with pytest.raises(QueueFullError):
                await ex.submit(lambda: None)  # capacity exceeded
        finally:
            gate.set()
        await task0
        return await blocked

    assert asyncio.run(main()) == "queued"


def test_timeout_yields_server_busy():
    async def main():
        ex = BoundedExecutor(capacity=4, concurrency=1)
        with pytest.raises(ServerBusyError):
            await ex.submit(lambda: (__import__("time").sleep(0.3), "x")[1], timeout=0.05)
        # The abandoned job still finishes in the worker without crashing anything.
        await asyncio.sleep(0.4)
        return await ex.submit(lambda: "recovered")

    assert asyncio.run(main()) == "recovered"


def test_exception_propagates_to_caller():
    def boom():
        raise ValueError("worker failed")

    async def main():
        ex = BoundedExecutor(capacity=2, concurrency=1)
        with pytest.raises(ValueError, match="worker failed"):
            await ex.submit(boom)

    asyncio.run(main())


def test_invalid_capacity_rejected():
    with pytest.raises(ValueError):
        BoundedExecutor(capacity=0)
    with pytest.raises(ValueError):
        BoundedExecutor(capacity=4, concurrency=0)
