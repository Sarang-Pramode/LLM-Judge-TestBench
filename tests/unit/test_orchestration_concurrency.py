"""Unit tests for :mod:`src.orchestration.concurrency`."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from src.orchestration.concurrency import (
    ConcurrencyPolicy,
    acquire_throttle,
    execute_parallel,
)

# ---------------------------------------------------------------------------
# ConcurrencyPolicy validation
# ---------------------------------------------------------------------------


class TestConcurrencyPolicy:
    def test_defaults(self) -> None:
        policy = ConcurrencyPolicy()
        assert policy.max_workers == 8
        assert policy.per_provider_limit is None
        assert policy.fail_fast_after is None

    def test_max_workers_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            ConcurrencyPolicy(max_workers=0)

    def test_per_provider_limit_zero_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            ConcurrencyPolicy(per_provider_limit=0)

    def test_fail_fast_after_zero_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            ConcurrencyPolicy(fail_fast_after=0)


# ---------------------------------------------------------------------------
# ProviderThrottle
# ---------------------------------------------------------------------------


class TestProviderThrottle:
    def test_none_limit_is_a_noop(self) -> None:
        throttle = acquire_throttle(None)
        # Acquire several times without blocking; a real semaphore would
        # deadlock the second call on this synchronous code path.
        with throttle:
            with throttle:
                with throttle:
                    pass

    def test_bounded_limit_caps_concurrent_holders(self) -> None:
        throttle = acquire_throttle(2)
        in_flight = 0
        peak = 0
        gate = threading.Lock()
        release = threading.Event()
        done = threading.Event()
        started = 0
        started_lock = threading.Lock()

        def worker() -> None:
            nonlocal in_flight, peak, started
            with started_lock:
                started += 1
            with throttle:
                with gate:
                    in_flight += 1
                    peak = max(peak, in_flight)
                release.wait(timeout=1.0)
                with gate:
                    in_flight -= 1

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        # Wait a bit for two workers to be inside the throttle; the
        # other two must be blocked on the semaphore.
        time.sleep(0.05)
        assert peak <= 2
        release.set()
        for t in threads:
            t.join(timeout=2.0)
        done.set()
        assert peak == 2


# ---------------------------------------------------------------------------
# execute_parallel
# ---------------------------------------------------------------------------


class TestExecuteParallel:
    def test_empty_returns_empty(self) -> None:
        assert execute_parallel([], policy=ConcurrencyPolicy()) == []

    def test_results_preserve_submission_order(self) -> None:
        # Deliberately delay the first task so completion order does
        # NOT match submission order. We must still return in input
        # order.
        def make(i: int, delay: float) -> Callable[[], int]:
            def _task() -> int:
                time.sleep(delay)
                return i

            return _task

        tasks = [make(0, 0.05), make(1, 0.0), make(2, 0.0)]
        results = execute_parallel(tasks, policy=ConcurrencyPolicy(max_workers=4))
        assert results == [0, 1, 2]

    def test_on_complete_fires_for_every_task(self) -> None:
        seen: list[int] = []
        lock = threading.Lock()

        def _cb(value: int) -> None:
            with lock:
                seen.append(value)

        def make(n: int) -> Callable[[], int]:
            return lambda: n

        tasks: list[Callable[[], int]] = [make(n) for n in range(10)]
        execute_parallel(tasks, policy=ConcurrencyPolicy(max_workers=4), on_complete=_cb)
        assert sorted(seen) == list(range(10))

    def test_exception_in_task_is_reraised(self) -> None:
        def boom() -> int:
            raise RuntimeError("broken")

        with pytest.raises(RuntimeError, match="broken"):
            execute_parallel([boom], policy=ConcurrencyPolicy(max_workers=2))

    def test_serial_path_with_one_worker(self) -> None:
        order: list[int] = []

        def push(i: int) -> int:
            order.append(i)
            return i

        tasks = [(lambda i=i: push(i)) for i in range(5)]
        results = execute_parallel(tasks, policy=ConcurrencyPolicy(max_workers=1))
        assert results == [0, 1, 2, 3, 4]
        # With a single worker tasks run in strict submission order.
        assert order == [0, 1, 2, 3, 4]

    def test_parallelism_is_actually_real(self) -> None:
        """Sanity check: 4 tasks that sleep 100ms each should finish in
        well under 400ms when running 4-wide. If the helper silently
        went serial this test would balloon to ~400ms.
        """

        def sleeper() -> None:
            time.sleep(0.1)

        tasks = [sleeper for _ in range(4)]
        t0 = time.perf_counter()
        execute_parallel(tasks, policy=ConcurrencyPolicy(max_workers=4))
        elapsed = time.perf_counter() - t0
        # Generous threshold to avoid flakiness on overloaded CI boxes.
        assert elapsed < 0.35, f"suspiciously serial: {elapsed:.3f}s"
