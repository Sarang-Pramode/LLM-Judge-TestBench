"""Thread-pool concurrency primitives for the evaluation runner.

Judges do I/O-bound work (LLM calls), so threads give us essentially
linear speedup up to the provider's rate-limit ceiling. We deliberately
avoid asyncio: the existing :class:`~src.llm.base.LLMClient` interface
is synchronous, and forcing every pillar judge to become ``async`` just
so the runner can be async would infect the whole stack. Threads keep
the judge signature clean while still giving us real parallelism.

Provided primitives:

- :class:`ConcurrencyPolicy` - user-facing knobs (max_workers, per-provider
  concurrent-call cap, and a dev-only "stop submitting after N failures"
  circuit breaker).
- :class:`ProviderThrottle` - opaque handle returned by
  :func:`acquire_throttle`; the runner maps each :class:`LLMClient` to
  one throttle so every judge that shares a client shares the semaphore.
- :func:`execute_parallel` - run a list of zero-arg callables through a
  bounded thread pool, streaming outputs through an optional
  ``on_complete`` hook, and returning results in submission order.

None of this is judge- or pillar-aware on purpose. The runner composes
these primitives; this module stays reusable for future batched
operations (e.g. dataset fingerprinting, bulk export).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

__all__ = [
    "ConcurrencyPolicy",
    "ProviderThrottle",
    "RunAborted",
    "acquire_throttle",
    "execute_parallel",
]


@dataclass(frozen=True)
class ConcurrencyPolicy:
    """How aggressively to parallelise a run.

    Attributes:
        max_workers: Upper bound on OS-level worker threads. Defaults
            to 8, which matches typical free-tier LLM rate limits and
            avoids accidentally saturating a laptop with 100+ threads.
        per_provider_limit: Maximum concurrent in-flight calls to any
            single LLM provider (mapped by client identity). ``None``
            disables the cap. Typically set just below the provider's
            documented rate limit. The cap composes with
            ``max_workers``: a worker that wants to call a throttled
            provider blocks on the semaphore rather than spinning.
        fail_fast_after: When non-``None``, the runner will stop
            scheduling *new* work after this many task failures are
            seen. Tasks already submitted to the pool still complete
            (cancellation-on-error is brittle with blocking IO). This
            is a cost/time guard, not a correctness guard.
    """

    max_workers: int = 8
    per_provider_limit: int | None = None
    fail_fast_after: int | None = None

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError(f"ConcurrencyPolicy.max_workers must be >= 1; got {self.max_workers}.")
        if self.per_provider_limit is not None and self.per_provider_limit < 1:
            raise ValueError(
                "ConcurrencyPolicy.per_provider_limit must be None or >= 1; "
                f"got {self.per_provider_limit!r}."
            )
        if self.fail_fast_after is not None and self.fail_fast_after < 1:
            raise ValueError(
                "ConcurrencyPolicy.fail_fast_after must be None or >= 1; "
                f"got {self.fail_fast_after!r}."
            )


class ProviderThrottle:
    """A bounded semaphore with a no-op contract when disabled.

    Use :func:`acquire_throttle` to construct one per provider; do not
    share across distinct ``LLMClient`` instances unless they really
    sit behind the same provider/rate-limit pool.
    """

    __slots__ = ("_sema",)

    def __init__(self, limit: int | None) -> None:
        self._sema: threading.BoundedSemaphore | None = (
            threading.BoundedSemaphore(limit) if limit is not None else None
        )

    def __enter__(self) -> ProviderThrottle:
        if self._sema is not None:
            self._sema.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._sema is not None:
            self._sema.release()


def acquire_throttle(limit: int | None) -> ProviderThrottle:
    """Return a :class:`ProviderThrottle` capped at ``limit`` (or unbounded)."""
    return ProviderThrottle(limit)


class RunAborted(RuntimeError):
    """Raised inside a skipped task after ``fail_fast_after`` was tripped.

    Callers catch this at the task boundary and mark the outcome as an
    aborted failure rather than letting it propagate up to the pool.
    """


def execute_parallel[T](
    tasks: Iterable[Callable[[], T]],
    *,
    policy: ConcurrencyPolicy,
    on_complete: Callable[[T], None] | None = None,
) -> list[T]:
    """Run ``tasks`` through a thread pool, returning results in order.

    Contract:

    - Each task is a zero-arg callable. Parallelism fans out across the
      task list; tasks themselves stay sync so judge code doesn't need
      to know about threading.
    - Results are returned in *submission* order, not completion order,
      so upstream callers get deterministic ordering regardless of how
      threads interleave.
    - ``on_complete`` runs once per task on the worker thread that
      produced the result. It is the hook the runner uses for progress
      counters / live dashboards; implementers MUST make the callback
      thread-safe.
    - Exceptions raised inside a task bubble up through the returned
      list as regular ``Future.exception()`` results would - except we
      re-raise here so the runner's own error handling (which catches
      inside the task body) is what surfaces errors. Tasks are
      therefore expected to catch their own errors and return a value
      that encodes them.
    - With ``max_workers == 1`` the pool runs tasks serially on a
      dedicated thread. We never inline on the caller thread; that
      would make progress callbacks and timeouts behave differently
      between serial and parallel runs.
    """
    task_list = list(tasks)
    if not task_list:
        return []

    results: list[T] = [None] * len(task_list)  # type: ignore[list-item]

    workers = min(policy.max_workers, len(task_list))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jtb-worker") as pool:
        future_to_index: dict[Future[T], int] = {
            pool.submit(task): idx for idx, task in enumerate(task_list)
        }
        for future in future_to_index:
            # Completion handled via add_done_callback so on_complete runs
            # on the worker that finished the task (lower latency for
            # live dashboards than waiting for the join below).
            idx = future_to_index[future]
            future.add_done_callback(_make_done_callback(idx, results, on_complete))

    # The executor's __exit__ already joined all workers. Any task that
    # raised would have stored an exception on its future; rehydrate now
    # so callers see it rather than the default-initialised slot.
    for future, idx in future_to_index.items():
        exc = future.exception()
        if exc is not None:
            raise exc
        results[idx] = future.result()

    return results


def _make_done_callback[T](
    index: int,
    results: list[T],
    on_complete: Callable[[T], None] | None,
) -> Callable[[Future[T]], None]:
    """Build a single-shot ``Future`` done-callback that stores the result."""

    def _cb(future: Future[T]) -> None:
        if future.cancelled():  # pragma: no cover - we never cancel futures
            return
        if future.exception() is not None:
            # Let execute_parallel's post-join loop re-raise with fuller
            # context; don't invoke on_complete for failures.
            return
        result = future.result()
        results[index] = result
        if on_complete is not None:
            on_complete(result)

    return _cb
