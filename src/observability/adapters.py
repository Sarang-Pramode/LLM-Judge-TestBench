"""Adapters that wire the observability loggers to the runner.

The runner in :mod:`src.orchestration.runner` accepts two optional
callbacks: ``on_outcome`` (fires once per finished judge task) and
``on_progress`` (fires with ``(done, total)`` after each task). This
module turns those hooks into Langfuse observations and MLflow metric
updates without the runner needing to know either backend exists.

Thread-safety note
------------------

Both callbacks run on worker threads. :class:`LangfuseTracer` is
thread-safe by design; :class:`MLflowLogger` methods are not called
from worker threads here - they're invoked by the page code around the
run (start + end), not per-row.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.core.types import NormalizedRow
from src.judges.base import JudgeOutcome
from src.observability.langfuse_tracer import LangfuseTracer

__all__ = [
    "ObservabilityCallbacks",
    "build_observability_callbacks",
]


OnOutcome = Callable[[JudgeOutcome], None]
OnProgress = Callable[[int, int], None]


@dataclass(frozen=True)
class ObservabilityCallbacks:
    """Pair of runner callbacks wired to observability backends.

    ``on_outcome`` is always set (possibly to a no-op); ``on_progress``
    is set only when a ``progress_cb`` was provided to the builder, so
    callers that already pass their own progress callback don't get it
    silently replaced.
    """

    on_outcome: OnOutcome
    on_progress: OnProgress | None


def build_observability_callbacks(
    *,
    rows: Sequence[NormalizedRow],
    tracer: LangfuseTracer | None = None,
    progress_cb: OnProgress | None = None,
) -> ObservabilityCallbacks:
    """Build runner callbacks from observability loggers.

    Args:
        rows: Normalized rows the runner will process. Used to resolve
            a ``record_id`` back to a :class:`NormalizedRow` without
            forcing the tracer to carry state.
        tracer: Optional Langfuse tracer. Its :meth:`log_outcome` is
            invoked per finished task. A ``None`` tracer produces a
            no-op outcome callback (so callers can unconditionally pass
            the result to the runner).
        progress_cb: Optional pass-through progress callback (the UI's
            ``st.progress`` driver typically supplies this).

    The outcome callback is guarded internally with a lock-free fast
    path plus exception swallowing, because runner workers must never
    see observability errors.
    """
    by_record: dict[str, NormalizedRow] = {row.record_id: row for row in rows}
    # Guard against programming errors in the tracer leaking into the
    # worker thread and aborting the run.
    error_lock = threading.Lock()
    reported_errors: set[str] = set()

    def _safe_on_outcome(outcome: JudgeOutcome) -> None:
        if tracer is None or not tracer.enabled:
            return
        try:
            row = by_record.get(outcome.record_id)
            if row is None:
                return
            tracer.log_outcome(outcome, row)
        except Exception as exc:  # defense-in-depth; tracer is already guarded
            key = type(exc).__name__
            with error_lock:
                if key in reported_errors:
                    return
                reported_errors.add(key)
            # We don't import logging here to avoid another surface;
            # the tracer's own guards will already have warned.

    return ObservabilityCallbacks(on_outcome=_safe_on_outcome, on_progress=progress_cb)
