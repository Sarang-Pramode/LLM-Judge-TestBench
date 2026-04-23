"""Evaluation runner.

High-level flow per run:

1. User assembles a :class:`RunPlan`: rows, pillars, bundles, LLM
   clients, optional KB, concurrency, and cache.
2. :meth:`EvaluationRunner.run` validates the plan, builds one judge
   instance per pillar, plans ``(row, pillar)`` tasks, and fans them
   across a thread pool.
3. Each task:
   - Looks up the cache. On hit, reuses the cached outcome.
   - Otherwise, runs the judge (which itself is failure-safe: errors
     land on ``JudgeOutcome.error`` rather than raising).
   - Stores successful outcomes back into the cache.
4. Results are sorted deterministically (``row_index``, ``pillar``)
   and wrapped in a :class:`RunResult` with a summary.

Design notes:

- Judges are instantiated once per run. Thread safety: judge state
  that matters for correctness (config, rubric, llm) is read-only.
  :class:`~src.judges.completeness.CompletenessJudge` caches KB
  matches by ``record_id`` - that cache is check-then-set, but even
  if two threads race the match result is deterministic, so the
  worst case is one duplicate match computation.
- Each :class:`LLMClient` maps to one :class:`ProviderThrottle`. Two
  pillars sharing the same client share the semaphore automatically
  (``id(llm)`` keyed), two pillars with distinct clients get their
  own throttles. This is what makes the per-provider rate-limit cap
  mean what it says.
- Progress and per-outcome callbacks run on worker threads. Users
  must keep them short and thread-safe.
- The runner never raises on partial failures. It returns all
  outcomes and lets the caller decide what to do. The only things
  that raise are misconfigurations (bad plan, unknown pillar).
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.core.exceptions import JudgeExecutionError
from src.core.types import NormalizedRow, RunContext
from src.judges.base import BaseJudge, JudgeOutcome
from src.judges.config import JudgeBundle
from src.judges.registry import build_judge
from src.llm.base import LLMClient, LLMUsage
from src.orchestration.batching import TaskSpec, plan_tasks
from src.orchestration.caching import NoCache, OutcomeCache, compute_cache_key
from src.orchestration.concurrency import (
    ConcurrencyPolicy,
    ProviderThrottle,
    acquire_throttle,
    execute_parallel,
)

if TYPE_CHECKING:
    from src.completeness.models import CompletenessKB

__all__ = [
    "EvaluationRunner",
    "RunPlan",
    "RunResult",
    "RunSummary",
]


OnOutcome = Callable[[JudgeOutcome], None]
OnProgress = Callable[[int, int], None]


# ---------------------------------------------------------------------------
# Plan + result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunPlan:
    """Everything the runner needs to execute one evaluation pass.

    Attributes:
        rows: Ordered dataset. Row order drives output order.
        pillars: Pillars to run. Must all have a matching entry in
            ``bundles`` and ``llm_by_pillar``. Duplicates rejected.
        bundles: ``pillar -> JudgeBundle`` (config + rubric).
        llm_by_pillar: ``pillar -> LLMClient``. Two pillars may share
            a client; the runner throttles per client.
        run_context: Per-run metadata passed to every judge call.
        kb: Optional completeness KB. Forwarded only to pillars whose
            judge class accepts it (currently ``completeness``).
        concurrency: Threading + per-provider rate-limit knobs.
        cache: Outcome cache. :class:`NoCache` by default.
        on_outcome: Optional callback invoked once per finished task
            (success or failure). Runs on a worker thread.
        on_progress: Optional ``(done, total)`` callback. Runs on a
            worker thread after each task completes.
    """

    rows: Sequence[NormalizedRow]
    pillars: Sequence[str]
    # ``Mapping`` (not ``dict``) keeps these invariant-free so callers
    # can pass ``dict[str, MockLLMClient]`` without mypy's dict
    # invariance fighting them. The runner only reads.
    bundles: Mapping[str, JudgeBundle]
    llm_by_pillar: Mapping[str, LLMClient]
    run_context: RunContext
    kb: CompletenessKB | None = None
    concurrency: ConcurrencyPolicy = field(default_factory=ConcurrencyPolicy)
    cache: OutcomeCache = field(default_factory=NoCache)
    on_outcome: OnOutcome | None = None
    on_progress: OnProgress | None = None


@dataclass(frozen=True)
class RunSummary:
    """Aggregate per-run statistics.

    Kept small and flat on purpose: downstream dashboards (Stage 9) and
    observability loggers (Stage 10) re-derive richer views from the
    full outcome list. The summary is the "at-a-glance" health check.
    """

    total_tasks: int
    succeeded: int
    failed: int
    cache_hits: int
    aborted: int
    duration_s: float
    latency_ms_p50: float
    latency_ms_p95: float
    total_input_tokens: int
    total_output_tokens: int
    pillar_stats: dict[str, tuple[int, int]]  # pillar -> (succeeded, failed)


@dataclass(frozen=True)
class RunResult:
    """Final output of :meth:`EvaluationRunner.run`.

    Outcomes are sorted by ``(row_index, pillar)`` so two runs with
    the same plan produce byte-identical outcome sequences (cache
    hits aside), which matters for snapshot tests and export diffs.
    """

    run_id: str
    started_at: datetime
    finished_at: datetime
    outcomes: list[JudgeOutcome]
    summary: RunSummary


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class EvaluationRunner:
    """Parallel evaluation driver.

    Stateless between calls: construct one ``EvaluationRunner`` and
    reuse it across runs. All per-run state lives on :class:`RunPlan`
    and the returned :class:`RunResult`.
    """

    def run(self, plan: RunPlan) -> RunResult:
        """Execute ``plan`` and return a :class:`RunResult`.

        Does NOT raise on partial failures. Raises only for
        misconfiguration (missing pillar wiring, invalid plan).
        """
        self._validate(plan)
        started_at = datetime.now(UTC)
        t0 = time.perf_counter()

        judges = self._build_judges(plan)
        throttles = self._build_throttles(plan)
        kb_fingerprint = plan.kb.fingerprint() if plan.kb is not None else None

        tasks = plan_tasks(plan.rows, plan.pillars)
        total = len(tasks)
        progress = _ProgressCounter(total=total, callback=plan.on_progress)
        metrics = _RunMetrics()

        def _runner(spec: TaskSpec) -> JudgeOutcome:
            return self._execute_task(
                spec=spec,
                plan=plan,
                judges=judges,
                throttles=throttles,
                kb_fingerprint=kb_fingerprint,
                metrics=metrics,
                progress=progress,
            )

        callables = [self._bind_task(_runner, spec) for spec in tasks]

        outcomes = execute_parallel(
            callables,
            policy=plan.concurrency,
            on_complete=plan.on_outcome,
        )

        outcomes_sorted = self._sort_outcomes(tasks, outcomes)
        finished_at = datetime.now(UTC)
        duration_s = time.perf_counter() - t0

        summary = metrics.build_summary(
            outcomes=outcomes_sorted,
            total_tasks=total,
            duration_s=duration_s,
        )

        return RunResult(
            run_id=plan.run_context.run_id,
            started_at=started_at,
            finished_at=finished_at,
            outcomes=outcomes_sorted,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate(self, plan: RunPlan) -> None:
        if not plan.rows:
            raise JudgeExecutionError("RunPlan.rows is empty; nothing to evaluate.")
        if not plan.pillars:
            raise JudgeExecutionError("RunPlan.pillars is empty; nothing to evaluate.")
        seen: set[str] = set()
        for pillar in plan.pillars:
            if pillar in seen:
                raise JudgeExecutionError(f"Pillar {pillar!r} appears twice in RunPlan.pillars.")
            seen.add(pillar)
            if pillar not in plan.bundles:
                raise JudgeExecutionError(f"RunPlan.bundles is missing pillar {pillar!r}.")
            if pillar not in plan.llm_by_pillar:
                raise JudgeExecutionError(f"RunPlan.llm_by_pillar is missing pillar {pillar!r}.")
        # Surface KB/context consistency early. If a KB is supplied but
        # run_context.kb_version is unset, log a hint by raising a clear
        # error - Stage 10 observability will log this properly, but
        # until then this at least forces the caller to decide.
        if plan.kb is not None and plan.run_context.kb_version is None:
            raise JudgeExecutionError(
                "RunPlan provides a completeness KB but RunContext.kb_version "
                "is None. Set RunContext.kb_version to kb.fingerprint() for "
                "reproducibility."
            )

    def _build_judges(self, plan: RunPlan) -> dict[str, BaseJudge]:
        """Instantiate one judge per pillar, up-front.

        Judges are reused across all rows; the runner never rebuilds a
        judge mid-run. Keeps memory flat and lets completeness's match
        cache survive across rows (good: same KB).
        """
        judges: dict[str, BaseJudge] = {}
        for pillar in plan.pillars:
            judges[pillar] = build_judge(
                pillar,
                bundle=plan.bundles[pillar],
                llm=plan.llm_by_pillar[pillar],
                kb=plan.kb,
            )
        return judges

    def _build_throttles(self, plan: RunPlan) -> dict[int, ProviderThrottle]:
        """One throttle per distinct ``LLMClient`` instance.

        Keying off ``id(llm)`` means pillars that share a client share
        a semaphore (correct for rate limiting) while pillars with
        distinct clients run without interference.
        """
        throttles: dict[int, ProviderThrottle] = {}
        for pillar in plan.pillars:
            client = plan.llm_by_pillar[pillar]
            key = id(client)
            if key not in throttles:
                throttles[key] = acquire_throttle(plan.concurrency.per_provider_limit)
        return throttles

    def _bind_task(
        self,
        runner: Callable[[TaskSpec], JudgeOutcome],
        spec: TaskSpec,
    ) -> Callable[[], JudgeOutcome]:
        """Capture the spec in a zero-arg lambda for ``execute_parallel``."""
        return lambda: runner(spec)

    def _execute_task(
        self,
        *,
        spec: TaskSpec,
        plan: RunPlan,
        judges: dict[str, BaseJudge],
        throttles: dict[int, ProviderThrottle],
        kb_fingerprint: str | None,
        metrics: _RunMetrics,
        progress: _ProgressCounter,
    ) -> JudgeOutcome:
        """Execute one (row, pillar) task. Never raises."""
        row = plan.rows[spec.row_index]
        judge = judges[spec.pillar]
        bundle = plan.bundles[spec.pillar]
        llm = plan.llm_by_pillar[spec.pillar]

        cache_key = compute_cache_key(
            pillar=spec.pillar,
            bundle=bundle,
            llm=llm,
            row=row,
            kb_fingerprint=kb_fingerprint,
        )

        cached = plan.cache.get(cache_key)
        if cached is not None:
            metrics.record_cache_hit(spec.pillar)
            progress.tick()
            return cached

        throttle = throttles[id(llm)]
        try:
            with throttle:
                outcome = judge.run(row, run_context=plan.run_context)
        except Exception as exc:  # pragma: no cover - judge.run is already safe
            # Defensive: judge.run is contracted to never raise on
            # provider/parse issues. If something deep (a new pillar's
            # validator) does raise, still deliver an outcome so the
            # run completes.
            outcome = JudgeOutcome(
                pillar=spec.pillar,
                record_id=row.record_id,
                latency_ms=0.0,
                attempts=0,
                usage=LLMUsage(),
                model_name=llm.model_name,
                run_id=plan.run_context.run_id,
                error=f"Unhandled judge exception: {exc}",
                error_type=type(exc).__name__,
            )

        if outcome.ok:
            plan.cache.set(cache_key, outcome)

        metrics.record(outcome)
        progress.tick()
        return outcome

    def _sort_outcomes(
        self,
        tasks: list[TaskSpec],
        outcomes: list[JudgeOutcome],
    ) -> list[JudgeOutcome]:
        """Sort outcomes by (row_index, pillar) for deterministic output.

        ``execute_parallel`` already returns in submission order, which
        is ``(row_index, pillar)``. We rebuild explicitly anyway so a
        future change to the pool (say, swapping in an async backend)
        can't silently break deterministic ordering.
        """
        if len(tasks) != len(outcomes):  # pragma: no cover - invariant
            raise JudgeExecutionError(
                f"Internal error: task/outcome length mismatch ({len(tasks)} vs {len(outcomes)})."
            )
        pairs = sorted(
            zip(tasks, outcomes, strict=True),
            key=lambda pair: (pair[0].row_index, pair[0].pillar),
        )
        return [outcome for _spec, outcome in pairs]


# ---------------------------------------------------------------------------
# Support classes
# ---------------------------------------------------------------------------


class _ProgressCounter:
    """Thread-safe done/total counter with an optional user callback."""

    __slots__ = ("_callback", "_done", "_lock", "_total")

    def __init__(self, *, total: int, callback: OnProgress | None) -> None:
        self._total = total
        self._done = 0
        self._lock = threading.Lock()
        self._callback = callback

    def tick(self) -> None:
        with self._lock:
            self._done += 1
            done = self._done
        if self._callback is not None:
            # Callback runs outside the lock so a slow callback doesn't
            # stall other workers that want to tick.
            self._callback(done, self._total)


class _RunMetrics:
    """Thread-safe aggregator the runner feeds during execution."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache_hits: dict[str, int] = {}
        self._pillar_success: dict[str, int] = {}
        self._pillar_fail: dict[str, int] = {}

    def record_cache_hit(self, pillar: str) -> None:
        with self._lock:
            self._cache_hits[pillar] = self._cache_hits.get(pillar, 0) + 1
            # A cache hit is always a successful outcome by
            # construction (NoCache never stores, InMemory filters on
            # ``outcome.ok``). Count it toward the pillar's success
            # tally so summaries match the user's mental model.
            self._pillar_success[pillar] = self._pillar_success.get(pillar, 0) + 1

    def record(self, outcome: JudgeOutcome) -> None:
        with self._lock:
            if outcome.ok:
                self._pillar_success[outcome.pillar] = (
                    self._pillar_success.get(outcome.pillar, 0) + 1
                )
            else:
                self._pillar_fail[outcome.pillar] = self._pillar_fail.get(outcome.pillar, 0) + 1

    def build_summary(
        self,
        *,
        outcomes: list[JudgeOutcome],
        total_tasks: int,
        duration_s: float,
    ) -> RunSummary:
        succeeded = sum(1 for o in outcomes if o.ok)
        failed = total_tasks - succeeded
        cache_hits = sum(self._cache_hits.values())
        # Providers that don't report usage leave these as None; treat
        # missing reports as zero-contribution so the summary always has
        # a valid total. Consumers that need "was usage reported?" can
        # inspect the raw outcomes.
        input_tokens = sum((o.usage.input_tokens or 0) for o in outcomes)
        output_tokens = sum((o.usage.output_tokens or 0) for o in outcomes)
        latencies = sorted(o.latency_ms for o in outcomes)

        pillar_stats: dict[str, tuple[int, int]] = {}
        all_pillars = set(self._pillar_success) | set(self._pillar_fail)
        for pillar in sorted(all_pillars):
            pillar_stats[pillar] = (
                self._pillar_success.get(pillar, 0),
                self._pillar_fail.get(pillar, 0),
            )

        return RunSummary(
            total_tasks=total_tasks,
            succeeded=succeeded,
            failed=failed,
            cache_hits=cache_hits,
            aborted=0,  # Reserved for fail_fast_after; not wired in v1.
            duration_s=duration_s,
            latency_ms_p50=_percentile(latencies, 50),
            latency_ms_p95=_percentile(latencies, 95),
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            pillar_stats=pillar_stats,
        )


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile on a sorted list.

    Returns 0.0 for empty input. We implement this inline to avoid
    pulling numpy just for two numbers per run.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_values[lo]
    frac = rank - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac
