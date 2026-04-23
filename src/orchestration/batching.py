"""Turn a dataset + pillar list into a flat, deterministic task plan.

The runner parallelises over ``(row, pillar)`` pairs rather than over
rows alone. That gives us two wins:

1. Pillars are heterogeneous - completeness in KB-informed mode is
   noticeably slower than toxicity because it builds a richer prompt.
   Flat fan-out lets the thread pool interleave fast and slow judges
   instead of waiting for the slowest pillar per row.
2. It makes per-pillar caching and retries trivial: every task has a
   unique, stable identity (``record_id``, ``pillar``).

This module is intentionally judge-agnostic. It only knows about row
identifiers and pillar names; all judge orchestration lives in the
runner.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.core.types import NormalizedRow

__all__ = [
    "TaskSpec",
    "chunk",
    "plan_tasks",
]


@dataclass(frozen=True)
class TaskSpec:
    """A single unit of work: evaluate one row with one pillar judge.

    Attributes:
        row_index: Position of the row in the input dataset. Used for
            stable sort order when displaying results.
        record_id: The row's business identifier (propagated to the
            judge outcome, cache key, and trace metadata).
        pillar: Pillar/judge name (e.g. ``"factual_accuracy"``).
    """

    row_index: int
    record_id: str
    pillar: str

    @property
    def task_id(self) -> str:
        """Human-readable identifier, handy in logs."""
        return f"{self.record_id}::{self.pillar}"


def plan_tasks(
    rows: Sequence[NormalizedRow],
    pillars: Sequence[str],
) -> list[TaskSpec]:
    """Build the full ``(row, pillar)`` cross-product plan.

    Ordering is ``(row_index, pillar_index)`` to match how a user reads
    a results table (row-by-row, left-to-right across pillar columns).
    The thread pool may still execute them out of order; we just want a
    deterministic baseline for post-run sorting.

    Raises:
        ValueError: if ``pillars`` is empty or contains duplicates. We
            reject duplicates up-front rather than silently dedupe so
            configuration mistakes surface loudly.
    """
    if not pillars:
        raise ValueError("plan_tasks requires at least one pillar.")
    seen: set[str] = set()
    for pillar in pillars:
        if pillar in seen:
            raise ValueError(f"Pillar {pillar!r} specified more than once in plan_tasks.")
        seen.add(pillar)

    tasks: list[TaskSpec] = []
    for row_index, row in enumerate(rows):
        for pillar in pillars:
            tasks.append(TaskSpec(row_index=row_index, record_id=row.record_id, pillar=pillar))
    return tasks


def chunk[T](items: Iterable[T], size: int) -> list[list[T]]:
    """Split ``items`` into sub-lists of length ``size`` (last may be short).

    Used by the runner for progress-reporting checkpoints: we emit the
    progress callback once per chunk rather than once per task, which
    keeps the UI responsive without spamming callbacks on big datasets.
    """
    if size < 1:
        raise ValueError(f"chunk size must be >= 1; got {size}.")
    batch: list[T] = []
    batches: list[list[T]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches
