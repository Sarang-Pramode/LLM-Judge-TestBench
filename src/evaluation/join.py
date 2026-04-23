"""Join :class:`NormalizedRow` with :class:`JudgeOutcome` into a flat,
metric-ready :class:`ScoredItem` sequence.

All downstream metric / slice / reviewer code operates on
``ScoredItem`` - not on ``NormalizedRow`` or ``JudgeOutcome`` directly.
That decoupling is deliberate:

- Metric code stays pure: it only sees integer score pairs plus a few
  slice strings, which makes golden-value unit tests trivial.
- Adding a new slice dimension (say ``domain``) in the future only
  requires adding a field here; every consumer picks it up for free.
- Filtering rules (missing label, failed outcome, out-of-range score)
  live in *one* place and are applied consistently.

Graceful handling contract:

- A row with ``label_<pillar>`` == ``None`` is dropped from that
  pillar's report (but not from other pillars).
- An outcome with ``result is None`` (error path) is dropped.
- An outcome whose score is outside ``[SCORE_MIN, SCORE_MAX]`` is
  rejected at :class:`NormalizedRow` / :class:`JudgeResult`
  construction, so we don't re-check here; we trust the invariants.
- An outcome for a ``record_id`` not in the row set is flagged via
  :class:`JoinStats.orphan_outcomes` rather than silently dropped.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from src.core.constants import PILLARS
from src.core.types import NormalizedRow
from src.judges.base import JudgeOutcome

__all__ = [
    "JoinStats",
    "JoinedDataset",
    "ScoredItem",
    "join_outcomes_with_labels",
]


@dataclass(frozen=True)
class ScoredItem:
    """One ``(row, outcome, pillar)`` triple with its SME label.

    This is the unit of work all Stage 8 modules iterate over. Keeping
    the slice-friendly fields (``category``, ``reviewer_name``, ...)
    here - rather than forcing consumers to re-read from
    :class:`NormalizedRow` - means the join layer is the single source
    of truth for which slice keys are available.
    """

    record_id: str
    pillar: str
    judge_score: int
    human_score: int
    category: str
    #: None when the row had no reviewer metadata. Reviewer analytics
    #: filters on this.
    reviewer_name: str | None = None
    reviewer_id: str | None = None
    intent: str | None = None
    topic: str | None = None
    conversation_id: str | None = None
    model_name: str | None = None

    @property
    def distance(self) -> int:
        """Absolute score distance. Used by every metric downstream."""
        return abs(self.judge_score - self.human_score)


@dataclass(frozen=True)
class JoinStats:
    """Diagnostics emitted alongside the joined sequence.

    These are NOT errors. A run can perfectly well end with many
    ``missing_labels`` (unlabelled data is normal) or many
    ``failed_outcomes`` (provider hiccups). The UI / observability
    layer surfaces them so users notice at a glance how much data
    actually contributed to the metrics.
    """

    #: Total (row, pillar) pairs considered (len(rows) x len(pillars)).
    considered: int = 0
    #: Dropped because the row had no SME label for that pillar.
    missing_labels: int = 0
    #: Dropped because the judge outcome had no result (provider/parse error).
    failed_outcomes: int = 0
    #: Dropped because no outcome existed for the (record_id, pillar).
    #: Distinct from ``failed_outcomes`` - here we're saying the runner
    #: never even produced an outcome for that pair.
    missing_outcomes: int = 0
    #: Outcomes that could not be paired to any row in ``rows``.
    orphan_outcomes: int = 0
    #: Successful joins.
    paired: int = 0
    #: Pillars actually encountered in the join output.
    pillars_seen: frozenset[str] = frozenset()


@dataclass(frozen=True)
class JoinedDataset:
    """Result of :func:`join_outcomes_with_labels`.

    Wrapping the list + stats in one object keeps Stage 8 call sites
    clean (``joined.items`` / ``joined.stats``) and lets observability
    Stage 10 log ``stats`` without re-iterating the item list.
    """

    items: list[ScoredItem] = field(default_factory=list)
    stats: JoinStats = field(default_factory=JoinStats)

    def for_pillar(self, pillar: str) -> list[ScoredItem]:
        """Filter to items for a specific pillar.

        Kept as a method (not a cached property) because downstream
        callers typically iterate all pillars once; caching would
        trade correctness for zero measurable win.
        """
        return [it for it in self.items if it.pillar == pillar]


# ---------------------------------------------------------------------------
# Join implementation
# ---------------------------------------------------------------------------


def join_outcomes_with_labels(
    rows: Sequence[NormalizedRow],
    outcomes: Iterable[JudgeOutcome],
    *,
    pillars: Sequence[str] | None = None,
) -> JoinedDataset:
    """Pair every successful outcome with its row and SME label.

    Args:
        rows: Normalized rows used for the run.
        outcomes: Outcomes produced by the runner (successes + failures).
        pillars: Restrict joining to these pillars. Defaults to
            :data:`src.core.constants.PILLARS` so callers can pass the
            runner's outcome list verbatim.

    Returns:
        :class:`JoinedDataset` holding the successfully paired
        :class:`ScoredItem` sequence plus :class:`JoinStats`.
    """
    active_pillars = set(pillars or PILLARS)
    row_by_id: dict[str, NormalizedRow] = {r.record_id: r for r in rows}

    outcomes_list = list(outcomes)
    items: list[ScoredItem] = []
    missing_labels = 0
    failed_outcomes = 0
    orphan_outcomes = 0
    seen_keys: set[tuple[str, str]] = set()
    considered = len(rows) * len(active_pillars)

    for outcome in outcomes_list:
        if outcome.pillar not in active_pillars:
            # Quietly ignore - user explicitly restricted pillars.
            continue
        if outcome.result is None:
            failed_outcomes += 1
            continue
        row = row_by_id.get(outcome.record_id)
        if row is None:
            orphan_outcomes += 1
            continue
        label = getattr(row, f"label_{outcome.pillar}", None)
        if label is None:
            missing_labels += 1
            continue

        seen_keys.add((outcome.record_id, outcome.pillar))
        items.append(
            ScoredItem(
                record_id=row.record_id,
                pillar=outcome.pillar,
                judge_score=outcome.result.score,
                human_score=int(label),
                category=row.category,
                reviewer_name=row.reviewer_name,
                reviewer_id=row.reviewer_id,
                intent=row.intent,
                topic=row.topic,
                conversation_id=row.conversation_id,
                model_name=row.model_name,
            )
        )

    # Count (row, pillar) pairs the runner never scored at all. Useful
    # for spotting partial runs: e.g. a user enabled 6 pillars but only
    # 5 outcomes per row showed up.
    expected_keys = {(r.record_id, p) for r in rows for p in active_pillars}
    outcome_keys = {(o.record_id, o.pillar) for o in outcomes_list if o.pillar in active_pillars}
    missing_outcomes = len(expected_keys - outcome_keys)

    stats = JoinStats(
        considered=considered,
        missing_labels=missing_labels,
        failed_outcomes=failed_outcomes,
        missing_outcomes=missing_outcomes,
        orphan_outcomes=orphan_outcomes,
        paired=len(items),
        pillars_seen=frozenset({it.pillar for it in items}),
    )
    return JoinedDataset(items=items, stats=stats)
