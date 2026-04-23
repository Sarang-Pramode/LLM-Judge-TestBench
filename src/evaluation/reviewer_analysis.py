"""Reviewer-level analytics.

Product requirements (from the project brief) spell out:

- Samples per reviewer.
- Per-pillar average human score by reviewer.
- Judge-vs-reviewer disagreement rate.
- Within-1 agreement by reviewer.
- Large-miss rate by reviewer (off-by-2+).
- Reviewer-vs-reviewer disagreement when multiple reviewers score the
  same ``record_id``.

Activation rule: reviewer analytics should only appear when reviewer
metadata exists. We expose :func:`has_reviewer_signal` so the UI can
gate the whole tab on a single boolean without poking at internals.

Notes:

- We key reviewer identity on ``reviewer_name`` when present, falling
  back to ``reviewer_id``. A reviewer with no name and no id is
  skipped (cannot be attributed).
- Reviewer-pair disagreement needs the same ``(record_id, pillar)``
  scored by more than one reviewer. On current datasets each row is
  reviewed by at most one reviewer, so the pair matrix is usually
  empty; the code still handles the multi-reviewer case correctly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

from src.evaluation.agreement import AgreementReport, compute_agreement_report
from src.evaluation.join import ScoredItem
from src.evaluation.metrics import (
    ScorePair,
    off_by_3_plus_rate,
    within_1_rate,
)

__all__ = [
    "ReviewerAnalytics",
    "ReviewerPairStats",
    "ReviewerPillarStats",
    "ReviewerStats",
    "compute_reviewer_analytics",
    "has_reviewer_signal",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewerPillarStats:
    """One reviewer, one pillar.

    ``disagreement_rate`` is the complement of exact-match; the
    within-1 metric is kept separately because "judge got it within a
    point" is the usual SME-friendly bar.
    """

    pillar: str
    support: int
    avg_human_score: float
    avg_judge_score: float
    disagreement_rate: float
    within_1_agreement: float
    large_miss_rate: float


@dataclass(frozen=True)
class ReviewerStats:
    """All metrics for a single reviewer across pillars.

    Wraps an :class:`AgreementReport` for parity with the main
    agreement layer plus reviewer-specific per-pillar stats.
    """

    reviewer: str
    sample_count: int
    report: AgreementReport
    per_pillar: dict[str, ReviewerPillarStats] = field(default_factory=dict)


@dataclass(frozen=True)
class ReviewerPairStats:
    """Disagreement summary between two reviewers on shared items.

    Populated only when both reviewers scored the same
    ``(record_id, pillar)`` (i.e. the dataset has multiple reviewers
    per item). On single-reviewer datasets this will stay empty.
    """

    reviewer_a: str
    reviewer_b: str
    overlap: int
    exact_match_rate: float
    within_1_rate: float
    large_miss_rate: float


@dataclass(frozen=True)
class ReviewerAnalytics:
    """Top-level reviewer analytics container."""

    per_reviewer: dict[str, ReviewerStats] = field(default_factory=dict)
    reviewer_pairs: list[ReviewerPairStats] = field(default_factory=list)

    def reviewers(self) -> list[str]:
        return sorted(self.per_reviewer)

    @property
    def has_data(self) -> bool:
        return bool(self.per_reviewer)


# ---------------------------------------------------------------------------
# Signals / helpers
# ---------------------------------------------------------------------------


def has_reviewer_signal(items: Sequence[ScoredItem]) -> bool:
    """True iff at least one scored item carries reviewer metadata.

    Dashboards use this to decide whether to render the reviewer tab.
    Cheap enough (O(n) in the worst case) that the UI can call it on
    every render without caching.
    """
    return any(_reviewer_key(it) is not None for it in items)


def _reviewer_key(item: ScoredItem) -> str | None:
    """Canonical reviewer identifier. Prefers name over id."""
    return item.reviewer_name or item.reviewer_id


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_reviewer_analytics(
    items: Sequence[ScoredItem],
    *,
    pillars: Sequence[str] | None = None,
) -> ReviewerAnalytics:
    """Build per-reviewer + reviewer-pair analytics.

    Items without a reviewer are ignored (reviewer analytics would be
    meaningless for them); the main :class:`AgreementReport` still
    sees the full dataset.
    """
    by_reviewer: dict[str, list[ScoredItem]] = {}
    for item in items:
        key = _reviewer_key(item)
        if key is None:
            continue
        by_reviewer.setdefault(key, []).append(item)

    per_reviewer: dict[str, ReviewerStats] = {}
    for reviewer, reviewer_items in by_reviewer.items():
        report = compute_agreement_report(
            reviewer_items,
            pillars=pillars,
            include_overall=True,
        )
        per_pillar = {
            pillar: _pillar_stats(reviewer_items, pillar=pillar)
            for pillar in (pillars or report.per_pillar.keys())
        }
        per_reviewer[reviewer] = ReviewerStats(
            reviewer=reviewer,
            sample_count=len(reviewer_items),
            report=report,
            per_pillar=per_pillar,
        )

    reviewer_pairs = _compute_reviewer_pairs(items)

    return ReviewerAnalytics(per_reviewer=per_reviewer, reviewer_pairs=reviewer_pairs)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _pillar_stats(
    reviewer_items: Sequence[ScoredItem],
    *,
    pillar: str,
) -> ReviewerPillarStats:
    """Reviewer-specific per-pillar summary.

    Distinct from :class:`PillarAgreement` in that we also surface the
    reviewer's average human score (useful for "this reviewer rates
    harsher than the team" observations) and skip the heavy kappa /
    confusion-matrix fields that belong on the generic agreement
    path.
    """
    pillar_items = [it for it in reviewer_items if it.pillar == pillar]
    support = len(pillar_items)
    if support == 0:
        return ReviewerPillarStats(
            pillar=pillar,
            support=0,
            avg_human_score=0.0,
            avg_judge_score=0.0,
            disagreement_rate=0.0,
            within_1_agreement=0.0,
            large_miss_rate=0.0,
        )

    pairs: list[ScorePair] = [(it.judge_score, it.human_score) for it in pillar_items]
    avg_human = sum(h for _, h in pairs) / support
    avg_judge = sum(j for j, _ in pairs) / support
    exact = sum(1 for j, h in pairs if j == h) / support

    return ReviewerPillarStats(
        pillar=pillar,
        support=support,
        avg_human_score=avg_human,
        avg_judge_score=avg_judge,
        disagreement_rate=1.0 - exact,
        within_1_agreement=within_1_rate(pairs),
        large_miss_rate=off_by_3_plus_rate(pairs),
    )


def _compute_reviewer_pairs(items: Sequence[ScoredItem]) -> list[ReviewerPairStats]:
    """Compute reviewer-vs-reviewer agreement on shared ``(record_id, pillar)``.

    On datasets where each row is reviewed by a single reviewer the
    returned list is empty - that is the expected behaviour, not an
    error. The Stage 9 UI should check ``reviewer_pairs`` before
    rendering the pair matrix.
    """
    # Group human scores by (record_id, pillar).
    by_key: dict[tuple[str, str], dict[str, int]] = {}
    for item in items:
        key = _reviewer_key(item)
        if key is None:
            continue
        target = by_key.setdefault((item.record_id, item.pillar), {})
        # Last-write-wins if the same reviewer scored the same
        # (record_id, pillar) twice. Duplicate reviewer entries on one
        # item are a data issue; we deliberately don't raise so a noisy
        # dataset doesn't block downstream analytics.
        target[key] = item.human_score

    # Aggregate per reviewer-pair.
    pair_scores: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for reviewer_scores in by_key.values():
        if len(reviewer_scores) < 2:
            continue
        for r_a, r_b in combinations(sorted(reviewer_scores), 2):
            pair_scores.setdefault((r_a, r_b), []).append(
                (reviewer_scores[r_a], reviewer_scores[r_b])
            )

    results: list[ReviewerPairStats] = []
    for (r_a, r_b), score_pairs in sorted(pair_scores.items()):
        overlap = len(score_pairs)
        if overlap == 0:  # pragma: no cover - guarded above
            continue
        exact = sum(1 for a, b in score_pairs if a == b) / overlap
        results.append(
            ReviewerPairStats(
                reviewer_a=r_a,
                reviewer_b=r_b,
                overlap=overlap,
                exact_match_rate=exact,
                within_1_rate=within_1_rate(score_pairs),
                large_miss_rate=off_by_3_plus_rate(score_pairs),
            )
        )
    return results
