"""Filter primitives for the disagreement explorer.

The dashboard renders a wide, sortable table of ``(row, outcome)``
disagreements. Users need to narrow that view by pillar, category,
reviewer, severity bucket, and a minimum confidence threshold. All
filter logic lives here - pages stay pure UI.

We deliberately operate on :class:`ScoredItem` sequences rather than
on the underlying :class:`JudgeOutcome` / :class:`NormalizedRow`
pair, because ``ScoredItem`` already carries every slice key we care
about. The richer disagreement row builders in :mod:`tables` pull
the original outcomes back in for display fields like
``failure_tags`` and ``decision_summary``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from src.evaluation.join import ScoredItem

__all__ = [
    "DisagreementFilter",
    "SeverityBucket",
    "apply_filter",
    "distinct_categories",
    "distinct_pillars",
    "distinct_reviewers",
    "severity_bucket",
]


class SeverityBucket(StrEnum):
    """Distance-based severity labels used across the UI.

    String-valued so the enum can be dropped directly into a
    ``st.selectbox`` without string conversion boilerplate.
    """

    ALL = "all"
    EXACT = "exact"  # distance == 0
    WITHIN_1 = "within_1"  # distance <= 1 (inclusive of exact)
    OFF_BY_2 = "off_by_2"  # distance == 2
    LARGE_MISS = "off_by_3_plus"  # distance >= 3


def severity_bucket(distance: int) -> SeverityBucket:
    """Map an absolute score distance to its severity bucket.

    ``WITHIN_1`` is returned for distance == 1; ``EXACT`` for 0.
    ``WITHIN_1`` as a *filter* option also matches the exact bucket,
    but as a *label* for a single pair, 0 is always "exact".
    """
    if distance == 0:
        return SeverityBucket.EXACT
    if distance == 1:
        return SeverityBucket.WITHIN_1
    if distance == 2:
        return SeverityBucket.OFF_BY_2
    return SeverityBucket.LARGE_MISS


@dataclass(frozen=True)
class DisagreementFilter:
    """Declarative filter state.

    Empty collections mean "no restriction on this dimension".
    ``severity=ALL`` means the same. Pages build one of these from
    the current widget values and pass it to :func:`apply_filter`.
    """

    pillars: frozenset[str] = field(default_factory=frozenset)
    categories: frozenset[str] = field(default_factory=frozenset)
    reviewers: frozenset[str] = field(default_factory=frozenset)
    severity: SeverityBucket = SeverityBucket.ALL
    #: Only show items with absolute score distance >= this value.
    #: ``0`` disables the filter. Handy for a "large misses only" toggle
    #: without using the coarser severity bucket.
    min_distance: int = 0

    def is_empty(self) -> bool:
        """True iff the filter would match everything."""
        return (
            not self.pillars
            and not self.categories
            and not self.reviewers
            and self.severity is SeverityBucket.ALL
            and self.min_distance == 0
        )


def _matches_severity(distance: int, bucket: SeverityBucket) -> bool:
    match bucket:
        case SeverityBucket.ALL:
            return True
        case SeverityBucket.EXACT:
            return distance == 0
        case SeverityBucket.WITHIN_1:
            # Cumulative: includes exact matches.
            return distance <= 1
        case SeverityBucket.OFF_BY_2:
            return distance == 2
        case SeverityBucket.LARGE_MISS:
            return distance >= 3


def apply_filter(
    items: Iterable[ScoredItem],
    filt: DisagreementFilter,
) -> list[ScoredItem]:
    """Apply ``filt`` to ``items`` and return a fresh list.

    The result preserves input order so table sort state in the UI is
    predictable. We don't deduplicate - each input item maps to at
    most one output item.
    """
    out: list[ScoredItem] = []
    for item in items:
        if filt.pillars and item.pillar not in filt.pillars:
            continue
        if filt.categories and item.category not in filt.categories:
            continue
        if filt.reviewers:
            reviewer_key = item.reviewer_name or item.reviewer_id
            if reviewer_key is None or reviewer_key not in filt.reviewers:
                continue
        distance = item.distance
        if distance < filt.min_distance:
            continue
        if not _matches_severity(distance, filt.severity):
            continue
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Distinct value helpers - used to populate UI selectboxes.
# ---------------------------------------------------------------------------


def distinct_pillars(items: Sequence[ScoredItem]) -> list[str]:
    """Sorted list of pillar names present in ``items``."""
    return sorted({it.pillar for it in items})


def distinct_categories(items: Sequence[ScoredItem]) -> list[str]:
    """Sorted list of category values present in ``items``."""
    return sorted({it.category for it in items})


def distinct_reviewers(items: Sequence[ScoredItem]) -> list[str]:
    """Sorted list of reviewer identifiers (name preferred over id)."""
    reviewers: set[str] = set()
    for it in items:
        key = it.reviewer_name or it.reviewer_id
        if key:
            reviewers.add(key)
    return sorted(reviewers)
