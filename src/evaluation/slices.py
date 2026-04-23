"""Group metrics by arbitrary :class:`ScoredItem` dimensions.

Stage 8's acceptance criteria call out three slicing dimensions:

- ``category`` (required, mandated by the dataset contract).
- ``reviewer_name`` / ``reviewer_id`` (opt-in; handled here for the
  straightforward "reviewer as slice" case, with the richer reviewer
  analytics living in :mod:`reviewer_analysis`).
- Anything else useful: ``intent``, ``topic``, ``model_name``,
  ``conversation_id``.

We model slicing as a callable ``(ScoredItem) -> str | None`` so new
dimensions can be added without touching this module. Items whose
slice key is ``None`` (for example: reviewer_name on unreviewed rows)
are routed to an explicit ``__unknown__`` bucket so the UI can still
show "unassigned" without merging them into a real slice.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from src.evaluation.agreement import AgreementReport, compute_agreement_report
from src.evaluation.join import ScoredItem

__all__ = [
    "UNKNOWN_SLICE",
    "SliceReport",
    "SliceSelector",
    "compute_sliced_report",
    "slice_by",
    "slice_by_category",
    "slice_by_intent",
    "slice_by_model",
    "slice_by_reviewer",
    "slice_by_topic",
]


#: Bucket used when a slice key is ``None``. Surfaced directly in UIs
#: so users see "unassigned" rather than silently losing rows.
UNKNOWN_SLICE: str = "__unknown__"


#: A slice selector extracts a string key from a scored item. Returning
#: ``None`` puts the item into :data:`UNKNOWN_SLICE`. Returning the
#: empty string is treated the same way - users nearly always mean
#: "no value" when they write ``""``.
SliceSelector = Callable[[ScoredItem], str | None]


# ---------------------------------------------------------------------------
# Common selectors
# ---------------------------------------------------------------------------


def slice_by_category(item: ScoredItem) -> str:
    """Category is required by the dataset contract, so this never None."""
    return item.category


def slice_by_reviewer(item: ScoredItem) -> str | None:
    """Prefer ``reviewer_name`` (human-friendly) over ``reviewer_id``."""
    return item.reviewer_name or item.reviewer_id


def slice_by_intent(item: ScoredItem) -> str | None:
    return item.intent


def slice_by_topic(item: ScoredItem) -> str | None:
    return item.topic


def slice_by_model(item: ScoredItem) -> str | None:
    return item.model_name


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceReport:
    """Sliced agreement: ``{slice_value -> AgreementReport}`` plus
    bookkeeping.

    Attributes:
        dimension: Human-readable dimension name (e.g. ``"category"``).
            Carried through for observability / dashboard labels.
        per_slice: One :class:`AgreementReport` per slice value. Uses
            :data:`UNKNOWN_SLICE` for items with a ``None`` key.
        slice_counts: Total scored items in each slice (not per-pillar
            ``support``; count includes items across all pillars).
    """

    dimension: str
    per_slice: dict[str, AgreementReport] = field(default_factory=dict)
    slice_counts: dict[str, int] = field(default_factory=dict)

    def slices(self) -> list[str]:
        """Deterministic (alphabetical) slice ordering."""
        return sorted(self.per_slice)


# ---------------------------------------------------------------------------
# Low-level slicing
# ---------------------------------------------------------------------------


def slice_by(
    items: Sequence[ScoredItem],
    selector: SliceSelector,
) -> dict[str, list[ScoredItem]]:
    """Group ``items`` by the selector, using :data:`UNKNOWN_SLICE` for
    ``None`` / empty keys.

    Item order within each bucket matches the input order, which keeps
    downstream metrics deterministic for a given input sequence.
    """
    buckets: dict[str, list[ScoredItem]] = {}
    for item in items:
        key = selector(item)
        if not key:
            key = UNKNOWN_SLICE
        buckets.setdefault(key, []).append(item)
    return buckets


# ---------------------------------------------------------------------------
# High-level: slice + compute per-pillar metrics in one call
# ---------------------------------------------------------------------------


def compute_sliced_report(
    items: Sequence[ScoredItem],
    *,
    selector: SliceSelector,
    dimension: str,
    pillars: Sequence[str] | None = None,
    include_overall_per_slice: bool = True,
) -> SliceReport:
    """Slice ``items`` by ``selector`` and compute per-pillar metrics
    for each slice.

    Args:
        items: All scored items (usually ``JoinedDataset.items``).
        selector: Extracts the slice key (see :type:`SliceSelector`).
        dimension: Human label used in the resulting
            :class:`SliceReport`. Purely cosmetic - no logic depends
            on its value.
        pillars: Restrict metrics to these pillars. Forwarded to
            :func:`compute_agreement_report`.
        include_overall_per_slice: Whether each slice's
            :class:`AgreementReport` should compute the cross-pillar
            overall. Disabled by default in slice views where the
            dashboard only cares about per-pillar numbers.

    The returned :class:`SliceReport` can be directly consumed by the
    dashboard: ``report.per_slice[slice_key].per_pillar[pillar]``
    gives you one point on a "category x pillar" heatmap.
    """
    buckets = slice_by(items, selector)
    per_slice: dict[str, AgreementReport] = {}
    slice_counts: dict[str, int] = {}
    for slice_value, slice_items in buckets.items():
        per_slice[slice_value] = compute_agreement_report(
            slice_items,
            pillars=pillars,
            include_overall=include_overall_per_slice,
        )
        slice_counts[slice_value] = len(slice_items)
    return SliceReport(
        dimension=dimension,
        per_slice=per_slice,
        slice_counts=slice_counts,
    )
