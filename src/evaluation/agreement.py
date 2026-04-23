"""Pillar-level agreement aggregation.

This layer sits between :mod:`src.evaluation.metrics` (pure math on
pair lists) and :mod:`src.evaluation.slices` (which groups by
category / reviewer / etc).

Every :class:`PillarAgreement` carries the full Stage 8 metric set
plus the underlying sample counts, so the dashboard can show "12
samples, weighted kappa 0.48, alignment 0.71" without recomputing
anything.

Metrics that require structure (weighted kappa, Spearman) are
``float | None`` - they return ``None`` when the sample is too small
or degenerate. The dashboard should render those as "n/a" rather
than zero, which is a different analytical statement.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from src.core.constants import PILLARS
from src.evaluation.join import ScoredItem
from src.evaluation.metrics import (
    ScorePair,
    confusion_matrix,
    exact_match_rate,
    mean_absolute_error,
    off_by_2_rate,
    off_by_3_plus_rate,
    score_distribution,
    severity_aware_alignment,
    spearman_correlation,
    weighted_kappa,
    within_1_rate,
)

__all__ = [
    "AgreementReport",
    "PillarAgreement",
    "compute_agreement_report",
    "compute_pillar_agreement",
    "pairs_from_items",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PillarAgreement:
    """Per-pillar agreement snapshot.

    The ``support`` field is the number of ``(judge, human)`` pairs
    that actually fed this computation - not the total number of rows
    the user uploaded. Rows with a missing label for this pillar or a
    failed judge outcome are excluded upstream in the join layer.
    """

    pillar: str
    support: int
    exact_match_rate: float
    within_1_rate: float
    off_by_2_rate: float
    off_by_3_plus_rate: float
    mean_absolute_error: float
    severity_aware_alignment: float
    weighted_kappa: float | None
    spearman_correlation: float | None
    judge_score_distribution: dict[int, int]
    human_score_distribution: dict[int, int]
    confusion_matrix: list[list[int]]

    @property
    def has_support(self) -> bool:
        """Whether this pillar has any labelled, successful pairs."""
        return self.support > 0


@dataclass(frozen=True)
class AgreementReport:
    """Collection of :class:`PillarAgreement` views plus an overall.

    ``overall`` blends all pillars into one "run-wide alignment"
    number - useful for executive summaries, but it IS a lossy
    aggregate: pillars with different distributions contribute
    equally per-pair. Treat as a headline number, not the analysis
    target.
    """

    per_pillar: dict[str, PillarAgreement] = field(default_factory=dict)
    overall: PillarAgreement | None = None

    def pillars(self) -> list[str]:
        """Stable, alphabetical pillar ordering for deterministic UIs."""
        return sorted(self.per_pillar)

    def get(self, pillar: str) -> PillarAgreement | None:
        return self.per_pillar.get(pillar)


# ---------------------------------------------------------------------------
# Public computation entrypoints
# ---------------------------------------------------------------------------


def pairs_from_items(items: Iterable[ScoredItem]) -> list[ScorePair]:
    """Extract ``(judge, human)`` pairs from scored items.

    Extracted as a helper so slice/reviewer code can reuse the same
    extraction without reaching into :class:`ScoredItem` directly.
    """
    return [(it.judge_score, it.human_score) for it in items]


def compute_pillar_agreement(
    items: Sequence[ScoredItem],
    *,
    pillar: str,
) -> PillarAgreement:
    """Roll up every Stage 8 metric for one pillar.

    ``items`` should already be filtered to a single pillar - we
    don't filter here because doing so every call wastes work when
    the caller loops across pillars.
    """
    pairs = pairs_from_items(items)
    return PillarAgreement(
        pillar=pillar,
        support=len(pairs),
        exact_match_rate=exact_match_rate(pairs),
        within_1_rate=within_1_rate(pairs),
        off_by_2_rate=off_by_2_rate(pairs),
        off_by_3_plus_rate=off_by_3_plus_rate(pairs),
        mean_absolute_error=mean_absolute_error(pairs),
        severity_aware_alignment=severity_aware_alignment(pairs),
        weighted_kappa=weighted_kappa(pairs),
        spearman_correlation=spearman_correlation(pairs),
        judge_score_distribution=score_distribution(j for j, _ in pairs),
        human_score_distribution=score_distribution(h for _, h in pairs),
        confusion_matrix=confusion_matrix(pairs),
    )


def compute_agreement_report(
    items: Sequence[ScoredItem],
    *,
    pillars: Sequence[str] | None = None,
    include_overall: bool = True,
) -> AgreementReport:
    """Build a per-pillar :class:`AgreementReport`.

    Pillars with zero ``support`` (no labelled + successful items)
    still appear in the report with ``support=0`` and defaulted
    metrics - that avoids "why is relevance missing from the
    dashboard?" confusion when a user forgot to label relevance.

    ``include_overall`` controls whether the cross-pillar aggregate
    is computed. It's included by default since the dashboard wants
    it, but tests often disable it to stay focused on one metric.
    """
    active_pillars = list(pillars) if pillars is not None else list(PILLARS)

    per_pillar: dict[str, PillarAgreement] = {}
    for pillar in active_pillars:
        pillar_items = [it for it in items if it.pillar == pillar]
        per_pillar[pillar] = compute_pillar_agreement(pillar_items, pillar=pillar)

    overall: PillarAgreement | None = None
    if include_overall:
        overall = compute_pillar_agreement(list(items), pillar="__overall__")

    return AgreementReport(per_pillar=per_pillar, overall=overall)
