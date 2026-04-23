"""Metric engine + agreement scoring.

Public API surface (Stage 8). Import from ``src.evaluation`` rather
than the submodules directly to keep refactors cheap.
"""

from __future__ import annotations

from src.evaluation.agreement import (
    AgreementReport,
    PillarAgreement,
    compute_agreement_report,
    compute_pillar_agreement,
    pairs_from_items,
)
from src.evaluation.join import (
    JoinedDataset,
    JoinStats,
    ScoredItem,
    join_outcomes_with_labels,
)
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
from src.evaluation.reviewer_analysis import (
    ReviewerAnalytics,
    ReviewerPairStats,
    ReviewerPillarStats,
    ReviewerStats,
    compute_reviewer_analytics,
    has_reviewer_signal,
)
from src.evaluation.slices import (
    UNKNOWN_SLICE,
    SliceReport,
    SliceSelector,
    compute_sliced_report,
    slice_by,
    slice_by_category,
    slice_by_intent,
    slice_by_model,
    slice_by_reviewer,
    slice_by_topic,
)

__all__ = [
    "UNKNOWN_SLICE",
    "AgreementReport",
    "JoinStats",
    "JoinedDataset",
    "PillarAgreement",
    "ReviewerAnalytics",
    "ReviewerPairStats",
    "ReviewerPillarStats",
    "ReviewerStats",
    "ScorePair",
    "ScoredItem",
    "SliceReport",
    "SliceSelector",
    "compute_agreement_report",
    "compute_pillar_agreement",
    "compute_reviewer_analytics",
    "compute_sliced_report",
    "confusion_matrix",
    "exact_match_rate",
    "has_reviewer_signal",
    "join_outcomes_with_labels",
    "mean_absolute_error",
    "off_by_2_rate",
    "off_by_3_plus_rate",
    "pairs_from_items",
    "score_distribution",
    "severity_aware_alignment",
    "slice_by",
    "slice_by_category",
    "slice_by_intent",
    "slice_by_model",
    "slice_by_reviewer",
    "slice_by_topic",
    "spearman_correlation",
    "weighted_kappa",
    "within_1_rate",
]
