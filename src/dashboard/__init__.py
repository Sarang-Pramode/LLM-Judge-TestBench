"""Reusable chart / table / filter helpers for the Streamlit pages.

All plotting lives under :mod:`src.dashboard`. Pages import from here
and never construct charts inline, per the ``streamlit-ui`` rule.
"""

from __future__ import annotations

from src.dashboard.charts import (
    build_category_pillar_heatmap,
    build_confusion_matrix_heatmap,
    build_large_miss_by_category_chart,
    build_no_data_chart,
    build_pillar_agreement_bar,
    build_reviewer_agreement_bar,
    build_score_distribution_bar,
)
from src.dashboard.filters import (
    DisagreementFilter,
    SeverityBucket,
    apply_filter,
    distinct_categories,
    distinct_pillars,
    distinct_reviewers,
    severity_bucket,
)
from src.dashboard.tables import (
    build_agreement_summary_rows,
    build_category_breakdown_rows,
    build_disagreement_rows,
    build_reviewer_pair_rows,
    build_reviewer_pillar_rows,
    build_reviewer_summary_rows,
)

__all__ = [
    "DisagreementFilter",
    "SeverityBucket",
    "apply_filter",
    "build_agreement_summary_rows",
    "build_category_breakdown_rows",
    "build_category_pillar_heatmap",
    "build_confusion_matrix_heatmap",
    "build_disagreement_rows",
    "build_large_miss_by_category_chart",
    "build_no_data_chart",
    "build_pillar_agreement_bar",
    "build_reviewer_agreement_bar",
    "build_reviewer_pair_rows",
    "build_reviewer_pillar_rows",
    "build_reviewer_summary_rows",
    "build_score_distribution_bar",
    "distinct_categories",
    "distinct_pillars",
    "distinct_reviewers",
    "severity_bucket",
]
