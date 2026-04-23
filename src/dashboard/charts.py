"""Altair chart builders.

The only place plotting lives. Pages call these and hand the
resulting ``alt.Chart`` to ``st.altair_chart(...)`` - there is no
inline chart construction inside page modules.

Why Altair? It ships with Streamlit (vega-altair is a transitive
dep), it serialises to JSON, and it's strict enough that unit tests
can assert on field encodings without rendering pixels.

Chart contract rules:

- Every chart accepts domain objects (``AgreementReport``,
  ``SliceReport``, ``ReviewerAnalytics``) and does its own pandas
  hand-off. Pages never touch pandas directly.
- Charts return an ``alt.Chart`` / ``alt.LayerChart`` and never call
  ``.show()`` or touch Streamlit.
- Empty-input charts return a small "no data" banner chart instead
  of raising, so the UI degrades gracefully.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd

from src.core.constants import SCORE_MAX, SCORE_MIN
from src.evaluation.agreement import AgreementReport, PillarAgreement
from src.evaluation.reviewer_analysis import ReviewerAnalytics
from src.evaluation.slices import SliceReport

__all__ = [
    "build_category_pillar_heatmap",
    "build_confusion_matrix_heatmap",
    "build_large_miss_by_category_chart",
    "build_no_data_chart",
    "build_pillar_agreement_bar",
    "build_reviewer_agreement_bar",
    "build_score_distribution_bar",
]


# ---------------------------------------------------------------------------
# Defaults + helpers
# ---------------------------------------------------------------------------

# Fixed height keeps charts readable when stacked vertically; Streamlit
# auto-sizes width. Picked by eye on typical dashboard viewports.
_DEFAULT_HEIGHT: int = 280


def build_no_data_chart(message: str = "No data to plot.") -> alt.Chart:
    """Render a neutral placeholder instead of raising on empty input.

    Returning a chart (rather than ``None``) keeps page code simple:
    ``st.altair_chart(build_x(...))`` always works.
    """
    df = pd.DataFrame([{"x": 0, "y": 0, "note": message}])
    return (
        alt.Chart(df)
        .mark_text(fontSize=14, color="#999")
        .encode(text="note:N")
        .properties(height=80)
    )


def _pillar_rows(report: AgreementReport) -> list[dict[str, Any]]:
    """Flatten an agreement report to a dataframe-ready row list."""
    rows: list[dict[str, Any]] = []
    for pillar, agreement in report.per_pillar.items():
        if not agreement.has_support:
            continue
        rows.append(
            {
                "pillar": pillar,
                "support": agreement.support,
                "exact_match": agreement.exact_match_rate,
                "within_1": agreement.within_1_rate,
                "severity_alignment": agreement.severity_aware_alignment,
                "mae": agreement.mean_absolute_error,
                "large_miss": agreement.off_by_3_plus_rate,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Overall agreement bar chart
# ---------------------------------------------------------------------------


def build_pillar_agreement_bar(
    report: AgreementReport,
    *,
    metric: str = "severity_alignment",
    title: str | None = None,
) -> alt.Chart:
    """Bar chart of one metric across pillars.

    Args:
        report: Agreement report from the evaluation module.
        metric: One of ``"severity_alignment"``, ``"exact_match"``,
            ``"within_1"``, ``"mae"``, ``"large_miss"``. Kept as a
            string rather than an enum so the enum doesn't leak into
            chart code - :mod:`dashboard` intentionally has zero
            awareness of the metric-layer type system.
        title: Optional chart title override.
    """
    rows = _pillar_rows(report)
    if not rows:
        return build_no_data_chart("No pillar has any labelled + successful rows yet.")
    df = pd.DataFrame(rows)
    if metric not in df.columns:
        raise ValueError(
            f"build_pillar_agreement_bar: unknown metric {metric!r}; "
            f"expected one of {sorted(c for c in df.columns if c not in {'pillar', 'support'})}."
        )
    # mae has inverse semantics (lower is better) so we draw it on
    # its own scale; the other rate metrics share the [0, 1] scale.
    y_encoding = (
        alt.Y(f"{metric}:Q", title=metric)
        if metric == "mae"
        else alt.Y(f"{metric}:Q", title=metric, scale=alt.Scale(domain=[0.0, 1.0]))
    )
    chart_title = title or f"{metric} by pillar"
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("pillar:N", sort="-y", title="pillar"),
            y=y_encoding,
            tooltip=[
                alt.Tooltip("pillar:N"),
                alt.Tooltip("support:Q"),
                alt.Tooltip("exact_match:Q", format=".2%"),
                alt.Tooltip("within_1:Q", format=".2%"),
                alt.Tooltip("severity_alignment:Q", format=".3f"),
                alt.Tooltip("mae:Q", format=".3f"),
                alt.Tooltip("large_miss:Q", format=".2%"),
            ],
        )
        .properties(height=_DEFAULT_HEIGHT, title=chart_title)
    )


# ---------------------------------------------------------------------------
# Score distribution
# ---------------------------------------------------------------------------


def build_score_distribution_bar(
    agreement: PillarAgreement,
    *,
    title: str | None = None,
) -> alt.Chart:
    """Grouped bar chart: judge vs human score counts for one pillar.

    Returns a no-data banner when the pillar has zero support. We
    keep judge and human in the same chart (grouped) rather than two
    stacked charts - one visual makes under/over-scoring tendencies
    immediately obvious.
    """
    if not agreement.has_support:
        return build_no_data_chart(f"{agreement.pillar}: no labelled data yet.")

    rows: list[dict[str, Any]] = []
    for score in range(SCORE_MIN, SCORE_MAX + 1):
        rows.append(
            {
                "score": score,
                "source": "judge",
                "count": agreement.judge_score_distribution.get(score, 0),
            }
        )
        rows.append(
            {
                "score": score,
                "source": "human",
                "count": agreement.human_score_distribution.get(score, 0),
            }
        )
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("score:O", title="score (1-5)"),
            xOffset="source:N",
            y=alt.Y("count:Q", title="# rows"),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(
                    domain=["judge", "human"],
                    range=["#4c78a8", "#f58518"],
                ),
            ),
            tooltip=["source", "score", "count"],
        )
        .properties(
            height=_DEFAULT_HEIGHT,
            title=title or f"Score distribution - {agreement.pillar}",
        )
    )


# ---------------------------------------------------------------------------
# Category x pillar heatmap
# ---------------------------------------------------------------------------


def build_category_pillar_heatmap(
    sliced: SliceReport,
    *,
    metric: str = "severity_alignment",
    title: str | None = None,
) -> alt.Chart:
    """Heatmap of ``metric`` with slice (usually category) on the y
    axis and pillar on the x axis.

    Cells with zero support render as a neutral grey so empty cells
    are visually distinct from "low alignment" cells.
    """
    rows: list[dict[str, Any]] = []
    for slice_value in sliced.slices():
        for pillar, agreement in sliced.per_slice[slice_value].per_pillar.items():
            value = _pick_metric(agreement, metric)
            rows.append(
                {
                    sliced.dimension: slice_value,
                    "pillar": pillar,
                    "value": value,
                    "support": agreement.support,
                }
            )
    if not rows:
        return build_no_data_chart(f"No {sliced.dimension} slices with data.")

    df = pd.DataFrame(rows)
    # When every row has support==0 there's nothing useful to plot.
    if df["support"].sum() == 0:
        return build_no_data_chart(f"No labelled data per {sliced.dimension}.")

    chart_title = title or f"{metric} by {sliced.dimension} x pillar"
    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("pillar:N", title="pillar"),
            y=alt.Y(f"{sliced.dimension}:N", title=sliced.dimension),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(scheme="blues"),
                legend=alt.Legend(title=metric),
            ),
            tooltip=[
                alt.Tooltip(f"{sliced.dimension}:N"),
                alt.Tooltip("pillar:N"),
                alt.Tooltip("value:Q", format=".3f"),
                alt.Tooltip("support:Q"),
            ],
        )
        .properties(height=_DEFAULT_HEIGHT, title=chart_title)
    )


def _pick_metric(agreement: PillarAgreement, metric: str) -> float:
    """Resolve a metric-name string to a numeric value on
    :class:`PillarAgreement`.

    Raises on unknown names rather than silently returning 0 - a
    typo here would otherwise produce a chart full of zeros with no
    obvious cause.
    """
    match metric:
        case "exact_match":
            return agreement.exact_match_rate
        case "within_1":
            return agreement.within_1_rate
        case "severity_alignment":
            return agreement.severity_aware_alignment
        case "mae":
            return agreement.mean_absolute_error
        case "large_miss":
            return agreement.off_by_3_plus_rate
    raise ValueError(f"Unknown metric {metric!r}.")


# ---------------------------------------------------------------------------
# Large-miss rate by category
# ---------------------------------------------------------------------------


def build_large_miss_by_category_chart(
    sliced: SliceReport,
    *,
    pillar: str | None = None,
    title: str | None = None,
) -> alt.Chart:
    """Bar chart of off-by-3+ rate per category.

    When ``pillar`` is given, the chart is restricted to that pillar;
    otherwise we plot the cross-pillar mean (simple average across
    pillars that have support in the slice).
    """
    rows: list[dict[str, Any]] = []
    for slice_value in sliced.slices():
        report = sliced.per_slice[slice_value]
        if pillar is not None:
            agreement = report.per_pillar.get(pillar)
            if agreement is None or not agreement.has_support:
                continue
            rows.append(
                {
                    sliced.dimension: slice_value,
                    "large_miss_rate": agreement.off_by_3_plus_rate,
                    "support": agreement.support,
                }
            )
        else:
            values = [a.off_by_3_plus_rate for a in report.per_pillar.values() if a.has_support]
            supports = [a.support for a in report.per_pillar.values() if a.has_support]
            if not values:
                continue
            rows.append(
                {
                    sliced.dimension: slice_value,
                    "large_miss_rate": sum(values) / len(values),
                    "support": sum(supports),
                }
            )
    if not rows:
        return build_no_data_chart("No categories with large-miss data.")
    df = pd.DataFrame(rows)
    chart_title = title or (
        f"Large-miss rate by {sliced.dimension}"
        + (f" - {pillar}" if pillar else " (mean across pillars)")
    )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{sliced.dimension}:N", sort="-y"),
            y=alt.Y(
                "large_miss_rate:Q",
                title="off-by-3+ rate",
                scale=alt.Scale(domain=[0.0, 1.0]),
            ),
            tooltip=[
                alt.Tooltip(f"{sliced.dimension}:N"),
                alt.Tooltip("large_miss_rate:Q", format=".2%"),
                alt.Tooltip("support:Q"),
            ],
        )
        .properties(height=_DEFAULT_HEIGHT, title=chart_title)
    )


# ---------------------------------------------------------------------------
# Confusion matrix heatmap
# ---------------------------------------------------------------------------


def build_confusion_matrix_heatmap(
    agreement: PillarAgreement,
    *,
    title: str | None = None,
) -> alt.Chart:
    """Heatmap view of a 5x5 confusion matrix.

    Orientation: human score on the y axis, judge score on the x
    axis (matches the numpy-array convention used by
    :func:`src.evaluation.metrics.confusion_matrix`). Diagonal cells
    are the perfect-match cells.
    """
    if not agreement.has_support:
        return build_no_data_chart(f"{agreement.pillar}: no labelled data yet.")
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(agreement.confusion_matrix):
        for j, count in enumerate(row):
            rows.append(
                {
                    "human": SCORE_MIN + i,
                    "judge": SCORE_MIN + j,
                    "count": count,
                }
            )
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("judge:O", title="judge score"),
        y=alt.Y("human:O", title="human score", sort="descending"),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("count:Q", scale=alt.Scale(scheme="greens")),
        tooltip=["human", "judge", "count"],
    )
    labels = base.mark_text(baseline="middle", fontSize=12).encode(
        text="count:Q",
        color=alt.condition(
            # Dark cells need light text for legibility.
            "datum.count > 3",
            alt.value("white"),
            alt.value("#333"),
        ),
    )
    return (heat + labels).properties(
        height=_DEFAULT_HEIGHT,
        title=title or f"Confusion matrix - {agreement.pillar}",
    )


# ---------------------------------------------------------------------------
# Reviewer agreement bar chart
# ---------------------------------------------------------------------------


def build_reviewer_agreement_bar(
    analytics: ReviewerAnalytics,
    *,
    pillar: str | None = None,
    title: str | None = None,
) -> alt.Chart:
    """Bar chart: exact-match rate per reviewer.

    Restricts to ``pillar`` when provided, otherwise uses the
    reviewer's overall exact-match rate across all pillars.
    """
    if not analytics.has_data:
        return build_no_data_chart("No reviewer metadata present.")
    rows: list[dict[str, Any]] = []
    for reviewer, stats in analytics.per_reviewer.items():
        if pillar is None:
            overall = stats.report.overall
            if overall is None or not overall.has_support:
                continue
            exact = overall.exact_match_rate
            support = overall.support
        else:
            pstats = stats.per_pillar.get(pillar)
            if pstats is None or pstats.support == 0:
                continue
            exact = 1.0 - pstats.disagreement_rate
            support = pstats.support
        rows.append(
            {
                "reviewer": reviewer,
                "exact_match_rate": exact,
                "support": support,
            }
        )
    if not rows:
        return build_no_data_chart("No reviewer data for this pillar.")
    df = pd.DataFrame(rows)
    chart_title = title or (
        "Judge vs reviewer exact-match rate" + (f" - {pillar}" if pillar else " (overall)")
    )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("reviewer:N", sort="-y"),
            y=alt.Y(
                "exact_match_rate:Q",
                scale=alt.Scale(domain=[0.0, 1.0]),
                title="exact match rate",
            ),
            tooltip=[
                alt.Tooltip("reviewer:N"),
                alt.Tooltip("exact_match_rate:Q", format=".2%"),
                alt.Tooltip("support:Q"),
            ],
        )
        .properties(height=_DEFAULT_HEIGHT, title=chart_title)
    )
