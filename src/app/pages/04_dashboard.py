"""Streamlit summary + per-category dashboard.

Three sections:

1. Overall run summary (run_id, totals, latency percentiles).
2. Agreement summary table + pillar bar chart, driven by
   :class:`AgreementReport`.
3. Per-category dashboard: heatmap (``category x pillar``), per-pillar
   category bar chart, and large-miss rate by category. Category is a
   required column so this section is always present.

All chart construction goes through ``src.dashboard.charts``.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.app.state import SS_LAST_RUN_RESULT, SS_LAST_RUN_ROWS
from src.core.constants import PILLARS
from src.dashboard.charts import (
    build_category_pillar_heatmap,
    build_confusion_matrix_heatmap,
    build_large_miss_by_category_chart,
    build_pillar_agreement_bar,
    build_score_distribution_bar,
)
from src.dashboard.tables import (
    build_agreement_summary_rows,
    build_category_breakdown_rows,
)
from src.evaluation import (
    compute_agreement_report,
    compute_sliced_report,
    has_reviewer_signal,
    join_outcomes_with_labels,
    slice_by_category,
)


def render() -> None:
    st.set_page_config(page_title="Dashboard", layout="wide")
    st.title("Evaluation dashboard")
    st.caption("Judge-vs-SME agreement at the run, pillar, and category level.")

    result = st.session_state.get(SS_LAST_RUN_RESULT)
    rows = st.session_state.get(SS_LAST_RUN_ROWS)
    if result is None or not rows:
        st.info("No run result yet. Complete a run on the **Run evaluation** page.")
        return

    joined = join_outcomes_with_labels(rows, result.outcomes)
    if joined.stats.paired == 0:
        st.warning(
            "The run completed but no (row, pillar) pair had both a "
            "successful judge outcome and an SME label. Metrics need "
            "labels to compute agreement."
        )
        _render_run_summary(result)
        return

    _render_run_summary(result)
    _render_join_diagnostics(joined.stats)

    report = compute_agreement_report(joined.items)

    st.subheader("Agreement summary")
    rows_out = build_agreement_summary_rows(report, include_overall=True)
    st.dataframe(rows_out, use_container_width=True)

    metric = st.selectbox(
        "Headline metric",
        options=[
            "severity_alignment",
            "exact_match",
            "within_1",
            "mae",
            "large_miss",
        ],
        index=0,
        help=(
            "severity_alignment weights disagreement distance; "
            "exact_match is strict; mae is lower-is-better."
        ),
    )
    st.altair_chart(
        build_pillar_agreement_bar(report, metric=metric),
        use_container_width=True,
    )

    _render_score_distributions(report)
    _render_category_section(joined.items, metric)
    _render_confusion_section(report)

    if has_reviewer_signal(joined.items):
        st.info(
            "Reviewer metadata detected. Open the **Reviewer analytics** "
            "page for per-reviewer agreement."
        )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _render_run_summary(result: Any) -> None:
    st.subheader("Run")
    summary = result.summary
    cols = st.columns(5)
    cols[0].metric("run_id", result.run_id)
    cols[1].metric("Tasks", summary.total_tasks)
    cols[2].metric("Succeeded", summary.succeeded)
    cols[3].metric("Failed", summary.failed)
    cols[4].metric("Duration (s)", f"{summary.duration_s:.2f}")


def _render_join_diagnostics(stats: Any) -> None:
    with st.expander("Coverage diagnostics", expanded=False):
        st.write(
            {
                "considered": stats.considered,
                "paired": stats.paired,
                "missing_labels": stats.missing_labels,
                "failed_outcomes": stats.failed_outcomes,
                "missing_outcomes": stats.missing_outcomes,
                "orphan_outcomes": stats.orphan_outcomes,
                "pillars_seen": sorted(stats.pillars_seen),
            }
        )


def _render_score_distributions(report: Any) -> None:
    st.subheader("Score distributions")
    st.caption("Judge vs SME score counts per pillar.")
    pillars_with_data = [p for p in report.pillars() if report.per_pillar[p].has_support]
    if not pillars_with_data:
        st.info("No labelled data to plot distributions.")
        return
    # Show four pillars per row; Streamlit handles dynamic columns.
    for i in range(0, len(pillars_with_data), 2):
        cols = st.columns(2)
        for col, pillar in zip(cols, pillars_with_data[i : i + 2], strict=False):
            col.altair_chart(
                build_score_distribution_bar(report.per_pillar[pillar]),
                use_container_width=True,
            )


def _render_category_section(items: Any, metric: str) -> None:
    st.subheader("Per-category dashboard")
    st.caption(
        "Category is a required first-class dimension. The heatmap below "
        "shows the selected headline metric across category x pillar."
    )
    sliced = compute_sliced_report(items, selector=slice_by_category, dimension="category")
    if not sliced.per_slice:
        st.info("No category data to slice.")
        return

    st.altair_chart(
        build_category_pillar_heatmap(sliced, metric=metric),
        use_container_width=True,
    )

    pillar_choice = st.selectbox(
        "Pillar focus for category bar chart",
        options=["(all pillars)", *PILLARS],
        index=0,
    )
    focus_pillar = None if pillar_choice == "(all pillars)" else pillar_choice
    st.altair_chart(
        build_large_miss_by_category_chart(sliced, pillar=focus_pillar),
        use_container_width=True,
    )

    with st.expander("Category breakdown table", expanded=False):
        st.dataframe(
            build_category_breakdown_rows(sliced, pillar=focus_pillar),
            use_container_width=True,
        )


def _render_confusion_section(report: Any) -> None:
    st.subheader("Confusion matrix")
    pillars_with_data = [p for p in report.pillars() if report.per_pillar[p].has_support]
    if not pillars_with_data:
        st.info("No labelled data to render confusion matrices.")
        return
    pillar = st.selectbox("Pillar", options=pillars_with_data, key="jtb.dash.cm.pillar")
    st.altair_chart(
        build_confusion_matrix_heatmap(report.per_pillar[pillar]),
        use_container_width=True,
    )


render()
