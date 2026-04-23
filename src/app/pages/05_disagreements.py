"""Streamlit disagreement explorer.

Filter + sort + inspect individual rows where the judge and SME
disagree. Filtering goes through :class:`DisagreementFilter`; the
row builder lives in :mod:`src.dashboard.tables`.
"""

from __future__ import annotations

import streamlit as st

from src.app.state import SS_LAST_RUN_RESULT, SS_LAST_RUN_ROWS
from src.dashboard import (
    DisagreementFilter,
    SeverityBucket,
    apply_filter,
    build_disagreement_rows,
    distinct_categories,
    distinct_pillars,
    distinct_reviewers,
)
from src.evaluation import join_outcomes_with_labels
from src.evaluation.join import ScoredItem


def render() -> None:
    st.set_page_config(page_title="Disagreements", layout="wide")
    st.title("Disagreement explorer")
    st.caption(
        "Every (row, pillar) with both a judge score and an SME label. "
        "Filter by pillar, category, reviewer, and severity bucket."
    )

    result = st.session_state.get(SS_LAST_RUN_RESULT)
    rows = st.session_state.get(SS_LAST_RUN_ROWS)
    if result is None or not rows:
        st.info("No run result yet. Complete a run on **Run evaluation**.")
        return

    joined = join_outcomes_with_labels(rows, result.outcomes)
    if joined.stats.paired == 0:
        st.warning(
            "No labelled judge outcomes to display. This page needs "
            "rows with SME labels to compute per-row disagreements."
        )
        return

    filt = _render_filter_controls(joined.items)
    filtered_items = apply_filter(joined.items, filt)

    _render_counts(total=joined.stats.paired, filtered=len(filtered_items))

    if not filtered_items:
        st.info("No rows match the current filters.")
        return

    table_rows = build_disagreement_rows(rows, result.outcomes, items=filtered_items)
    st.dataframe(
        table_rows,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Filter UI
# ---------------------------------------------------------------------------


def _render_filter_controls(items: list[ScoredItem]) -> DisagreementFilter:
    pillars = distinct_pillars(items)
    categories = distinct_categories(items)
    reviewers = distinct_reviewers(items)

    with st.container(border=True):
        st.caption("Filters")
        col_a, col_b, col_c, col_d = st.columns(4)
        selected_pillars = col_a.multiselect("Pillar", options=pillars, default=[])
        selected_categories = col_b.multiselect("Category", options=categories, default=[])
        selected_reviewers = col_c.multiselect(
            "Reviewer",
            options=reviewers,
            default=[],
            disabled=not reviewers,
            help=("No reviewer metadata in this run" if not reviewers else None),
        )
        severity_choice = col_d.selectbox(
            "Severity",
            options=[b.value for b in SeverityBucket],
            index=0,
            help=(
                "all: no severity filter; within_1 includes exact matches; "
                "off_by_3_plus surfaces only the large misses."
            ),
        )

        min_distance = st.slider(
            "Minimum |judge - human| distance",
            min_value=0,
            max_value=4,
            value=0,
            help="Hard floor on disagreement size; 0 disables.",
        )

    return DisagreementFilter(
        pillars=frozenset(selected_pillars),
        categories=frozenset(selected_categories),
        reviewers=frozenset(selected_reviewers),
        severity=SeverityBucket(severity_choice),
        min_distance=min_distance,
    )


def _render_counts(*, total: int, filtered: int) -> None:
    cols = st.columns(3)
    cols[0].metric("Total labelled pairs", total)
    cols[1].metric("Matching filter", filtered)
    cols[2].metric(
        "Share shown",
        f"{(filtered / total * 100):.1f}%" if total else "-",
    )


render()
