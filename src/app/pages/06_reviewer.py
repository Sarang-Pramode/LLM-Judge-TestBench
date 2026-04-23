"""Streamlit reviewer analytics page.

Conditional: only activates when the run's joined dataset contains
reviewer metadata. When no reviewer signal is present the page
explains why and points the user at the Upload page.

Three panels:

1. Per-reviewer summary (overall exact-match %, large-miss %, etc).
2. Per-reviewer x pillar detail (average scores, disagreement rates).
3. Reviewer-pair overlap table when two or more reviewers scored the
   same ``(record_id, pillar)`` pair.
"""

from __future__ import annotations

import streamlit as st

from src.app.state import SS_LAST_RUN_RESULT, SS_LAST_RUN_ROWS
from src.core.constants import PILLARS
from src.dashboard.charts import build_reviewer_agreement_bar
from src.dashboard.tables import (
    build_reviewer_pair_rows,
    build_reviewer_pillar_rows,
    build_reviewer_summary_rows,
)
from src.evaluation import (
    compute_reviewer_analytics,
    has_reviewer_signal,
    join_outcomes_with_labels,
)


def render() -> None:
    st.set_page_config(page_title="Reviewer analytics", layout="wide")
    st.title("Reviewer analytics")
    st.caption(
        "Per-reviewer agreement against the judge, plus reviewer-pair "
        "overlap where the same (row, pillar) was scored more than once."
    )

    result = st.session_state.get(SS_LAST_RUN_RESULT)
    rows = st.session_state.get(SS_LAST_RUN_ROWS)
    if result is None or not rows:
        st.info("No run result yet. Complete a run on **Run evaluation**.")
        return

    joined = join_outcomes_with_labels(rows, result.outcomes)
    if not joined.items:
        st.warning(
            "No scored rows with SME labels were produced for this run, "
            "so reviewer analytics cannot be computed. Check that labels "
            "are mapped on **Upload** and that judge tasks succeeded on "
            "**Run evaluation**."
        )
        return
    if not has_reviewer_signal(joined.items):
        st.warning(
            "No reviewer metadata in this run. On **Upload**, map "
            "``reviewer_name`` and/or ``reviewer_id`` (under **Reviewer**). "
            "If you changed datasets, re-open **Upload** so column mapping "
            "refreshes for the new file."
        )
        return

    analytics = compute_reviewer_analytics(joined.items)
    if not analytics.has_data:
        st.info("Reviewer metadata was present but every reviewer is empty.")
        return

    st.subheader("Per-reviewer summary")
    st.dataframe(
        build_reviewer_summary_rows(analytics),
        use_container_width=True,
        hide_index=True,
    )

    pillar_choice = st.selectbox(
        "Focus pillar",
        options=["(overall)", *PILLARS],
        index=0,
    )
    focus_pillar = None if pillar_choice == "(overall)" else pillar_choice
    st.altair_chart(
        build_reviewer_agreement_bar(analytics, pillar=focus_pillar),
        use_container_width=True,
    )

    st.subheader("Per-reviewer x pillar")
    st.dataframe(
        build_reviewer_pillar_rows(analytics),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Reviewer pair overlap")
    pair_rows = build_reviewer_pair_rows(analytics)
    if not pair_rows:
        st.caption(
            "No overlap: each (row, pillar) was scored by at most one "
            "reviewer. Add duplicate reviews to see inter-reviewer "
            "agreement here."
        )
    else:
        st.dataframe(pair_rows, use_container_width=True, hide_index=True)


render()
