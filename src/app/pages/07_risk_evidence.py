"""Risk / evidence analytics: thresholds, drift, and Plotly deep dives."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.app.state import (
    SS_BASELINE_SNAPSHOT,
    SS_LAST_RUN_RESULT,
    SS_LAST_RUN_ROWS,
    SS_RUN_CONFIG,
    RunConfig,
)
from src.core.constants import PILLARS
from src.core.exceptions import ConfigLoadError
from src.core.settings import get_settings
from src.dashboard.plotly_charts import (
    build_drift_pmf_figure,
    build_human_vs_judge_scatter,
    build_judge_human_pmf_figure,
    build_residual_histogram,
    multi_pillar_summary_figure,
)
from src.evaluation.agreement import compute_agreement_report
from src.evaluation.diagnostics import (
    BaselineSnapshot,
    build_baseline_snapshot,
    compute_run_diagnostics,
)
from src.evaluation.join import join_outcomes_with_labels
from src.evaluation.thresholds import (
    GateStatus,
    PillarGateReport,
    evaluate_agreement_against_thresholds,
    load_evaluation_thresholds,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
THRESHOLDS_YAML = REPO_ROOT / "configs" / "evaluation_thresholds.yaml"


def _suggested_actions(pillar_report: PillarGateReport) -> str:
    hints: list[str] = []
    for g in pillar_report.gates:
        if g.status != GateStatus.FAIL and g.status != GateStatus.WARN:
            continue
        if g.metric == "within_1_rate":
            hints.append("Tighten rubric anchors or prompt for finer ordinal discrimination.")
        elif g.metric == "weighted_kappa":
            hints.append("Review systematic off-by-one patterns; consider SME calibration session.")
        elif g.metric == "mean_absolute_error":
            hints.append("Reduce judge temperature or add few-shot exemplars at score boundaries.")
        elif g.metric == "severity_aware_alignment":
            hints.append("Focus on large-miss cases in Disagreements; adjust failure-tag guidance.")
        elif g.metric == "large_miss_rate":
            hints.append(
                "High severe errors: check grounding/context requirements for this pillar."
            )
    return "; ".join(hints) if hints else "No gate breaches - monitor drift on the next run."


def render() -> None:
    st.set_page_config(page_title="Risk evidence", layout="wide")
    st.title("Risk & evidence analytics")
    st.caption(
        "North-star gates, judge-SME distributions, regression fit, and drift vs a pinned baseline. "
        "Pair with MLflow to compare runs."
    )

    result = st.session_state.get(SS_LAST_RUN_RESULT)
    rows = st.session_state.get(SS_LAST_RUN_ROWS)
    run_config: RunConfig | None = st.session_state.get(SS_RUN_CONFIG)

    if result is None or not rows:
        st.info("Run an evaluation first (**Run evaluation** page).")
        return

    pillars_seq = list(run_config.pillars) if run_config is not None else list(PILLARS)
    joined = join_outcomes_with_labels(rows, result.outcomes, pillars=pillars_seq)
    if not joined.items:
        st.warning(
            "No labelled scored pairs in the last run - agreement metrics and gates need SME labels."
        )
        return

    report = compute_agreement_report(joined.items, pillars=pillars_seq, include_overall=True)

    try:
        cfg_thr = load_evaluation_thresholds(THRESHOLDS_YAML)
    except ConfigLoadError as exc:
        st.error(f"Could not load thresholds: {exc}")
        cfg_thr = None

    raw_base = st.session_state.get(SS_BASELINE_SNAPSHOT)
    baseline: BaselineSnapshot | None = None
    if isinstance(raw_base, dict):
        try:
            baseline = BaselineSnapshot.from_serializable(raw_base)
        except (KeyError, TypeError, ValueError):
            baseline = None

    fingerprint = None
    if rows:
        from src.observability.run_metadata import dataset_fingerprint

        fingerprint = dataset_fingerprint(rows)

    diag = compute_run_diagnostics(
        joined.items,
        pillars=pillars_seq,
        dataset_fingerprint=fingerprint,
        baseline=baseline,
    )

    st.subheader("Baseline for drift")
    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "Pin this run as baseline",
            help="Stores judge PMFs; drift compares when dataset fingerprint matches.",
        ):
            snap = build_baseline_snapshot(
                joined.items,
                dataset_fingerprint=fingerprint or "",
                pillars=tuple(pillars_seq),
                run_id=result.run_id,
            )
            st.session_state[SS_BASELINE_SNAPSHOT] = snap.to_serializable()
            st.success(f"Baseline pinned (run {result.run_id}).")
    with c2:
        if st.button("Clear baseline"):
            st.session_state.pop(SS_BASELINE_SNAPSHOT, None)
            st.info("Baseline cleared.")

    if baseline:
        st.caption(
            f"Baseline run: `{baseline.run_id or 'unknown'}` | "
            f"Fingerprint match for drift: **{diag.baseline_compatible}**"
        )
    else:
        st.caption(
            "No baseline pinned - drift charts appear after you pin one on the same dataset fingerprint."
        )

    if cfg_thr is not None:
        thr_rep = evaluate_agreement_against_thresholds(report, cfg_thr)
        st.subheader("North-star gates")
        st.metric("Worst gate status", thr_rep.worst_status().value)
        gate_rows = []
        for pillar_gate in thr_rep.per_pillar.values():
            gate_rows.append(
                {
                    "pillar": pillar_gate.pillar,
                    "overall": pillar_gate.overall.value,
                    "suggested_actions": _suggested_actions(pillar_gate),
                }
            )
        st.dataframe(gate_rows, use_container_width=True, hide_index=True)

    st.subheader("Executive summary (Plotly)")
    st.plotly_chart(multi_pillar_summary_figure(diag), use_container_width=True)

    st.subheader("Per-pillar visuals")
    for p in pillars_seq:
        d = diag.pillars.get(p)
        if d is None or d.support == 0:
            continue
        st.markdown(f"#### {p}")
        st.plotly_chart(build_judge_human_pmf_figure(d), use_container_width=True)
        st.plotly_chart(
            build_human_vs_judge_scatter(joined.items, p, ols=d.ols_human_on_judge),
            use_container_width=True,
        )
        st.plotly_chart(build_residual_histogram(joined.items, p), use_container_width=True)
        if baseline and diag.baseline_compatible and p in baseline.judge_pmfs:
            st.plotly_chart(
                build_drift_pmf_figure(p, d.judge_pmf, baseline.judge_pmfs[p]),
                use_container_width=True,
            )

    st.subheader("MLflow")
    settings = get_settings()
    if settings.mlflow_tracking_uri:
        st.caption(
            "Tracking URI configured - this run logs diagnostics, gates, and Plotly HTML when enabled on **Run evaluation**."
        )
    else:
        st.caption("Set `JTB_MLFLOW_TRACKING_URI` to compare runs in MLflow.")


render()
