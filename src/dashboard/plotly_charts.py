"""Interactive Plotly charts for risk / evidence analytics.

All figure construction lives here per Streamlit UI rules; pages only
call ``st.plotly_chart`` or export HTML for MLflow.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.evaluation.diagnostics import OLSResult, PillarDiagnostics, RunDiagnostics
from src.evaluation.join import ScoredItem

__all__ = [
    "build_drift_pmf_figure",
    "build_human_vs_judge_scatter",
    "build_judge_human_pmf_figure",
    "build_residual_histogram",
    "combined_risk_evidence_html",
]


def build_judge_human_pmf_figure(diag: PillarDiagnostics) -> go.Figure:
    scores = list(range(1, 6))
    fig = go.Figure()
    fig.add_bar(
        name="Judge",
        x=scores,
        y=[diag.judge_pmf[s] for s in scores],
        marker_color="#636EFA",
    )
    fig.add_bar(
        name="Human (SME)",
        x=scores,
        y=[diag.human_pmf[s] for s in scores],
        marker_color="#EF553B",
    )
    fig.update_layout(
        barmode="group",
        title=f"Score distribution - {diag.pillar}",
        xaxis_title="Ordinal score (1-5)",
        yaxis_title="Probability",
        legend_title="Source",
        height=360,
    )
    return fig


def build_human_vs_judge_scatter(
    items: Sequence[ScoredItem],
    pillar: str,
    *,
    ols: OLSResult | None = None,
    seed: int = 42,
) -> go.Figure:
    pts = [it for it in items if it.pillar == pillar]
    rnd = random.Random(seed)
    judge = [float(it.judge_score) + rnd.uniform(-0.09, 0.09) for it in pts]
    human = [float(it.human_score) + rnd.uniform(-0.09, 0.09) for it in pts]
    hover = [f"{it.record_id}<br>cat: {it.category}" for it in pts]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=judge,
            y=human,
            mode="markers",
            marker=dict(size=8, opacity=0.65),
            text=hover,
            hoverinfo="text+x+y",
            name="Rows",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[1, 5],
            y=[1, 5],
            mode="lines",
            name="Identity (perfect)",
            line=dict(dash="dash", color="gray"),
        )
    )
    if ols is not None:
        xs = [1.0, 5.0]
        ys = [ols.intercept + ols.slope * x for x in xs]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"OLS human~judge (R²={ols.r_squared:.2f})",
                line=dict(color="green"),
            )
        )
    fig.update_layout(
        title=f"Human vs judge - {pillar}",
        xaxis_title="Judge score (jittered)",
        yaxis_title="Human score (jittered)",
        height=400,
        xaxis=dict(range=[0.5, 5.5]),
        yaxis=dict(range=[0.5, 5.5]),
    )
    return fig


def build_residual_histogram(items: Sequence[ScoredItem], pillar: str) -> go.Figure:
    pts = [it for it in items if it.pillar == pillar]
    res = [it.judge_score - it.human_score for it in pts]
    fig = go.Figure(
        data=[
            go.Histogram(
                x=res,
                nbinsx=11,
                marker_color="#AB63FA",
                name="judge - human",
            )
        ]
    )
    fig.update_layout(
        title=f"Residual distribution - {pillar}",
        xaxis_title="Judge score - human score",
        yaxis_title="Count",
        height=320,
        bargap=0.15,
    )
    return fig


def build_drift_pmf_figure(
    pillar: str,
    current: dict[int, float],
    baseline: dict[int, float],
) -> go.Figure:
    scores = list(range(1, 6))
    fig = go.Figure()
    fig.add_bar(
        name="Current judge",
        x=scores,
        y=[current[s] for s in scores],
        marker_color="#636EFA",
    )
    fig.add_bar(
        name="Baseline judge",
        x=scores,
        y=[baseline[s] for s in scores],
        marker_color="#FECB52",
    )
    fig.update_layout(
        barmode="group",
        title=f"Judge PMF drift - {pillar}",
        xaxis_title="Score",
        yaxis_title="Probability",
        height=360,
    )
    return fig


def combined_risk_evidence_html(
    items: Sequence[ScoredItem],
    diag: RunDiagnostics,
    *,
    baseline_pmfs: dict[str, dict[int, float]] | None = None,
) -> str:
    """Single HTML document with one figure per pillar (PMF + scatter + residuals)."""
    pillars = list(diag.pillars.keys())
    if not pillars:
        return "<html><body><p>No diagnostics</p></body></html>"

    parts: list[str] = []
    include_js = True
    for p in pillars:
        d = diag.pillars[p]
        fig_pmf = build_judge_human_pmf_figure(d)
        parts.append(fig_pmf.to_html(full_html=False, include_plotlyjs=include_js))
        include_js = False
        fig_sc = build_human_vs_judge_scatter(items, p, ols=d.ols_human_on_judge)
        parts.append(fig_sc.to_html(full_html=False, include_plotlyjs=False))
        fig_r = build_residual_histogram(items, p)
        parts.append(fig_r.to_html(full_html=False, include_plotlyjs=False))
        if baseline_pmfs and p in baseline_pmfs and diag.baseline_compatible:
            fig_d = build_drift_pmf_figure(p, d.judge_pmf, baseline_pmfs[p])
            parts.append(fig_d.to_html(full_html=False, include_plotlyjs=False))

    body = "".join(f'<div style="margin-bottom:2rem">{h}</div>' for h in parts)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Risk evidence</title></head>
<body><h1>Judge testbench - risk evidence</h1>{body}</body></html>"""


def multi_pillar_summary_figure(diag: RunDiagnostics) -> go.Figure:
    """Grid of mean residual and OLS R² by pillar (compact executive view)."""
    pillars = sorted(diag.pillars.keys())
    if not pillars:
        return go.Figure()
    residuals = [diag.pillars[p].mean_residual_judge_minus_human for p in pillars]
    r2s: list[float] = []
    for p in pillars:
        ols = diag.pillars[p].ols_human_on_judge
        r2s.append(ols.r_squared if ols is not None else 0.0)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Mean residual (judge - human)", "OLS R2 (human ~ judge)"),
    )
    fig.add_trace(
        go.Bar(x=pillars, y=residuals, name="mean residual", marker_color="#00CC96"), row=1, col=1
    )
    fig.add_trace(go.Bar(x=pillars, y=r2s, name="R²", marker_color="#FF6692"), row=1, col=2)
    fig.update_layout(height=400, showlegend=False, title_text="Pillar summary")
    return fig
