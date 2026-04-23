"""Row builders for Streamlit ``st.dataframe`` / ``st.table`` views.

Each ``build_*`` helper returns a list of plain dicts with JSON-safe
values. That keeps pages trivial (``st.dataframe(rows)``) and makes
the functions easy to snapshot-test without a Streamlit runtime.

Kept separate from :mod:`charts` because tables and charts evolve
independently - adding a chart rarely requires a new column.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.core.types import NormalizedRow
from src.evaluation.agreement import AgreementReport, PillarAgreement
from src.evaluation.join import ScoredItem
from src.evaluation.reviewer_analysis import ReviewerAnalytics, ReviewerPillarStats
from src.evaluation.slices import SliceReport
from src.judges.base import JudgeOutcome

__all__ = [
    "build_agreement_summary_rows",
    "build_category_breakdown_rows",
    "build_disagreement_rows",
    "build_reviewer_pair_rows",
    "build_reviewer_pillar_rows",
    "build_reviewer_summary_rows",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fmt_pct(value: float) -> float:
    """Normalize a rate to a percentage with reasonable precision.

    We return a float rather than a string so Streamlit sorts numerically.
    Rounding to 4 dp matches what st.dataframe renders by default.
    """
    return round(value * 100, 2)


def _fmt_num(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _row_from_pillar(pillar: PillarAgreement) -> dict[str, Any]:
    return {
        "pillar": pillar.pillar,
        "support": pillar.support,
        "exact_match_%": _fmt_pct(pillar.exact_match_rate),
        "within_1_%": _fmt_pct(pillar.within_1_rate),
        "off_by_2_%": _fmt_pct(pillar.off_by_2_rate),
        "off_by_3_plus_%": _fmt_pct(pillar.off_by_3_plus_rate),
        "mae": _fmt_num(pillar.mean_absolute_error),
        "severity_alignment": _fmt_num(pillar.severity_aware_alignment),
        "weighted_kappa": _fmt_num(pillar.weighted_kappa),
        "spearman": _fmt_num(pillar.spearman_correlation),
    }


# ---------------------------------------------------------------------------
# Agreement summary tables
# ---------------------------------------------------------------------------


def build_agreement_summary_rows(
    report: AgreementReport,
    *,
    include_overall: bool = True,
) -> list[dict[str, Any]]:
    """One row per pillar from an :class:`AgreementReport`.

    ``overall`` (if present) is appended last with pillar =
    ``"__overall__"`` so users can sort it out with a simple filter.
    """
    rows = [_row_from_pillar(report.per_pillar[p]) for p in report.pillars()]
    if include_overall and report.overall is not None:
        rows.append(_row_from_pillar(report.overall))
    return rows


def build_category_breakdown_rows(
    sliced: SliceReport,
    *,
    pillar: str | None = None,
) -> list[dict[str, Any]]:
    """Flatten a :class:`SliceReport` into one row per (slice, pillar).

    When ``pillar`` is set we emit one row per slice restricted to
    that pillar - the layout that feeds "category dashboard for
    relevance" style views. Otherwise we emit the cross-product of
    slices x pillars, useful for a raw download view.
    """
    rows: list[dict[str, Any]] = []
    for slice_value in sliced.slices():
        report = sliced.per_slice[slice_value]
        pillars_to_emit = [pillar] if pillar is not None else report.pillars()
        for p in pillars_to_emit:
            agreement = report.per_pillar.get(p)
            if agreement is None:
                continue
            row = _row_from_pillar(agreement)
            row = {sliced.dimension: slice_value, **row}
            row["slice_total_items"] = sliced.slice_counts.get(slice_value, 0)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Reviewer tables
# ---------------------------------------------------------------------------


def build_reviewer_summary_rows(analytics: ReviewerAnalytics) -> list[dict[str, Any]]:
    """One row per reviewer with the cross-pillar agreement."""
    rows: list[dict[str, Any]] = []
    for reviewer in analytics.reviewers():
        stats = analytics.per_reviewer[reviewer]
        overall = stats.report.overall
        rows.append(
            {
                "reviewer": reviewer,
                "samples": stats.sample_count,
                "exact_match_%": (
                    _fmt_pct(overall.exact_match_rate) if overall is not None else None
                ),
                "within_1_%": (_fmt_pct(overall.within_1_rate) if overall is not None else None),
                "large_miss_%": (
                    _fmt_pct(overall.off_by_3_plus_rate) if overall is not None else None
                ),
                "severity_alignment": (
                    _fmt_num(overall.severity_aware_alignment) if overall is not None else None
                ),
            }
        )
    return rows


def build_reviewer_pillar_rows(analytics: ReviewerAnalytics) -> list[dict[str, Any]]:
    """One row per (reviewer, pillar) with per-pillar reviewer metrics."""
    rows: list[dict[str, Any]] = []
    for reviewer in analytics.reviewers():
        for pillar, stats in analytics.per_reviewer[reviewer].per_pillar.items():
            rows.append(_row_from_reviewer_pillar(reviewer, pillar, stats))
    return rows


def _row_from_reviewer_pillar(
    reviewer: str, pillar: str, stats: ReviewerPillarStats
) -> dict[str, Any]:
    return {
        "reviewer": reviewer,
        "pillar": pillar,
        "support": stats.support,
        "avg_human": _fmt_num(stats.avg_human_score, digits=2),
        "avg_judge": _fmt_num(stats.avg_judge_score, digits=2),
        "disagreement_%": _fmt_pct(stats.disagreement_rate),
        "within_1_%": _fmt_pct(stats.within_1_agreement),
        "large_miss_%": _fmt_pct(stats.large_miss_rate),
    }


def build_reviewer_pair_rows(analytics: ReviewerAnalytics) -> list[dict[str, Any]]:
    """One row per reviewer-pair with overlap-based agreement."""
    return [
        {
            "reviewer_a": pair.reviewer_a,
            "reviewer_b": pair.reviewer_b,
            "overlap": pair.overlap,
            "exact_match_%": _fmt_pct(pair.exact_match_rate),
            "within_1_%": _fmt_pct(pair.within_1_rate),
            "large_miss_%": _fmt_pct(pair.large_miss_rate),
        }
        for pair in analytics.reviewer_pairs
    ]


# ---------------------------------------------------------------------------
# Disagreement table - the heaviest view in the app
# ---------------------------------------------------------------------------


def build_disagreement_rows(
    rows: Sequence[NormalizedRow],
    outcomes: Sequence[JudgeOutcome],
    *,
    items: Sequence[ScoredItem] | None = None,
) -> list[dict[str, Any]]:
    """Build the disagreement explorer's row set.

    The explorer wants a dense, human-readable row per successful
    ``(row, pillar, outcome)`` with both scores side by side, the
    distance bucket, and the fields users actually read when
    triaging a miss (``decision_summary``, ``failure_tags``,
    ``confidence``, the raw inputs).

    Args:
        rows: Normalized rows used for the run.
        outcomes: Outcomes produced by the runner.
        items: Optional pre-filtered :class:`ScoredItem` sequence -
            if provided, only rows present here are emitted. Lets
            pages apply a :class:`DisagreementFilter` first and pass
            the filtered view in, avoiding a double-loop.
    """
    row_by_id = {r.record_id: r for r in rows}
    allowed_keys: set[tuple[str, str]] | None = None
    if items is not None:
        allowed_keys = {(it.record_id, it.pillar) for it in items}

    out: list[dict[str, Any]] = []
    for outcome in outcomes:
        if outcome.result is None:
            continue
        row = row_by_id.get(outcome.record_id)
        if row is None:
            continue
        if allowed_keys is not None and (outcome.record_id, outcome.pillar) not in allowed_keys:
            continue
        human_label = getattr(row, f"label_{outcome.pillar}", None)
        judge_score = outcome.result.score
        distance = abs(judge_score - human_label) if human_label is not None else None
        out.append(
            {
                "record_id": row.record_id,
                "pillar": outcome.pillar,
                "category": row.category,
                "reviewer": row.reviewer_name or row.reviewer_id,
                "intent": row.intent,
                "topic": row.topic,
                "judge_score": judge_score,
                "human_score": human_label,
                "distance": distance,
                "confidence": round(outcome.result.confidence, 3),
                "decision_summary": outcome.result.decision_summary,
                "failure_tags": ",".join(outcome.result.failure_tags),
                "why_not_higher": outcome.result.why_not_higher or "",
                "why_not_lower": outcome.result.why_not_lower or "",
                "user_input": row.user_input,
                "agent_output": row.agent_output,
                "latency_ms": round(outcome.latency_ms, 2),
                "model": outcome.model_name or "",
            }
        )
    return out
