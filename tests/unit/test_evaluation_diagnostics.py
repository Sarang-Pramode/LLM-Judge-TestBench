"""Tests for :mod:`src.evaluation.diagnostics`."""

from __future__ import annotations

from src.evaluation.diagnostics import (
    BaselineSnapshot,
    build_baseline_snapshot,
    compute_pillar_diagnostics,
    compute_run_diagnostics,
    histogram_pmf,
    jensen_shannon_divergence,
    psi_ordinal,
)
from src.evaluation.join import ScoredItem


def test_histogram_pmf_uniform_when_empty() -> None:
    p = histogram_pmf([])
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert len(p) == 5


def test_js_identical_near_zero() -> None:
    p = histogram_pmf([1, 2, 3, 4, 5])
    assert jensen_shannon_divergence(p, p) < 1e-6


def test_psi_identical_zero() -> None:
    p = histogram_pmf([1, 1, 2, 2, 3])
    assert abs(psi_ordinal(p, p)) < 1e-6


def test_ols_human_on_judge_perfect_diagonal() -> None:
    items = [
        ScoredItem(
            record_id=f"r{i}",
            pillar="relevance",
            judge_score=s,
            human_score=s,
            category="c",
        )
        for i, s in enumerate([1, 2, 3, 4, 5], start=1)
    ]
    d = compute_pillar_diagnostics(items, pillar="relevance")
    assert d.ols_human_on_judge is not None
    assert abs(d.ols_human_on_judge.slope - 1.0) < 0.01
    assert abs(d.ols_human_on_judge.intercept) < 0.05
    assert d.ols_human_on_judge.r_squared > 0.99


def test_baseline_roundtrip_serializable() -> None:
    items = [
        ScoredItem("a", "p1", 3, 4, "x"),
        ScoredItem("b", "p1", 4, 4, "x"),
    ]
    snap = build_baseline_snapshot(
        items,
        dataset_fingerprint="fp1",
        pillars=("p1",),
        run_id="run-a",
    )
    back = BaselineSnapshot.from_serializable(snap.to_serializable())
    assert back.dataset_fingerprint == "fp1"
    assert back.judge_pmfs["p1"] == snap.judge_pmfs["p1"]


def test_drift_metrics_when_baseline_compatible() -> None:
    items = [
        ScoredItem("a", "p1", 3, 4, "x"),
        ScoredItem("b", "p1", 3, 3, "x"),
    ]
    snap = build_baseline_snapshot(
        items,
        dataset_fingerprint="fp-drift",
        pillars=("p1",),
    )
    items2 = [
        ScoredItem("a", "p1", 4, 4, "x"),
        ScoredItem("b", "p1", 4, 3, "x"),
    ]
    run = compute_run_diagnostics(
        items2,
        pillars=("p1",),
        dataset_fingerprint="fp-drift",
        baseline=snap,
    )
    assert run.baseline_compatible is True
    d = run.pillars["p1"]
    assert d.js_vs_baseline is not None
    assert d.psi_vs_baseline is not None
