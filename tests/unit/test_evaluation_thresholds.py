"""Tests for :mod:`src.evaluation.thresholds`."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.agreement import AgreementReport, PillarAgreement
from src.evaluation.thresholds import (
    GateStatus,
    evaluate_agreement_against_thresholds,
    load_evaluation_thresholds,
    resolve_pillar_bounds,
    threshold_report_to_mlflow_metrics,
)


def _pa(**kwargs: object) -> PillarAgreement:
    defaults = dict(
        pillar="factual_accuracy",
        support=10,
        exact_match_rate=0.5,
        within_1_rate=0.9,
        off_by_2_rate=0.05,
        off_by_3_plus_rate=0.05,
        mean_absolute_error=0.3,
        severity_aware_alignment=0.85,
        weighted_kappa=0.6,
        spearman_correlation=0.5,
        judge_score_distribution={1: 0, 2: 0, 3: 2, 4: 5, 5: 3},
        human_score_distribution={1: 0, 2: 0, 3: 2, 4: 5, 5: 3},
        confusion_matrix=[[0] * 5 for _ in range(5)],
    )
    defaults.update(kwargs)
    return PillarAgreement(**defaults)  # type: ignore[arg-type]


def test_load_repo_thresholds() -> None:
    repo = Path(__file__).resolve().parents[2]
    cfg = load_evaluation_thresholds(repo / "configs" / "evaluation_thresholds.yaml")
    assert cfg.version
    bounds = resolve_pillar_bounds(cfg, "factual_accuracy")
    assert "within_1_rate" in bounds


def test_gate_pass_high_within_1() -> None:
    repo = Path(__file__).resolve().parents[2]
    cfg = load_evaluation_thresholds(repo / "configs" / "evaluation_thresholds.yaml")
    report = AgreementReport(
        per_pillar={
            "factual_accuracy": _pa(
                within_1_rate=0.95,
                weighted_kappa=0.7,
                off_by_2_rate=0.0,
                off_by_3_plus_rate=0.0,
                mean_absolute_error=0.2,
                severity_aware_alignment=0.9,
            )
        },
        overall=None,
    )
    rep = evaluate_agreement_against_thresholds(report, cfg)
    assert rep.per_pillar["factual_accuracy"].overall == GateStatus.PASS


def test_gate_fail_low_within_1() -> None:
    repo = Path(__file__).resolve().parents[2]
    cfg = load_evaluation_thresholds(repo / "configs" / "evaluation_thresholds.yaml")
    report = AgreementReport(
        per_pillar={
            "factual_accuracy": _pa(
                within_1_rate=0.5,
                weighted_kappa=0.5,
                off_by_2_rate=0.0,
                off_by_3_plus_rate=0.0,
                mean_absolute_error=0.2,
                severity_aware_alignment=0.9,
            )
        },
        overall=None,
    )
    rep = evaluate_agreement_against_thresholds(report, cfg)
    assert rep.per_pillar["factual_accuracy"].overall == GateStatus.FAIL


def test_threshold_mlflow_metrics_numeric() -> None:
    repo = Path(__file__).resolve().parents[2]
    cfg = load_evaluation_thresholds(repo / "configs" / "evaluation_thresholds.yaml")
    report = AgreementReport(
        per_pillar={
            "relevance": _pa(
                pillar="relevance",
                within_1_rate=0.95,
                off_by_2_rate=0.0,
                off_by_3_plus_rate=0.0,
                mean_absolute_error=0.2,
                severity_aware_alignment=0.9,
                weighted_kappa=0.7,
            )
        },
        overall=None,
    )
    rep = evaluate_agreement_against_thresholds(report, cfg)
    m = threshold_report_to_mlflow_metrics(rep)
    assert any(k.startswith("threshold/relevance/") for k in m)
    assert m["threshold/relevance/overall"] >= 0.0
