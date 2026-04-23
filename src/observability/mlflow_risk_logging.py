"""MLflow risk / diagnostics logging (module-level entry points).

Lives in a dedicated module so :mod:`src.app.pages.03_run_eval` can import
stable names without depending on ordering inside :mod:`mlflow_logger`
or on partially-synced copies of that file.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.evaluation.diagnostics import RunDiagnostics
from src.evaluation.thresholds import (
    RunThresholdReport,
    threshold_report_to_mlflow_metrics,
    threshold_report_to_serializable,
)
from src.observability.mlflow_logger import MLflowLogger, _guarded, _sanitize

__all__ = [
    "log_diagnostics_mlflow",
    "log_plotly_html_mlflow",
    "log_threshold_report_mlflow",
]


def _diagnostics_to_mlflow_metrics(diag: RunDiagnostics) -> dict[str, float]:
    """Scalar metrics for MLflow ``log_metrics``."""
    metrics: dict[str, float] = {}
    for p, d in diag.pillars.items():
        safe = _sanitize(p)
        metrics[f"diag/{safe}/mean_residual_judge_minus_human"] = float(
            d.mean_residual_judge_minus_human
        )
        metrics[f"diag/{safe}/pct_positive_residual"] = float(d.pct_positive_residual)
        if d.ols_human_on_judge is not None:
            metrics[f"diag/{safe}/ols_slope"] = float(d.ols_human_on_judge.slope)
            metrics[f"diag/{safe}/ols_intercept"] = float(d.ols_human_on_judge.intercept)
            metrics[f"diag/{safe}/ols_r2"] = float(d.ols_human_on_judge.r_squared)
        if d.js_vs_baseline is not None:
            metrics[f"diag/{safe}/js_vs_baseline"] = float(d.js_vs_baseline)
        if d.psi_vs_baseline is not None:
            metrics[f"diag/{safe}/psi_vs_baseline"] = float(d.psi_vs_baseline)
    metrics["diag/baseline_compatible"] = 1.0 if diag.baseline_compatible else 0.0
    return metrics


def _diagnostics_to_mlflow_tags(diag: RunDiagnostics) -> dict[str, str]:
    """Run tags for MLflow ``set_tags``."""
    tags: dict[str, str] = {
        "jtb_baseline_compatible": str(diag.baseline_compatible).lower(),
    }
    if diag.baseline_run_id:
        tags["jtb_baseline_run_id"] = str(diag.baseline_run_id)
    if diag.baseline_fingerprint:
        tags["jtb_baseline_fingerprint"] = str(diag.baseline_fingerprint)
    if diag.dataset_fingerprint:
        tags["jtb_diag_dataset_fingerprint"] = str(diag.dataset_fingerprint)
    return tags


@_guarded("log_diagnostics")
def log_diagnostics_mlflow(logger: MLflowLogger, diag: RunDiagnostics) -> None:
    """Log diagnostics (metrics, tags, ``diagnostics.json``) to the active run."""
    if not logger.enabled or logger._backend is None:
        return
    metrics = _diagnostics_to_mlflow_metrics(diag)
    if metrics:
        logger._backend.log_metrics(metrics)
    tags = _diagnostics_to_mlflow_tags(diag)
    if tags:
        logger._backend.set_tags(tags)
    logger.log_artifact_json("diagnostics.json", diag.to_serializable())


@_guarded("log_threshold_report")
def log_threshold_report_mlflow(logger: MLflowLogger, report: RunThresholdReport) -> None:
    """Log north-star gates (metrics, tags, ``threshold_gates.json``)."""
    if not logger.enabled or logger._backend is None:
        return
    m = threshold_report_to_mlflow_metrics(report)
    if m:
        logger._backend.log_metrics(m)
    logger._backend.set_tags({"jtb_worst_gate_status": report.worst_status().value})
    logger.log_artifact_json("threshold_gates.json", threshold_report_to_serializable(report))


@_guarded("log_plotly_html")
def log_plotly_html_mlflow(
    logger: MLflowLogger,
    html: str,
    *,
    artifact_subdir: str = "plotly",
    filename: str = "risk_evidence.html",
) -> None:
    """Attach Plotly HTML as an MLflow artifact."""
    if not logger.enabled or logger._backend is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / filename
        out.write_text(html, encoding="utf-8")
        logger._backend.log_artifact(str(out), artifact_path=artifact_subdir)
