"""MLflow adapter for aggregate run logging.

Runs are the MLflow unit of granularity here: one execution of the
runner on a dataset x judge set x model config maps to one MLflow run.
Per-row traces live in :mod:`src.observability.langfuse_tracer`; MLflow
is used strictly for parameters, aggregate metrics, and small artifacts
that summarize the run.

Design principles
-----------------

1. **Never break a run.** Every method is wrapped in ``_guarded`` which
   catches any backend exception and logs a single warning. This satisfies
   the observability hard rule ("observability failures MUST NOT break a
   run") and means callers don't need their own try/except.

2. **Zero imports of ``mlflow`` at module load.** The backend is resolved
   lazily inside :class:`MLflowLogger`. This keeps the module importable
   in environments where mlflow is not installed (CI, offline demos).

3. **Injectable backend.** Tests pass a fake object that records calls;
   production resolves to the real ``mlflow`` module. The logger never
   reaches into backend internals - it uses the small surface declared
   by :class:`_MLflowBackend`.

4. **Disabled == silent.** A logger without a tracking URI, or one where
   ``mlflow`` can't be imported, flips into disabled mode. Public methods
   return no-ops. Tests can still exercise the call graph via
   :class:`FakeMLflowBackend`.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from src.evaluation.agreement import AgreementReport, PillarAgreement
from src.evaluation.diagnostics import RunDiagnostics
from src.evaluation.reviewer_analysis import ReviewerAnalytics
from src.evaluation.slices import SliceReport
from src.evaluation.thresholds import RunThresholdReport
from src.observability.run_metadata import (
    RunMetadata,
    to_mlflow_params,
    to_mlflow_tags,
)
from src.orchestration.runner import RunResult, RunSummary

__all__ = [
    "MLflowLogger",
    "build_mlflow_logger",
]


logger = logging.getLogger(__name__)

# Sentinel: caller omitted ``tracking_uri`` → fall back to AppSettings.
_TRACKING_URI_DEFAULT: Any = object()


# ---------------------------------------------------------------------------
# Backend contract
# ---------------------------------------------------------------------------


class _MLflowBackend(Protocol):
    """Minimal surface of the ``mlflow`` module that we use.

    Declaring a Protocol lets tests ship a fake without importing mlflow
    at all, and lets the real backend be the mlflow module itself
    (Python's structural typing handles both).
    """

    def set_tracking_uri(self, uri: str) -> None: ...

    def set_experiment(self, experiment_name: str) -> Any: ...

    def start_run(
        self,
        run_name: str | None = ...,
        tags: Mapping[str, str] | None = ...,
    ) -> Any: ...

    def log_params(self, params: Mapping[str, Any]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = ...) -> None: ...

    def log_text(self, text: str, artifact_file: str) -> None: ...

    def log_artifact(self, local_path: str, artifact_path: str | None = ...) -> None: ...

    def set_tags(self, tags: Mapping[str, str]) -> None: ...

    def end_run(self, status: str = ...) -> None: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# One-shot warning cache keyed by error type. Many calls will fail for the
# same reason (wrong URI, auth) and we don't want to spam logs.
_warned_errors: set[str] = set()


def _warn_once(tag: str, exc: BaseException) -> None:
    key = f"{tag}:{type(exc).__name__}"
    if key in _warned_errors:
        return
    _warned_errors.add(key)
    logger.warning("MLflowLogger.%s failed (%s): %s", tag, type(exc).__name__, exc)


def _guarded(tag: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: swallow backend exceptions, warn once per error type."""

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        def inner(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                _warn_once(tag, exc)
                return None

        inner.__name__ = fn.__name__
        inner.__doc__ = fn.__doc__
        return inner

    return wrap


# MLflow metric keys cannot contain certain characters. Keep this permissive:
# MLflow 3.x accepts letters, digits, underscores, dashes, periods, slashes,
# and spaces. We replace everything else with underscore.
_SAFE_METRIC_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./ ")


def _sanitize(label: str) -> str:
    return "".join(ch if ch in _SAFE_METRIC_CHARS else "_" for ch in label)


def _resolve_default_backend() -> _MLflowBackend | None:
    """Import the real ``mlflow`` module lazily."""
    try:
        import mlflow

        return mlflow  # type: ignore[return-value]
    except ImportError:
        logger.info(
            "mlflow is not installed; MLflowLogger will run in disabled mode. "
            "Install mlflow-skinny to enable experiment logging.",
        )
        return None


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class MLflowLogger:
    """MLflow run adapter.

    Usage:

        >>> logger = build_mlflow_logger()  # reads settings
        >>> with logger.active_run(metadata):
        ...     logger.log_run_summary(result.summary)
        ...     logger.log_agreement_report(report)
    """

    def __init__(
        self,
        *,
        tracking_uri: str | None,
        experiment_name: str = "llm-judge-testbench",
        enabled: bool = True,
        backend: _MLflowBackend | None = None,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name

        resolved_backend = backend if backend is not None else _resolve_default_backend()
        # Disabled when: caller forced enabled=False, no tracking URI,
        # or the backend couldn't be imported. In all three cases, every
        # public method becomes a cheap no-op.
        self._enabled = bool(enabled and tracking_uri and resolved_backend is not None)
        self._backend: _MLflowBackend | None = resolved_backend if self._enabled else None
        self._run_active = False

        if self._enabled and self._backend is not None:
            self._configure_backend()

    # ---- Public properties --------------------------------------------

    @property
    def enabled(self) -> bool:
        """True when metrics will actually reach the tracking backend."""
        return self._enabled

    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    # ---- Backend configuration ----------------------------------------

    @_guarded("configure")
    def _configure_backend(self) -> None:
        assert self._backend is not None
        assert self._tracking_uri is not None
        self._backend.set_tracking_uri(self._tracking_uri)
        self._backend.set_experiment(self._experiment_name)

    # ---- Run lifecycle ------------------------------------------------

    @_guarded("start_run")
    def start_run(self, meta: RunMetadata) -> None:
        """Open an MLflow run and log params + tags derived from ``meta``.

        Safe to call multiple times: a second call is a no-op (MLflow
        would otherwise nest runs, which is confusing for dashboards).
        """
        if not self._enabled or self._backend is None:
            return
        if self._run_active:
            return
        self._backend.start_run(run_name=meta.run_id, tags=to_mlflow_tags(meta))
        self._backend.log_params(to_mlflow_params(meta))
        self._run_active = True

    @_guarded("end_run")
    def end_run(self, status: str = "FINISHED") -> None:
        """Close the active run. Idempotent."""
        if not self._enabled or self._backend is None:
            return
        if not self._run_active:
            return
        self._backend.end_run(status=status)
        self._run_active = False

    @contextmanager
    def active_run(self, meta: RunMetadata) -> Iterator[MLflowLogger]:
        """Context manager that starts and ends a run around ``meta``.

        Always calls :meth:`end_run` on exit, even on exception. If the
        wrapped block raises, the MLflow run is ended with status
        ``FAILED`` so the experiment UI shows the failure clearly.
        """
        self.start_run(meta)
        status = "FINISHED"
        try:
            yield self
        except Exception:
            status = "FAILED"
            raise
        finally:
            self.end_run(status)

    # ---- Metric logging -----------------------------------------------

    @_guarded("log_run_summary")
    def log_run_summary(self, summary: RunSummary) -> None:
        """Log run-level aggregates (counts, latency percentiles, tokens)."""
        if not self._enabled or self._backend is None:
            return
        metrics: dict[str, float] = {
            "total_tasks": float(summary.total_tasks),
            "rows_successfully_scored": float(summary.succeeded),
            "rows_failed_parsing": float(summary.failed),
            "cache_hits": float(summary.cache_hits),
            "aborted": float(summary.aborted),
            "duration_s": float(summary.duration_s),
            "latency_ms_p50": float(summary.latency_ms_p50),
            "latency_ms_p95": float(summary.latency_ms_p95),
            "total_input_tokens": float(summary.total_input_tokens),
            "total_output_tokens": float(summary.total_output_tokens),
        }
        for pillar, (succeeded, failed) in summary.pillar_stats.items():
            metrics[f"pillar_succeeded_{pillar}"] = float(succeeded)
            metrics[f"pillar_failed_{pillar}"] = float(failed)
        self._backend.log_metrics(metrics)

    @_guarded("log_agreement_report")
    def log_agreement_report(self, report: AgreementReport) -> None:
        """Log per-pillar agreement metrics and the optional overall view.

        Metrics that are ``None`` (e.g. undefined kappa for tiny samples)
        are skipped rather than forced to zero; zero would be misleading.
        """
        if not self._enabled or self._backend is None:
            return
        metrics: dict[str, float] = {}
        for pillar in report.pillars():
            agreement = report.per_pillar[pillar]
            metrics.update(_pillar_metrics(pillar, agreement))
        if report.overall is not None:
            metrics.update(_pillar_metrics("overall", report.overall))
        if metrics:
            self._backend.log_metrics(metrics)

    @_guarded("log_slice_report")
    def log_slice_report(self, sliced: SliceReport) -> None:
        """Log a handful of per-category numbers for quick filtering.

        Full slice data lives in the artifact (see :meth:`log_artifact_json`).
        Here we only surface ``support`` and ``severity_aware_alignment``
        per (pillar, slice) so MLflow's sparkline view stays useful.
        """
        if not self._enabled or self._backend is None:
            return
        metrics: dict[str, float] = {}
        for slice_name in sliced.slices():
            bucket = sliced.per_slice[slice_name]
            for pillar in bucket.pillars():
                agreement = bucket.per_pillar[pillar]
                prefix = f"slice/{_sanitize(slice_name)}/{pillar}"
                metrics[f"{prefix}/support"] = float(agreement.support)
                metrics[f"{prefix}/severity_aware_alignment"] = float(
                    agreement.severity_aware_alignment
                )
        if metrics:
            self._backend.log_metrics(metrics)

    @_guarded("log_reviewer_analytics")
    def log_reviewer_analytics(self, analytics: ReviewerAnalytics) -> None:
        """Log per-reviewer summary metrics plus pair overlaps.

        Uses the reviewer's ``AgreementReport`` so severity alignment and
        MAE are consistent with the main agreement panel in the UI.
        """
        if not self._enabled or self._backend is None:
            return
        metrics: dict[str, float] = {}
        for reviewer in analytics.reviewers():
            stats = analytics.per_reviewer.get(reviewer)
            if stats is None:
                continue
            reviewer_key = _sanitize(reviewer)
            metrics[f"reviewer/{reviewer_key}/sample_count"] = float(stats.sample_count)
            for pillar in stats.report.pillars():
                agreement = stats.report.per_pillar[pillar]
                metrics.update(_pillar_metrics(f"reviewer/{reviewer_key}/{pillar}", agreement))
            for pillar, per_pillar_stats in stats.per_pillar.items():
                prefix = f"reviewer/{reviewer_key}/{pillar}"
                metrics[f"{prefix}/disagreement_rate"] = float(per_pillar_stats.disagreement_rate)
                metrics[f"{prefix}/within_1_agreement"] = float(per_pillar_stats.within_1_agreement)
                metrics[f"{prefix}/large_miss_rate"] = float(per_pillar_stats.large_miss_rate)
        for pair in analytics.reviewer_pairs:
            prefix = f"reviewer_pair/{_sanitize(pair.reviewer_a)}__{_sanitize(pair.reviewer_b)}"
            metrics[f"{prefix}/overlap"] = float(pair.overlap)
            metrics[f"{prefix}/exact_match_rate"] = float(pair.exact_match_rate)
            metrics[f"{prefix}/within_1_rate"] = float(pair.within_1_rate)
            metrics[f"{prefix}/large_miss_rate"] = float(pair.large_miss_rate)
        if metrics:
            self._backend.log_metrics(metrics)

    def log_threshold_report(self, report: RunThresholdReport) -> None:
        """Log north-star gate encodings + JSON artifact."""
        from src.observability.mlflow_risk_logging import log_threshold_report_mlflow

        log_threshold_report_mlflow(self, report)

    def log_diagnostics(self, diag: RunDiagnostics) -> None:
        """Log residual / OLS / drift scalars plus full diagnostics JSON."""
        from src.observability.mlflow_risk_logging import log_diagnostics_mlflow

        log_diagnostics_mlflow(self, diag)

    def log_plotly_html(
        self,
        html: str,
        *,
        artifact_subdir: str = "plotly",
        filename: str = "risk_evidence.html",
    ) -> None:
        """Write raw HTML (e.g. Plotly export) under ``artifacts/{artifact_subdir}/``."""
        from src.observability.mlflow_risk_logging import log_plotly_html_mlflow

        log_plotly_html_mlflow(self, html, artifact_subdir=artifact_subdir, filename=filename)

    # ---- Artifact logging ---------------------------------------------

    @_guarded("log_artifact_json")
    def log_artifact_json(self, name: str, payload: Any) -> None:
        """Serialize ``payload`` as pretty JSON and attach it to the run.

        Prefers ``log_text`` (avoids temp files) and falls back to
        ``log_artifact`` when the backend only supports the latter.
        """
        if not self._enabled or self._backend is None:
            return
        body = json.dumps(payload, indent=2, sort_keys=True, default=str)
        if hasattr(self._backend, "log_text"):
            self._backend.log_text(body, artifact_file=name)
            return
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / name
            artifact_path.write_text(body, encoding="utf-8")
            self._backend.log_artifact(str(artifact_path))

    @_guarded("log_run_result")
    def log_run_result(self, result: RunResult, *, max_outcomes: int = 500) -> None:
        """Convenience: log summary + a truncated outcomes snapshot artifact.

        ``max_outcomes`` caps the artifact size so we don't push megabytes
        of judge JSON into every MLflow run. Full outcomes belong in the
        exports layer (Stage 11).
        """
        self.log_run_summary(result.summary)
        snapshot = {
            "run_id": result.run_id,
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "summary": {
                "total_tasks": result.summary.total_tasks,
                "succeeded": result.summary.succeeded,
                "failed": result.summary.failed,
            },
            "outcomes_sample": [
                {
                    "record_id": o.record_id,
                    "pillar": o.pillar,
                    "latency_ms": o.latency_ms,
                    "error_type": o.error_type,
                    "score": o.result.score if o.result is not None else None,
                }
                for o in result.outcomes[:max_outcomes]
            ],
            "outcomes_truncated": len(result.outcomes) > max_outcomes,
        }
        self.log_artifact_json("run_result.json", snapshot)


# ---------------------------------------------------------------------------
# Metric extraction helpers (module-level so tests can reuse)
# ---------------------------------------------------------------------------


def _pillar_metrics(prefix: str, agreement: PillarAgreement) -> dict[str, float]:
    """Flatten a :class:`PillarAgreement` into MLflow-friendly scalars.

    ``None`` values drop out entirely so MLflow's plots don't show bogus
    zeros for undefined kappa/Spearman cases.
    """
    out: dict[str, float] = {
        f"{prefix}/support": float(agreement.support),
        f"{prefix}/exact_match_rate": float(agreement.exact_match_rate),
        f"{prefix}/within_1_rate": float(agreement.within_1_rate),
        f"{prefix}/off_by_2_rate": float(agreement.off_by_2_rate),
        f"{prefix}/off_by_3_plus_rate": float(agreement.off_by_3_plus_rate),
        f"{prefix}/mean_absolute_error": float(agreement.mean_absolute_error),
        f"{prefix}/severity_aware_alignment": float(agreement.severity_aware_alignment),
    }
    if agreement.weighted_kappa is not None:
        out[f"{prefix}/weighted_kappa"] = float(agreement.weighted_kappa)
    if agreement.spearman_correlation is not None:
        out[f"{prefix}/spearman_correlation"] = float(agreement.spearman_correlation)
    return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_mlflow_logger(
    *,
    tracking_uri: str | None | Any = _TRACKING_URI_DEFAULT,
    experiment_name: str | None = None,
    enabled: bool = True,
    backend: _MLflowBackend | None = None,
) -> MLflowLogger:
    """Construct a logger, defaulting to :class:`src.core.settings.AppSettings`.

    If ``tracking_uri`` is omitted, it is read from settings. If it is
    passed explicitly as ``None``, the logger stays disabled (no env
    fallback), which matches "observability off" call sites and tests.

    When ``experiment_name`` is ``None``, the value from settings is used.
    """
    from src.core.settings import get_settings

    settings = get_settings()
    if tracking_uri is _TRACKING_URI_DEFAULT:
        tracking_uri = settings.mlflow_tracking_uri
    if experiment_name is None:
        experiment_name = settings.mlflow_experiment_name
    return MLflowLogger(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name or "llm-judge-testbench",
        enabled=enabled,
        backend=backend,
    )
