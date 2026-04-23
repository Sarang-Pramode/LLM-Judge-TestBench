"""Unit tests for :class:`src.observability.mlflow_logger.MLflowLogger`.

We never import mlflow here: the logger accepts an injectable backend,
so we ship a :class:`FakeMLflowBackend` that records every call. This
lets us assert on:

- Configuration (tracking URI + experiment name pushed during init).
- Run lifecycle (start/end, idempotency, context-manager semantics).
- Metric extraction (agreement, slice, reviewer, run-summary mappings).
- Guarded failures (backend exceptions never propagate).
- Disabled mode (no backend calls at all when missing creds).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from src.evaluation.agreement import AgreementReport, PillarAgreement
from src.evaluation.diagnostics import (
    OLSResult,
    PillarDiagnostics,
    RunDiagnostics,
    histogram_pmf,
)
from src.evaluation.reviewer_analysis import (
    ReviewerAnalytics,
    ReviewerPairStats,
    ReviewerPillarStats,
    ReviewerStats,
)
from src.evaluation.slices import SliceReport
from src.evaluation.thresholds import (
    GateResult,
    GateStatus,
    PillarGateReport,
    RunThresholdReport,
)
from src.observability.mlflow_logger import (
    MLflowLogger,
    _pillar_metrics,
    _sanitize,
    build_mlflow_logger,
)
from src.observability.mlflow_risk_logging import log_diagnostics_mlflow
from src.observability.run_metadata import RunMetadata

# ---------------------------------------------------------------------------
# Fake backend
# ---------------------------------------------------------------------------


class FakeMLflowBackend:
    """Duck-typed stand-in for the ``mlflow`` module.

    Records every call in ``calls`` so tests can assert precisely.
    """

    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._fail_on = fail_on or set()

    def _record(self, name: str, **kwargs: Any) -> None:
        if name in self._fail_on:
            raise RuntimeError(f"fake backend configured to fail on {name!r}")
        self.calls.append((name, kwargs))

    def set_tracking_uri(self, uri: str) -> None:
        self._record("set_tracking_uri", uri=uri)

    def set_experiment(self, experiment_name: str) -> Any:
        self._record("set_experiment", experiment_name=experiment_name)
        return {"experiment_id": "e-1"}

    def start_run(self, run_name: str | None = None, tags: Mapping[str, str] | None = None) -> Any:
        self._record("start_run", run_name=run_name, tags=dict(tags or {}))
        return {"info": {"run_id": "fake-1"}}

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._record("log_params", params=dict(params))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._record("log_metrics", metrics=dict(metrics), step=step)

    def log_text(self, text: str, artifact_file: str) -> None:
        self._record("log_text", text=text, artifact_file=artifact_file)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self._record("log_artifact", local_path=local_path, artifact_path=artifact_path)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._record("set_tags", tags=dict(tags))

    def end_run(self, status: str = "FINISHED") -> None:
        self._record("end_run", status=status)

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _meta() -> RunMetadata:
    return RunMetadata(
        run_id="r-1",
        started_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        dataset_fingerprint="sha256:abc",
        dataset_row_count=3,
        pillars=("factual_accuracy",),
        provider="mock",
        model_alias_by_pillar={"factual_accuracy": "judge-default"},
        model_name_by_pillar={"factual_accuracy": "mock"},
        prompt_version_by_pillar={"factual_accuracy": "v1"},
        rubric_version_by_pillar={"factual_accuracy": "rv1"},
        run_config_hash="sha256:cfg",
        kb_version="kb-v1",
    )


def _pillar(pillar: str = "factual_accuracy", support: int = 5) -> PillarAgreement:
    return PillarAgreement(
        pillar=pillar,
        support=support,
        exact_match_rate=0.6,
        within_1_rate=0.8,
        off_by_2_rate=0.15,
        off_by_3_plus_rate=0.05,
        mean_absolute_error=0.5,
        severity_aware_alignment=0.75,
        weighted_kappa=0.42,
        spearman_correlation=0.55,
        judge_score_distribution={4: 3, 5: 2},
        human_score_distribution={4: 3, 5: 2},
        confusion_matrix=[[0] * 5 for _ in range(5)],
    )


def _enabled_logger() -> tuple[MLflowLogger, FakeMLflowBackend]:
    backend = FakeMLflowBackend()
    logger = MLflowLogger(
        tracking_uri="http://localhost:5000",
        experiment_name="exp",
        enabled=True,
        backend=backend,
    )
    return logger, backend


# ---------------------------------------------------------------------------
# Enablement
# ---------------------------------------------------------------------------


class TestEnablement:
    def test_missing_uri_disables(self) -> None:
        logger = MLflowLogger(tracking_uri=None, backend=FakeMLflowBackend())
        assert logger.enabled is False

    def test_explicit_disable_blocks_backend_calls(self) -> None:
        backend = FakeMLflowBackend()
        logger = MLflowLogger(tracking_uri="http://localhost:5000", enabled=False, backend=backend)
        assert logger.enabled is False
        logger.start_run(_meta())
        logger.log_run_summary(_dummy_summary())  # must not reach backend
        assert backend.calls == []

    def test_enabled_configures_backend_on_init(self) -> None:
        logger, backend = _enabled_logger()
        assert logger.enabled is True
        names = backend.names()
        assert "set_tracking_uri" in names
        assert "set_experiment" in names


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_run_logs_params_and_tags(self) -> None:
        logger, backend = _enabled_logger()
        logger.start_run(_meta())
        names = backend.names()
        # start_run precedes log_params
        assert names.index("start_run") < names.index("log_params")
        # params contain per-pillar keys and top-level keys
        params = dict(
            next(kwargs for name, kwargs in backend.calls if name == "log_params")["params"]
        )
        assert params["run_id"] == "r-1"
        assert params["prompt_version_factual_accuracy"] == "v1"
        # tags include namespace
        tag_payload = next(kwargs for name, kwargs in backend.calls if name == "start_run")["tags"]
        assert any(k.startswith("jtb.") for k in tag_payload)

    def test_start_run_is_idempotent(self) -> None:
        logger, backend = _enabled_logger()
        logger.start_run(_meta())
        first_count = sum(1 for name, _ in backend.calls if name == "start_run")
        logger.start_run(_meta())
        second_count = sum(1 for name, _ in backend.calls if name == "start_run")
        assert first_count == second_count == 1

    def test_end_run_before_start_is_noop(self) -> None:
        logger, backend = _enabled_logger()
        logger.end_run()
        assert "end_run" not in backend.names()

    def test_active_run_context_ends_on_exception(self) -> None:
        logger, backend = _enabled_logger()
        with pytest.raises(ValueError):
            with logger.active_run(_meta()):
                raise ValueError("boom")
        end_calls = [kwargs for name, kwargs in backend.calls if name == "end_run"]
        assert end_calls and end_calls[-1]["status"] == "FAILED"

    def test_active_run_context_ends_normally(self) -> None:
        logger, backend = _enabled_logger()
        with logger.active_run(_meta()):
            pass
        end_calls = [kwargs for name, kwargs in backend.calls if name == "end_run"]
        assert end_calls and end_calls[-1]["status"] == "FINISHED"


# ---------------------------------------------------------------------------
# Metric flattening
# ---------------------------------------------------------------------------


def _dummy_summary() -> Any:
    from src.orchestration.runner import RunSummary

    return RunSummary(
        total_tasks=2,
        succeeded=2,
        failed=0,
        cache_hits=0,
        aborted=0,
        duration_s=0.25,
        latency_ms_p50=120.0,
        latency_ms_p95=130.0,
        total_input_tokens=10,
        total_output_tokens=5,
        pillar_stats={"factual_accuracy": (2, 0)},
    )


class TestLogRunSummary:
    def test_all_fields_emitted(self) -> None:
        logger, backend = _enabled_logger()
        logger.start_run(_meta())
        logger.log_run_summary(_dummy_summary())
        metrics_call = next(kwargs for name, kwargs in backend.calls if name == "log_metrics")
        metrics = metrics_call["metrics"]
        for key in (
            "total_tasks",
            "rows_successfully_scored",
            "rows_failed_parsing",
            "cache_hits",
            "duration_s",
            "latency_ms_p50",
            "latency_ms_p95",
            "total_input_tokens",
            "total_output_tokens",
            "pillar_succeeded_factual_accuracy",
            "pillar_failed_factual_accuracy",
        ):
            assert key in metrics

    def test_metrics_are_floats(self) -> None:
        pillar_metrics = _pillar_metrics("factual_accuracy", _pillar())
        assert all(isinstance(v, float) for v in pillar_metrics.values())


class TestLogAgreement:
    def test_pillar_and_overall(self) -> None:
        logger, backend = _enabled_logger()
        logger.start_run(_meta())
        report = AgreementReport(
            per_pillar={"factual_accuracy": _pillar()},
            overall=_pillar("overall", support=5),
        )
        logger.log_agreement_report(report)
        metrics = next(kwargs for name, kwargs in backend.calls if name == "log_metrics")["metrics"]
        assert metrics["factual_accuracy/exact_match_rate"] == pytest.approx(0.6)
        assert metrics["factual_accuracy/weighted_kappa"] == pytest.approx(0.42)
        assert metrics["overall/severity_aware_alignment"] == pytest.approx(0.75)

    def test_none_kappa_is_skipped(self) -> None:
        pillar = _pillar()
        pillar = PillarAgreement(**{**pillar.__dict__, "weighted_kappa": None})
        metrics = _pillar_metrics("factual_accuracy", pillar)
        assert "factual_accuracy/weighted_kappa" not in metrics


class TestLogSliceReport:
    def test_per_slice_pillar_metric(self) -> None:
        logger, backend = _enabled_logger()
        logger.start_run(_meta())
        report = SliceReport(
            dimension="category",
            per_slice={"billing": AgreementReport(per_pillar={"factual_accuracy": _pillar()})},
            slice_counts={"billing": 5},
        )
        logger.log_slice_report(report)
        metrics = next(kwargs for name, kwargs in backend.calls if name == "log_metrics")["metrics"]
        assert "slice/billing/factual_accuracy/support" in metrics
        assert "slice/billing/factual_accuracy/severity_aware_alignment" in metrics

    def test_unsafe_slice_chars_sanitized(self) -> None:
        assert _sanitize("a/b c?") == "a/b c_"
        assert _sanitize("hello<>") == "hello__"


class TestLogReviewerAnalytics:
    def test_includes_agreement_and_pair_metrics(self) -> None:
        logger, backend = _enabled_logger()
        logger.start_run(_meta())
        stats = ReviewerStats(
            reviewer="alice",
            sample_count=3,
            report=AgreementReport(per_pillar={"factual_accuracy": _pillar()}),
            per_pillar={
                "factual_accuracy": ReviewerPillarStats(
                    pillar="factual_accuracy",
                    support=3,
                    avg_human_score=4.0,
                    avg_judge_score=4.0,
                    disagreement_rate=0.1,
                    within_1_agreement=0.9,
                    large_miss_rate=0.0,
                )
            },
        )
        analytics = ReviewerAnalytics(
            per_reviewer={"alice": stats},
            reviewer_pairs=[
                ReviewerPairStats(
                    reviewer_a="alice",
                    reviewer_b="bob",
                    overlap=3,
                    exact_match_rate=0.66,
                    within_1_rate=1.0,
                    large_miss_rate=0.0,
                )
            ],
        )
        logger.log_reviewer_analytics(analytics)
        metrics = next(kwargs for name, kwargs in backend.calls if name == "log_metrics")["metrics"]
        assert metrics["reviewer/alice/sample_count"] == 3.0
        assert "reviewer/alice/factual_accuracy/exact_match_rate" in metrics
        assert "reviewer/alice/factual_accuracy/disagreement_rate" in metrics
        assert "reviewer_pair/alice__bob/overlap" in metrics


# ---------------------------------------------------------------------------
# Artifact + result logging
# ---------------------------------------------------------------------------


class TestArtifacts:
    def test_log_artifact_json_prefers_log_text(self) -> None:
        logger, backend = _enabled_logger()
        logger.log_artifact_json("summary.json", {"a": 1})
        text_calls = [kwargs for name, kwargs in backend.calls if name == "log_text"]
        assert text_calls and text_calls[-1]["artifact_file"] == "summary.json"
        # Body is valid JSON
        import json

        payload = json.loads(text_calls[-1]["text"])
        assert payload == {"a": 1}

    def test_log_run_result_emits_summary_and_snapshot(self) -> None:
        logger, backend = _enabled_logger()
        logger.start_run(_meta())

        # Build a tiny RunResult
        import datetime

        from src.judges.base import JudgeOutcome
        from src.llm.base import LLMUsage
        from src.orchestration.runner import RunResult

        outcomes = [
            JudgeOutcome(
                pillar="factual_accuracy",
                record_id="r1",
                latency_ms=12.0,
                attempts=1,
                usage=LLMUsage(input_tokens=1, output_tokens=1),
                model_name="mock",
            )
        ]
        result = RunResult(
            run_id="r-1",
            started_at=datetime.datetime.now(datetime.UTC),
            finished_at=datetime.datetime.now(datetime.UTC),
            outcomes=outcomes,
            summary=_dummy_summary(),
        )
        logger.log_run_result(result)
        names = backend.names()
        assert names.count("log_metrics") >= 1
        assert any(
            kwargs.get("artifact_file") == "run_result.json"
            for name, kwargs in backend.calls
            if name == "log_text"
        )


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


class TestResilience:
    def test_start_run_failure_is_swallowed(self) -> None:
        backend = FakeMLflowBackend(fail_on={"start_run"})
        logger = MLflowLogger(
            tracking_uri="http://x", experiment_name="e", enabled=True, backend=backend
        )
        # Must not raise.
        logger.start_run(_meta())

    def test_log_metrics_failure_is_swallowed(self) -> None:
        backend = FakeMLflowBackend(fail_on={"log_metrics"})
        logger = MLflowLogger(
            tracking_uri="http://x", experiment_name="e", enabled=True, backend=backend
        )
        logger.start_run(_meta())
        # Must not raise.
        logger.log_run_summary(_dummy_summary())
        logger.log_agreement_report(AgreementReport(per_pillar={"factual_accuracy": _pillar()}))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestDiagnosticsAndThresholds:
    def test_log_diagnostics_writes_metrics_and_json(self) -> None:
        backend = FakeMLflowBackend()
        logger = MLflowLogger(
            tracking_uri="http://x", experiment_name="e", enabled=True, backend=backend
        )
        logger.start_run(_meta())
        pmf = histogram_pmf([3, 3, 4, 4, 5])
        diag = RunDiagnostics(
            pillars={
                "factual_accuracy": PillarDiagnostics(
                    pillar="factual_accuracy",
                    support=5,
                    judge_pmf=pmf,
                    human_pmf=pmf,
                    mean_residual_judge_minus_human=0.0,
                    pct_positive_residual=0.2,
                    ols_human_on_judge=OLSResult(1.0, 0.0, 0.95),
                    js_vs_baseline=0.01,
                    psi_vs_baseline=0.02,
                )
            },
            baseline_compatible=True,
        )
        log_diagnostics_mlflow(logger, diag)
        metric_calls = [c for c in backend.calls if c[0] == "log_metrics"]
        assert metric_calls
        tag_calls = [c for c in backend.calls if c[0] == "set_tags"]
        assert tag_calls
        assert any(
            c[1]["tags"].get("jtb_baseline_compatible") == "true" for c in tag_calls
        )
        text_calls = [c for c in backend.calls if c[0] == "log_text"]
        assert any("diagnostics.json" in str(c[1].get("artifact_file", "")) for c in text_calls)

    def test_log_threshold_report_writes_metrics_and_json(self) -> None:
        backend = FakeMLflowBackend()
        logger = MLflowLogger(
            tracking_uri="http://x", experiment_name="e", enabled=True, backend=backend
        )
        logger.start_run(_meta())
        pr = PillarGateReport(
            pillar="relevance",
            gates=(GateResult("within_1_rate", GateStatus.PASS, 0.9, "ok"),),
            overall=GateStatus.PASS,
        )
        logger.log_threshold_report(RunThresholdReport(version="1", per_pillar={"relevance": pr}))
        assert any(c[0] == "log_metrics" for c in backend.calls)
        assert any(
            c[0] == "set_tags" and c[1]["tags"].get("jtb_worst_gate_status") == "pass"
            for c in backend.calls
        )
        assert any(
            c[0] == "log_text" and "threshold_gates.json" in c[1].get("artifact_file", "")
            for c in backend.calls
        )

    def test_log_plotly_html_uses_log_artifact(self) -> None:
        backend = FakeMLflowBackend()
        logger = MLflowLogger(
            tracking_uri="http://x", experiment_name="e", enabled=True, backend=backend
        )
        logger.start_run(_meta())
        logger.log_plotly_html("<html>x</html>")
        assert any(c[0] == "log_artifact" for c in backend.calls)


class TestFactory:
    def test_build_with_explicit_args_disables_when_no_uri(self) -> None:
        logger = build_mlflow_logger(tracking_uri=None, enabled=True)
        assert logger.enabled is False

    def test_build_with_uri_and_fake_backend_enables(self) -> None:
        backend = FakeMLflowBackend()
        logger = build_mlflow_logger(
            tracking_uri="http://x", experiment_name="exp", backend=backend
        )
        assert logger.enabled is True
        assert logger.experiment_name == "exp"
