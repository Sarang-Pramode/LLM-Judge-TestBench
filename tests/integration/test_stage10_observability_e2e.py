"""Stage 10 end-to-end: runner + observability, stub backends.

Exercises the full observability path without touching the real MLflow
or Langfuse services:

1. Builds a small dataset and bundles for two pillars (keeping it
   bounded so the test runs in milliseconds).
2. Wires an ``MLflowLogger`` + ``LangfuseTracer`` backed by the same
   fakes the unit tests use.
3. Runs ``EvaluationRunner`` with the observability callbacks plugged
   in via :func:`build_observability_callbacks`.
4. Asserts that:
   - Every task produced exactly one Langfuse child observation.
   - MLflow received params + metrics + an artifact.
   - The fingerprints referenced in MLflow match the content-based
     fingerprint we handed the runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.constants import PILLARS
from src.core.types import NormalizedRow, RunContext
from src.evaluation.agreement import compute_agreement_report
from src.evaluation.join import join_outcomes_with_labels
from src.judges import JudgeCoreOutput, load_judge_bundle
from src.llm.base import LLMUsage
from src.llm.mock_client import MockLLMClient
from src.observability import (
    build_langfuse_tracer,
    build_mlflow_logger,
    build_observability_callbacks,
    build_run_metadata,
    dataset_fingerprint,
    run_config_hash,
)
from src.orchestration import ConcurrencyPolicy, EvaluationRunner, RunPlan
from tests.unit.test_observability_langfuse import FakeLangfuseBackend
from tests.unit.test_observability_mlflow import FakeMLflowBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGES_DIR = REPO_ROOT / "configs" / "judges"
RUBRICS_DIR = REPO_ROOT / "configs" / "rubrics"


def _rows() -> list[NormalizedRow]:
    return [
        NormalizedRow(
            record_id="r-1",
            user_input="How do I dispute a transaction?",
            agent_output="You can start a dispute by opening the app...",
            category="billing",
            retrieved_context=["Dispute policy: you can initiate within 60 days."],
            label_factual_accuracy=4,
            label_relevance=5,
        ),
        NormalizedRow(
            record_id="r-2",
            user_input="Where is my card PIN?",
            agent_output="Your PIN is 1234.",
            category="security",
            retrieved_context=["We never reveal PINs in support chat."],
            label_factual_accuracy=1,
            label_relevance=3,
        ),
    ]


def _scripted_output(pillar: str, score: int) -> JudgeCoreOutput:
    return JudgeCoreOutput(
        pillar=pillar,
        score=score,
        confidence=0.8,
        decision_summary=f"Stage10 {pillar} score {score}.",
        evidence_for_score=[],
        failure_tags=[],
        rubric_anchor=score,
        why_not_higher="Minor concern.",
        why_not_lower="Broadly acceptable.",
    )


def _build_plan(
    pillars: list[str], rows: list[NormalizedRow], on_outcome: Any, on_progress: Any
) -> tuple[RunPlan, dict[str, MockLLMClient]]:
    bundles = {
        p: load_judge_bundle(JUDGES_DIR / f"{p}.yaml", rubric_root=RUBRICS_DIR) for p in pillars
    }
    llms = {
        p: MockLLMClient(
            model_name=f"mock-{p}",
            structured_script=[_scripted_output(p, 4) for _ in rows],
            usage=LLMUsage(input_tokens=5, output_tokens=2),
        )
        for p in pillars
    }
    fingerprint = dataset_fingerprint(rows)
    run_ctx = RunContext(
        run_id="stage10-e2e",
        dataset_fingerprint=fingerprint,
        model_alias="mock-judge",
        run_config_hash=run_config_hash({"pillars": pillars}),
    )
    plan = RunPlan(
        rows=rows,
        pillars=pillars,
        bundles=bundles,
        llm_by_pillar=llms,
        run_context=run_ctx,
        concurrency=ConcurrencyPolicy(max_workers=4),
        on_outcome=on_outcome,
        on_progress=on_progress,
    )
    return plan, llms


def test_observability_end_to_end_with_stub_backends(all_pillars_registered: None) -> None:
    pillars = ["factual_accuracy", "relevance"]
    rows = _rows()

    mlflow_backend = FakeMLflowBackend()
    langfuse_backend = FakeLangfuseBackend()

    mlflow_logger = build_mlflow_logger(
        tracking_uri="http://localhost:5000",
        experiment_name="stage10",
        backend=mlflow_backend,
    )
    langfuse_tracer = build_langfuse_tracer(
        host="http://localhost",
        public_key="pk",
        secret_key="sk",
        redact_pii=False,
        backend=langfuse_backend,
    )

    # Pre-plan step so we can derive metadata; this mirrors the Run page.
    tmp_plan, llms = _build_plan(pillars, rows, on_outcome=None, on_progress=None)
    meta = build_run_metadata(
        run_id=tmp_plan.run_context.run_id,
        dataset_fingerprint=tmp_plan.run_context.dataset_fingerprint,
        dataset_row_count=len(rows),
        pillars=pillars,
        bundles=dict(tmp_plan.bundles),
        llm_by_pillar=llms,
        provider="mock",
        run_config={"pillars": pillars},
    )
    langfuse_tracer.start_run(meta)
    callbacks = build_observability_callbacks(rows=rows, tracer=langfuse_tracer)

    plan, _ = _build_plan(
        pillars, rows, on_outcome=callbacks.on_outcome, on_progress=callbacks.on_progress
    )

    with mlflow_logger.active_run(meta):
        result = EvaluationRunner().run(plan)
        joined = join_outcomes_with_labels(rows, result.outcomes)
        report = compute_agreement_report(joined.items, pillars=pillars)
        mlflow_logger.log_run_summary(result.summary)
        mlflow_logger.log_agreement_report(report)
        mlflow_logger.log_artifact_json("report.json", {"run_id": result.run_id})

    langfuse_tracer.end_run(
        status="FINISHED",
        summary={"succeeded": result.summary.succeeded, "failed": result.summary.failed},
    )

    # --- Runner side effects ------------------------------------------
    assert result.summary.total_tasks == len(rows) * len(pillars)
    assert result.summary.failed == 0

    # --- Langfuse side effects ----------------------------------------
    root_observations = [
        obs
        for obs in langfuse_backend.observations
        if obs.kwargs.get("name", "").startswith("jtb.run.")
    ]
    assert len(root_observations) == 1
    root = root_observations[0]
    # One generation per finished task, attached under the root span.
    assert len(root.children) == result.summary.total_tasks
    for child in root.children:
        assert child.kwargs["as_type"] == "generation"
        assert child.ended is True
    assert root.ended is True
    assert langfuse_backend.flushed >= 1

    # --- MLflow side effects ------------------------------------------
    names = mlflow_backend.names()
    assert names.count("start_run") == 1
    # Params include pillars + dataset fingerprint that match the runner's.
    params_call = next(kwargs for name, kwargs in mlflow_backend.calls if name == "log_params")
    params = params_call["params"]
    assert params["dataset_fingerprint"] == meta.dataset_fingerprint
    assert params["pillars"] == ",".join(pillars)

    # At least one metrics batch and one text artifact were pushed.
    assert any(name == "log_metrics" for name, _ in mlflow_backend.calls)
    artifact_files = [
        kwargs["artifact_file"] for name, kwargs in mlflow_backend.calls if name == "log_text"
    ]
    assert "report.json" in artifact_files

    # Run was closed cleanly.
    end_calls = [kwargs for name, kwargs in mlflow_backend.calls if name == "end_run"]
    assert end_calls and end_calls[-1]["status"] == "FINISHED"


def test_observability_disabled_when_no_credentials(all_pillars_registered: None) -> None:
    """Running with disabled loggers still produces a correct result.

    Confirms the acceptance criterion "graceful no-op tests when
    observability backends are disabled".
    """
    pillars = ["factual_accuracy"]
    rows = _rows()

    mlflow_logger = build_mlflow_logger(tracking_uri=None)  # disabled
    langfuse_tracer = build_langfuse_tracer(
        host=None, public_key=None, secret_key=None, redact_pii=False
    )
    assert mlflow_logger.enabled is False
    assert langfuse_tracer.enabled is False

    tmp_plan, llms = _build_plan(pillars, rows, on_outcome=None, on_progress=None)
    meta = build_run_metadata(
        run_id=tmp_plan.run_context.run_id,
        dataset_fingerprint=tmp_plan.run_context.dataset_fingerprint,
        dataset_row_count=len(rows),
        pillars=pillars,
        bundles=dict(tmp_plan.bundles),
        llm_by_pillar=llms,
        provider="mock",
    )

    langfuse_tracer.start_run(meta)
    callbacks = build_observability_callbacks(rows=rows, tracer=langfuse_tracer)
    plan, _ = _build_plan(
        pillars, rows, on_outcome=callbacks.on_outcome, on_progress=callbacks.on_progress
    )

    # Must not raise, must produce a result.
    with mlflow_logger.active_run(meta):
        result = EvaluationRunner().run(plan)
        mlflow_logger.log_run_summary(result.summary)
    langfuse_tracer.end_run("FINISHED")

    assert result.summary.total_tasks == len(rows) * len(pillars)
    # PILLARS is always non-empty so the assertion above is meaningful.
    assert PILLARS
