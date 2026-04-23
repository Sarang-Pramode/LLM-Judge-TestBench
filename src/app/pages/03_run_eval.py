"""Streamlit Run page.

Executes the configured evaluation run. Gates on:

- A normalized dataset in session state (from Upload).
- A :class:`RunConfig` in session state (from Configure).
- A provider we can actually build a client for (``mock`` always
  works; ``google`` requires an API key).

Implementation rules:

- Long work goes through :class:`EvaluationRunner`. The UI never
  touches judges, provider SDKs, or metrics directly.
- We surface progress via ``st.progress`` and ``st.status`` by
  handing the runner an ``on_progress`` callback. Worker threads
  must attach the main thread's :class:`~streamlit.runtime.scriptrunner_utils.script_run_context.ScriptRunContext`
  via :func:`~streamlit.runtime.scriptrunner_utils.script_run_context.add_script_run_ctx`
  before calling ``st.progress`` — otherwise Streamlit raises
  ``NoSessionContext``. A lock still serialises updates to avoid
  flicker when many workers finish in the same tick.
- Stores the :class:`RunResult` plus the rows used (pinned
  snapshot) so the dashboard keeps working if the user re-uploads
  a different file without running again.
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import streamlit as st
from pydantic import BaseModel
from streamlit.runtime.scriptrunner_utils.script_run_context import (
    add_script_run_ctx,
    get_script_run_ctx,
)

from src.app.state import (
    SS_BASELINE_SNAPSHOT,
    SS_DATASET_NAME,
    SS_LAST_RUN_RESULT,
    SS_LAST_RUN_ROWS,
    SS_NORMALIZED_ROWS,
    SS_RUN_CONFIG,
    RunConfig,
)
from src.completeness.kb_loader import load_kb
from src.completeness.models import CompletenessKB
from src.core.constants import PILLARS
from src.core.exceptions import ConfigLoadError, ProviderError
from src.core.settings import get_settings
from src.core.types import NormalizedRow, RunContext
from src.dashboard.plotly_charts import combined_risk_evidence_html
from src.evaluation.agreement import compute_agreement_report
from src.evaluation.diagnostics import BaselineSnapshot, compute_run_diagnostics
from src.evaluation.join import join_outcomes_with_labels
from src.evaluation.reviewer_analysis import compute_reviewer_analytics, has_reviewer_signal
from src.evaluation.slices import compute_sliced_report, slice_by_category
from src.evaluation.thresholds import (
    evaluate_agreement_against_thresholds,
    load_evaluation_thresholds,
)
from src.judges import load_judge_bundle
from src.judges.base import JudgeCoreOutput
from src.llm.base import LLMClient, LLMRequest, LLMUsage
from src.llm.factory import build_client
from src.llm.mock_client import MockLLMClient
from src.observability import (
    LangfuseTracer,
    MLflowLogger,
    build_langfuse_tracer,
    build_mlflow_logger,
    build_observability_callbacks,
    build_run_metadata,
    dataset_fingerprint,
    run_config_hash,
)
from src.observability.mlflow_risk_logging import (
    log_diagnostics_mlflow,
    log_plotly_html_mlflow,
    log_threshold_report_mlflow,
)
from src.orchestration import ConcurrencyPolicy, EvaluationRunner, RunPlan
from src.orchestration.caching import InMemoryOutcomeCache, NoCache

REPO_ROOT = Path(__file__).resolve().parents[3]
JUDGES_DIR = REPO_ROOT / "configs" / "judges"
RUBRICS_DIR = REPO_ROOT / "configs" / "rubrics"
DEFAULT_KB = REPO_ROOT / "configs" / "completeness_kb" / "seed.yaml"
THRESHOLDS_YAML = REPO_ROOT / "configs" / "evaluation_thresholds.yaml"


def render() -> None:
    st.set_page_config(page_title="Run evaluation", layout="wide")
    st.title("Run evaluation")
    st.caption(
        "Executes the configured pillars against the uploaded dataset. "
        "Partial failures are reported; the run never crashes the UI."
    )

    rows: list[NormalizedRow] | None = st.session_state.get(SS_NORMALIZED_ROWS)
    run_config: RunConfig | None = st.session_state.get(SS_RUN_CONFIG)

    if not rows:
        st.warning("Upload a dataset first (Upload page).")
        return
    if run_config is None:
        st.warning("Configure the run first (Configure page).")
        return

    _render_preflight(rows, run_config)

    if st.button("Start run", type="primary"):
        _run(rows, run_config)

    last_result = st.session_state.get(SS_LAST_RUN_RESULT)
    if last_result is not None:
        _render_last_run_summary(last_result)


# ---------------------------------------------------------------------------
# Preflight summary
# ---------------------------------------------------------------------------


def _render_preflight(rows: list[NormalizedRow], run_config: RunConfig) -> None:
    cols = st.columns(4)
    cols[0].metric("Rows", len(rows))
    cols[1].metric("Pillars", len(run_config.pillars))
    cols[2].metric("Tasks", len(rows) * len(run_config.pillars))
    cols[3].metric("Provider", run_config.provider)

    labelled = sum(
        1 for row in rows if any(getattr(row, f"label_{p}", None) is not None for p in PILLARS)
    )
    st.caption(
        f"{labelled} / {len(rows)} rows carry at least one SME label. "
        "Rows without labels still get judge scores, but they won't "
        "contribute to agreement metrics."
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _run(rows: list[NormalizedRow], run_config: RunConfig) -> None:
    dataset_name = st.session_state.get(SS_DATASET_NAME, "dataset")
    try:
        bundles = {
            pillar: load_judge_bundle(JUDGES_DIR / f"{pillar}.yaml", rubric_root=RUBRICS_DIR)
            for pillar in run_config.pillars
        }
    except Exception as exc:
        st.error(f"Failed to load judge bundles: {exc}")
        return

    try:
        kb = _load_kb_if_used(run_config)
    except Exception as exc:
        st.warning(
            f"Completeness KB could not be loaded ({exc}); the "
            "completeness judge will run in generic fallback mode."
        )
        kb = None

    try:
        llms = _build_llm_clients(run_config)
    except ProviderError as exc:
        st.error(str(exc))
        return

    run_context = _build_run_context(rows, run_config, kb)
    run_metadata = build_run_metadata(
        run_id=run_context.run_id,
        dataset_fingerprint=run_context.dataset_fingerprint,
        dataset_row_count=len(rows),
        pillars=run_config.pillars,
        bundles=bundles,
        llm_by_pillar=llms,
        provider=run_config.provider,
        run_config=_run_config_for_hash(run_config),
        kb_version=kb.fingerprint() if kb is not None else None,
        extra_tags={"dataset_name": dataset_name},
    )

    mlflow_logger = build_mlflow_logger(enabled=run_config.enable_mlflow)
    langfuse_tracer = build_langfuse_tracer(enabled=run_config.enable_langfuse)

    _render_observability_status(mlflow_logger, langfuse_tracer)

    cache = InMemoryOutcomeCache() if run_config.use_cache else NoCache()
    concurrency = ConcurrencyPolicy(
        max_workers=run_config.max_workers,
        per_provider_limit=run_config.per_provider_limit,
    )

    total_tasks = len(rows) * len(run_config.pillars)
    progress_bar = st.progress(0.0, text=f"Running {total_tasks} tasks...")
    progress_lock = threading.Lock()
    # EvaluationRunner workers run on thread-pool threads; Streamlit UI
    # calls require the parent ScriptRunContext on the current thread.
    _streamlit_ctx = get_script_run_ctx(suppress_warning=True)

    def _on_progress(done: int, total: int) -> None:
        # Worker threads call this; attach session context, then serialise
        # widget updates to avoid flicker when many workers finish in one tick.
        if _streamlit_ctx is not None:
            add_script_run_ctx(thread=threading.current_thread(), ctx=_streamlit_ctx)
        with progress_lock:
            if _streamlit_ctx is not None:
                progress_bar.progress(done / total, text=f"{done} / {total} tasks complete")

    langfuse_tracer.start_run(run_metadata)
    callbacks = build_observability_callbacks(
        rows=rows,
        tracer=langfuse_tracer,
        progress_cb=_on_progress,
    )

    plan = RunPlan(
        rows=rows,
        pillars=list(run_config.pillars),
        bundles=bundles,
        llm_by_pillar=llms,
        run_context=run_context,
        kb=kb,
        concurrency=concurrency,
        cache=cache,
        on_outcome=callbacks.on_outcome,
        on_progress=callbacks.on_progress,
    )

    with st.status("Executing run...", expanded=True) as status:
        try:
            with mlflow_logger.active_run(run_metadata):
                result = EvaluationRunner().run(plan)
                pillars_seq = list(run_config.pillars)
                joined = join_outcomes_with_labels(rows, result.outcomes, pillars=pillars_seq)
                report = compute_agreement_report(
                    joined.items, pillars=pillars_seq, include_overall=True
                )
                mlflow_logger.log_agreement_report(report)
                sliced = compute_sliced_report(
                    joined.items,
                    selector=slice_by_category,
                    dimension="category",
                    pillars=pillars_seq,
                    include_overall_per_slice=False,
                )
                mlflow_logger.log_slice_report(sliced)
                if has_reviewer_signal(joined.items):
                    mlflow_logger.log_reviewer_analytics(
                        compute_reviewer_analytics(joined.items, pillars=pillars_seq)
                    )
                raw_base = st.session_state.get(SS_BASELINE_SNAPSHOT)
                baseline_obj: BaselineSnapshot | None = None
                if isinstance(raw_base, dict):
                    try:
                        baseline_obj = BaselineSnapshot.from_serializable(raw_base)
                    except (KeyError, TypeError, ValueError):
                        baseline_obj = None
                diag = compute_run_diagnostics(
                    joined.items,
                    pillars=pillars_seq,
                    dataset_fingerprint=run_context.dataset_fingerprint,
                    baseline=baseline_obj,
                )
                log_diagnostics_mlflow(mlflow_logger, diag)
                try:
                    cfg_thr = load_evaluation_thresholds(THRESHOLDS_YAML)
                except ConfigLoadError:
                    cfg_thr = None
                if cfg_thr is not None:
                    log_threshold_report_mlflow(
                        mlflow_logger,
                        evaluate_agreement_against_thresholds(report, cfg_thr),
                    )
                try:
                    html_bundle = combined_risk_evidence_html(
                        joined.items,
                        diag,
                        baseline_pmfs=(baseline_obj.judge_pmfs if baseline_obj else None),
                    )
                    log_plotly_html_mlflow(mlflow_logger, html_bundle)
                except Exception:
                    pass
                mlflow_logger.log_run_result(result)
        except Exception as exc:
            status.update(label="Run failed", state="error")
            langfuse_tracer.end_run(status="FAILED", summary={"error": str(exc)})
            st.exception(exc)
            return
        langfuse_tracer.end_run(
            status="FINISHED",
            summary={
                "succeeded": result.summary.succeeded,
                "failed": result.summary.failed,
            },
        )
        status.update(label=f"Run complete: {result.run_id}", state="complete")

    st.session_state[SS_LAST_RUN_RESULT] = result
    st.session_state[SS_LAST_RUN_ROWS] = list(rows)
    st.success(
        f"Run {result.run_id} finished in {result.summary.duration_s:.2f}s. "
        f"{result.summary.succeeded}/{result.summary.total_tasks} tasks succeeded."
    )
    st.info("Open **Dashboard** to explore metrics, and **Disagreements** to triage.")


# ---------------------------------------------------------------------------
# LLM client wiring
# ---------------------------------------------------------------------------


def _build_llm_clients(run_config: RunConfig) -> dict[str, LLMClient]:
    """Return one :class:`LLMClient` per pillar.

    For ``mock``: we hand every pillar a dedicated mock keyed by
    pillar name, driven by a label-aware structured function so the
    demo dashboards have realistic-looking agreement.

    For ``google``: we use the factory's ``judge-default`` alias.
    All pillars share the same underlying client (throttling is
    per-client, so the runner's ``per_provider_limit`` actually
    constrains total Gemini QPS).
    """
    if run_config.provider == "mock":
        return {pillar: _build_mock(pillar) for pillar in run_config.pillars}
    if run_config.provider == "google":
        settings = get_settings()
        if settings.google_api_key is None:
            raise ProviderError("Google provider selected but JTB_GOOGLE_API_KEY is not set.")
        client = build_client("judge-default", settings=settings)
        return {pillar: client for pillar in run_config.pillars}
    raise ProviderError(f"Unsupported provider {run_config.provider!r}.")


def _build_mock(pillar: str) -> MockLLMClient:
    """Mock client that returns a plausible judge output for demos.

    Strategy: we cannot read the SME label here (the client only
    sees the rendered prompt), so we produce a centred, low-variance
    score of 4 with a tiny deterministic pillar-salted offset. It's
    enough to exercise every dashboard and table code path without
    a real model.
    """

    def _respond(
        request: LLMRequest,
        schema: type[BaseModel],
    ) -> BaseModel:
        # Hash the serialised user message to obtain a reproducible
        # score in {2, 3, 4, 5}. Avoids score=1 so failure-tag
        # rubrics still validate without a failure tag.
        digest = hashlib.sha1(request.user_prompt.encode("utf-8")).hexdigest()
        score = 2 + (int(digest[:4], 16) % 4)
        payload: dict[str, Any] = {
            "pillar": pillar,
            "score": score,
            "confidence": 0.6,
            "decision_summary": f"Mock {pillar} score {score}.",
            "evidence_for_score": [],
            "failure_tags": [],
            "rubric_anchor": score,
            "why_not_higher": "Mock client - no real rationale.",
            "why_not_lower": "Mock client - no real rationale.",
        }
        if pillar == "completeness":
            payload["elements_present"] = []
            payload["elements_missing"] = []
        # ``_coerce_to_schema`` will upgrade a dict -> schema instance.
        return schema.model_validate(_trim_to_schema(payload, schema))

    return MockLLMClient(
        model_name=f"mock-{pillar}",
        structured_fn=_respond,
        usage=LLMUsage(input_tokens=10, output_tokens=5),
    )


def _trim_to_schema(payload: dict[str, Any], schema: type[BaseModel]) -> dict[str, Any]:
    """Drop keys the schema doesn't accept.

    Pillar-specific judge outputs (e.g. :class:`CompletenessCoreOutput`)
    extend :class:`JudgeCoreOutput` with extra fields. The generic
    mock payload carries completeness-only keys too; trim them when
    the target schema rejects them.
    """
    if not issubclass(schema, JudgeCoreOutput):
        return payload
    allowed = set(schema.model_fields)
    return {k: v for k, v in payload.items() if k in allowed}


# ---------------------------------------------------------------------------
# KB + run context
# ---------------------------------------------------------------------------


def _load_kb_if_used(run_config: RunConfig) -> CompletenessKB | None:
    if "completeness" not in run_config.pillars:
        return None
    if not DEFAULT_KB.exists():
        return None
    return load_kb(DEFAULT_KB)


def _build_run_context(
    rows: list[NormalizedRow],
    run_config: RunConfig,
    kb: CompletenessKB | None,
) -> RunContext:
    """Content-based fingerprint + run-config hash, wrapped in a RunContext.

    The legacy name-based fingerprint was replaced by the content-based
    :func:`dataset_fingerprint` so MLflow / Langfuse can dedupe runs on
    identical data regardless of upload filename.
    """
    fingerprint = dataset_fingerprint(rows)
    config_hash = run_config_hash(_run_config_for_hash(run_config))
    return RunContext(
        run_id=f"jtb-{uuid.uuid4().hex[:8]}",
        dataset_fingerprint=fingerprint,
        kb_version=kb.fingerprint() if kb is not None else None,
        model_alias=run_config.provider,
        run_config_hash=config_hash,
    )


def _run_config_for_hash(run_config: RunConfig) -> dict[str, Any]:
    """Canonicalize a :class:`RunConfig` for fingerprinting.

    We deliberately exclude the observability toggles: they don't change
    what the judges do, and excluding them keeps the hash stable across
    "same data + pillars + provider" reruns even if the user flips
    tracing on or off.
    """
    data = asdict(run_config)
    data.pop("enable_mlflow", None)
    data.pop("enable_langfuse", None)
    return data


def _render_observability_status(
    mlflow_logger: MLflowLogger,
    langfuse_tracer: LangfuseTracer,
) -> None:
    """Small badge row showing whether loggers actually reach their backends."""
    col_a, col_b = st.columns(2)
    col_a.metric(
        "MLflow",
        "enabled" if mlflow_logger.enabled else "disabled",
        help=(
            "Aggregate run metrics. Disabled when tracking URI is missing "
            "or the MLflow library is unavailable."
        ),
    )
    col_b.metric(
        "Langfuse",
        "enabled" if langfuse_tracer.enabled else "disabled",
        help=(
            "Per-row traces. Disabled when credentials are missing or "
            "the Langfuse library is unavailable."
        ),
    )


# ---------------------------------------------------------------------------
# Last-run summary panel
# ---------------------------------------------------------------------------


def _render_last_run_summary(result: Any) -> None:
    st.divider()
    st.subheader("Last run summary")
    summary = result.summary
    cols = st.columns(5)
    cols[0].metric("Tasks", summary.total_tasks)
    cols[1].metric("Succeeded", summary.succeeded)
    cols[2].metric("Failed", summary.failed)
    cols[3].metric("Cache hits", summary.cache_hits)
    cols[4].metric("Duration (s)", f"{summary.duration_s:.2f}")

    cols = st.columns(4)
    cols[0].metric("p50 latency (ms)", f"{summary.latency_ms_p50:.1f}")
    cols[1].metric("p95 latency (ms)", f"{summary.latency_ms_p95:.1f}")
    cols[2].metric("Input tokens", summary.total_input_tokens)
    cols[3].metric("Output tokens", summary.total_output_tokens)

    if summary.pillar_stats:
        st.write("Per-pillar outcome counts:")
        st.dataframe(
            [
                {"pillar": p, "succeeded": s, "failed": f}
                for p, (s, f) in summary.pillar_stats.items()
            ],
            use_container_width=True,
        )


render()
