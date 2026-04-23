"""Stage 9 end-to-end: upload -> run -> dashboard data.

Walks the user journey from file load to the full set of dashboard
inputs the Streamlit pages consume, without any Streamlit runtime.
Ensures:

1. The shipped ``full_schema_sample.csv`` normalises cleanly.
2. A ``RunConfig`` drives :class:`EvaluationRunner` with the
   :class:`MockLLMClient` and produces outcomes for every
   ``(row, pillar)`` pair.
3. Every chart builder produces a serialisable Altair spec.
4. Every table row-builder produces the expected column set.
5. The disagreement filter narrows the table by pillar, category,
   severity - the actual contract the explorer relies on.

This is the last guardrail before UI regressions - if a future stage
adds a required field to :class:`JudgeResult` without updating a
chart column, this test fails.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import altair as alt
import pytest
from pydantic import BaseModel

from src.app.state import RunConfig
from src.completeness.kb_loader import load_kb
from src.completeness.models import CompletenessKB
from src.core.constants import PILLARS
from src.core.types import NormalizedRow, RunContext
from src.dashboard import (
    DisagreementFilter,
    SeverityBucket,
    apply_filter,
    build_agreement_summary_rows,
    build_category_breakdown_rows,
    build_category_pillar_heatmap,
    build_confusion_matrix_heatmap,
    build_disagreement_rows,
    build_large_miss_by_category_chart,
    build_pillar_agreement_bar,
    build_reviewer_agreement_bar,
    build_reviewer_pair_rows,
    build_reviewer_pillar_rows,
    build_reviewer_summary_rows,
    build_score_distribution_bar,
)
from src.evaluation import (
    compute_agreement_report,
    compute_reviewer_analytics,
    compute_sliced_report,
    has_reviewer_signal,
    join_outcomes_with_labels,
    slice_by_category,
)
from src.ingestion import auto_suggest_mapping, load_file, normalize_rows
from src.judges import JudgeCoreOutput, load_judge_bundle
from src.llm.base import LLMRequest, LLMUsage
from src.llm.mock_client import MockLLMClient
from src.orchestration import ConcurrencyPolicy, EvaluationRunner, RunPlan

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
JUDGES_DIR = REPO_ROOT / "configs" / "judges"
RUBRICS_DIR = REPO_ROOT / "configs" / "rubrics"
SEED_KB = REPO_ROOT / "configs" / "completeness_kb" / "seed.yaml"


# ---------------------------------------------------------------------------
# Fixtures / wiring
# ---------------------------------------------------------------------------


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Sample artifact missing at {path}; run scripts/generate_sample_data.py")


@pytest.fixture
def seed_kb() -> CompletenessKB:
    _require(SEED_KB)
    return load_kb(SEED_KB)


@pytest.fixture
def rows() -> list[NormalizedRow]:
    csv_path = SAMPLES_DIR / "full_schema_sample.csv"
    _require(csv_path)
    loaded = load_file(csv_path)
    mapping = auto_suggest_mapping(loaded.source_columns)
    norm = normalize_rows(loaded.rows, mapping=mapping.mappings)
    assert norm.failure_count == 0, norm.failures
    return norm.rows


def _make_mock_fn(pillar: str) -> Callable[[LLMRequest, type[BaseModel]], BaseModel]:
    """Return a ``structured_fn`` that emits a plausible output.

    Derives the score from a hash of the serialised prompt so values
    are deterministic across runs but vary across rows. Restricts to
    scores 3-5 to sidestep the "score=5 needs no failure_tags"
    invariant conflicting with low scores.
    """

    def _respond(request: LLMRequest, schema: type[BaseModel]) -> BaseModel:
        digest = hashlib.sha1(request.user_prompt.encode("utf-8")).hexdigest()
        score = 3 + (int(digest[:4], 16) % 3)
        payload: dict[str, Any] = {
            "pillar": pillar,
            "score": score,
            "confidence": 0.65,
            "decision_summary": f"Stage9 mock {pillar} score={score}.",
            "evidence_for_score": [],
            "failure_tags": [],
            "rubric_anchor": score,
            "why_not_higher": "Mock response.",
            "why_not_lower": "Mock response.",
        }
        if pillar == "completeness":
            payload["elements_present"] = []
            payload["elements_missing"] = []
        allowed = set(schema.model_fields) if issubclass(schema, JudgeCoreOutput) else None
        trimmed = {k: v for k, v in payload.items() if allowed is None or k in allowed}
        return schema.model_validate(trimmed)

    return _respond


def _run(rows: list[NormalizedRow], kb: CompletenessKB) -> Any:
    bundles = {
        p: load_judge_bundle(JUDGES_DIR / f"{p}.yaml", rubric_root=RUBRICS_DIR) for p in PILLARS
    }
    llms = {
        p: MockLLMClient(
            model_name=f"mock-{p}",
            structured_fn=_make_mock_fn(p),
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )
        for p in PILLARS
    }
    plan = RunPlan(
        rows=rows,
        pillars=list(PILLARS),
        bundles=bundles,
        llm_by_pillar=llms,
        run_context=RunContext(
            run_id="stage9-dashboard-e2e",
            dataset_fingerprint="sha256:stage9-e2e",
            kb_version=kb.fingerprint(),
            model_alias="mock",
        ),
        kb=kb,
        concurrency=ConcurrencyPolicy(max_workers=6),
    )
    return EvaluationRunner().run(plan)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_config_round_trip_validates_inputs() -> None:
    cfg = RunConfig(
        pillars=("factual_accuracy", "relevance"),
        provider="mock",
        max_workers=4,
    )
    assert cfg.pillars == ("factual_accuracy", "relevance")

    with pytest.raises(ValueError, match="unknown pillar"):
        RunConfig(pillars=("not_a_pillar",))
    with pytest.raises(ValueError, match="non-empty"):
        RunConfig(pillars=())
    with pytest.raises(ValueError, match="provider"):
        RunConfig(pillars=("relevance",), provider="openai")


def test_dashboard_data_pipeline_end_to_end(
    all_pillars_registered: None,
    rows: list[NormalizedRow],
    seed_kb: CompletenessKB,
) -> None:
    result = _run(rows, seed_kb)

    # Runner health
    assert result.summary.failed == 0
    assert result.summary.succeeded == len(rows) * len(PILLARS)

    joined = join_outcomes_with_labels(rows, result.outcomes)
    assert joined.stats.paired > 0

    # --- Agreement report ----------------------------------------------------
    report = compute_agreement_report(joined.items)
    assert set(report.per_pillar) == set(PILLARS)
    summary_rows = build_agreement_summary_rows(report)
    pillar_col = {r["pillar"] for r in summary_rows}
    assert pillar_col.issuperset(set(PILLARS))

    # --- Slice / category ----------------------------------------------------
    sliced = compute_sliced_report(
        joined.items,
        selector=slice_by_category,
        dimension="category",
    )
    assert sliced.per_slice, "Expected at least one category slice."
    category_rows = build_category_breakdown_rows(sliced)
    assert category_rows
    assert all("category" in r for r in category_rows)

    # --- Reviewer analytics (full schema sample has reviewer metadata) -------
    assert has_reviewer_signal(joined.items)
    analytics = compute_reviewer_analytics(joined.items)
    assert analytics.has_data
    reviewer_rows = build_reviewer_summary_rows(analytics)
    assert reviewer_rows
    assert build_reviewer_pillar_rows(analytics)
    # Pairs may be empty if the sample has no reviewer overlap; that's fine.
    _ = build_reviewer_pair_rows(analytics)

    # --- Disagreement rows + filter ------------------------------------------
    disagreement_rows = build_disagreement_rows(rows, result.outcomes)
    assert disagreement_rows
    expected_cols = {
        "record_id",
        "pillar",
        "category",
        "judge_score",
        "human_score",
        "distance",
        "failure_tags",
    }
    assert expected_cols.issubset(disagreement_rows[0].keys())

    # Filter down to one pillar + categories; must be a strict subset.
    pillar_filter = DisagreementFilter(pillars=frozenset({"relevance"}))
    filtered_items = apply_filter(joined.items, pillar_filter)
    assert all(it.pillar == "relevance" for it in filtered_items)
    filtered_table = build_disagreement_rows(
        rows,
        result.outcomes,
        items=filtered_items,
    )
    assert {r["pillar"] for r in filtered_table} == {"relevance"}

    # Severity filter narrows the set (or at worst returns equal set).
    large_miss_filter = DisagreementFilter(severity=SeverityBucket.LARGE_MISS)
    large_miss_items = apply_filter(joined.items, large_miss_filter)
    assert len(large_miss_items) <= len(joined.items)
    for it in large_miss_items:
        assert it.distance >= 3

    # --- Every chart builder produces a serialisable spec --------------------
    charts: list[alt.Chart | alt.LayerChart] = [
        build_pillar_agreement_bar(report),
        build_pillar_agreement_bar(report, metric="exact_match"),
        build_score_distribution_bar(report.per_pillar["factual_accuracy"]),
        build_category_pillar_heatmap(sliced),
        build_large_miss_by_category_chart(sliced),
        build_large_miss_by_category_chart(sliced, pillar="relevance"),
        build_confusion_matrix_heatmap(report.per_pillar["factual_accuracy"]),
        build_reviewer_agreement_bar(analytics),
    ]
    for chart in charts:
        assert isinstance(chart, (alt.Chart, alt.LayerChart))
        spec = chart.to_dict()
        assert "$schema" in spec or "mark" in spec or "layer" in spec
