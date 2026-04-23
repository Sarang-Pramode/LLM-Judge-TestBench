"""Stage 8 end-to-end: runner -> join -> metrics / slices / reviewer.

This integration test exercises the full post-run path:

1. Load the shipped ``full_schema_sample.csv`` (all labels populated).
2. Drive every pillar through :class:`EvaluationRunner` with the
   :class:`MockLLMClient`, scripting *intentional* off-by-one and
   off-by-two judge scores so metrics produce non-trivial values.
3. Join outcomes to SME labels and verify:
   - :func:`compute_agreement_report` yields the expected per-pillar
     support + headline metrics.
   - :func:`compute_sliced_report` by category produces one bucket
     per category present in the dataset.
   - :func:`compute_reviewer_analytics` activates because the sample
     has reviewer metadata.

The point is to catch any integration drift between the runner
outputs and the metric layer (e.g. field renames, changed JudgeResult
invariants) with a realistic dataset - unit tests alone would miss
that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.completeness.kb_loader import load_kb
from src.completeness.models import CompletenessKB
from src.core.constants import PILLARS
from src.core.types import NormalizedRow, RunContext
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
from src.llm.base import LLMUsage
from src.llm.mock_client import MockLLMClient
from src.orchestration import ConcurrencyPolicy, EvaluationRunner, RunPlan
from src.rubrics.loader import load_rubric
from src.rubrics.models import Rubric

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
JUDGES_DIR = REPO_ROOT / "configs" / "judges"
RUBRICS_DIR = REPO_ROOT / "configs" / "rubrics"
SEED_KB = REPO_ROOT / "configs" / "completeness_kb" / "seed.yaml"


# ---------------------------------------------------------------------------
# Fixtures / loaders
# ---------------------------------------------------------------------------


@pytest.fixture
def seed_kb() -> CompletenessKB:
    if not SEED_KB.exists():
        pytest.skip(f"Seed KB missing at {SEED_KB}")
    return load_kb(SEED_KB)


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Sample artifact missing at {path}; run scripts/generate_sample_data.py")


def _load_rows() -> list[NormalizedRow]:
    path = SAMPLES_DIR / "full_schema_sample.csv"
    _require(path)
    loaded = load_file(path)
    suggested = auto_suggest_mapping(loaded.source_columns)
    norm = normalize_rows(loaded.rows, mapping=suggested.mappings)
    assert norm.failure_count == 0, norm.failures
    return norm.rows


def _scripted_score(row_idx: int, pillar: str, label: int | None) -> int:
    """Return a judge score with a deterministic offset from ``label``.

    The offset pattern is chosen to populate every distance bucket so
    metrics produce non-zero values across the run:

    - Row 0: exact match on every pillar (distance 0).
    - Row 1: judge scores one below the label (distance 1), clamped.
    - Row 2: judge scores two below (distance 2), clamped.
    - Row 3: judge scores far below (distance 3+), clamped to [1, 5].
    """
    base = label if label is not None else 3
    offsets = [0, 1, 2, 3]
    offset = offsets[row_idx % len(offsets)]
    raw = base - offset
    return max(1, min(5, raw))


def _build_output(
    pillar: str,
    rubric: Rubric,
    row_idx: int,
    row: NormalizedRow,
    kb: CompletenessKB,
) -> JudgeCoreOutput:
    label = getattr(row, f"label_{pillar}")
    score = _scripted_score(row_idx, pillar, label)
    payload: dict[str, Any] = {
        "pillar": pillar,
        "score": score,
        "confidence": 0.7,
        "decision_summary": f"Scripted {pillar} score={score} (row {row_idx}).",
        "evidence_for_score": [],
        "failure_tags": [rubric.failure_tags[0]] if score < 5 and rubric.failure_tags else [],
        "rubric_anchor": score,
    }
    if score < 5:
        payload["why_not_higher"] = "Deliberate scripted disagreement."
    if score > 1:
        payload["why_not_lower"] = "Not bottomed out."
    if pillar == "completeness":
        entry = next((e for e in kb.entries if e.intent == row.intent), None)
        if entry is not None and entry.required_elements:
            payload["elements_present"] = entry.required_elements[:1]
            payload["elements_missing"] = entry.required_elements[1:]
    return JudgeCoreOutput(**payload)


def _build_scripts(
    rows: list[NormalizedRow],
    rubrics: dict[str, Rubric],
    kb: CompletenessKB,
) -> dict[str, list[JudgeCoreOutput]]:
    return {
        pillar: [
            _build_output(pillar, rubrics[pillar], idx, row, kb) for idx, row in enumerate(rows)
        ]
        for pillar in PILLARS
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _run(rows: list[NormalizedRow], kb: CompletenessKB) -> Any:
    rubrics = {p: load_rubric(RUBRICS_DIR / f"{p}.yaml") for p in PILLARS}
    bundles = {
        p: load_judge_bundle(JUDGES_DIR / f"{p}.yaml", rubric_root=RUBRICS_DIR) for p in PILLARS
    }
    scripts = _build_scripts(rows, rubrics, kb)
    llms = {
        p: MockLLMClient(
            model_name=f"mock-{p}",
            structured_script=scripts[p],
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
            run_id="stage8-metrics-e2e",
            dataset_fingerprint="sha256:full-schema-sample-v1",
            kb_version=kb.fingerprint(),
            model_alias="mock-judge",
        ),
        kb=kb,
        concurrency=ConcurrencyPolicy(max_workers=6),
    )
    return EvaluationRunner().run(plan)


def test_agreement_report_after_runner_has_expected_shape(
    all_pillars_registered: None, seed_kb: CompletenessKB
) -> None:
    rows = _load_rows()
    result = _run(rows, seed_kb)
    assert result.summary.failed == 0

    joined = join_outcomes_with_labels(rows, result.outcomes)
    # Every (row, pillar) pair had both a label and a successful outcome.
    assert joined.stats.paired == len(rows) * len(PILLARS)
    assert joined.stats.missing_labels == 0
    assert joined.stats.failed_outcomes == 0

    report = compute_agreement_report(joined.items)
    # Every pillar should have support == number of labelled rows.
    labelled_rows = sum(1 for row in rows if row.label_factual_accuracy is not None)
    for pillar in PILLARS:
        pa = report.per_pillar[pillar]
        assert pa.support == labelled_rows
        # Scripted offsets include 0,1,2,3 so exact match >= 1/rows and
        # MAE > 0 once non-zero offsets exist.
        assert pa.exact_match_rate >= 1 / labelled_rows
        assert pa.mean_absolute_error > 0.0

    assert report.overall is not None
    assert report.overall.support == len(rows) * len(PILLARS)


def test_sliced_report_by_category_produces_per_category_metrics(
    all_pillars_registered: None, seed_kb: CompletenessKB
) -> None:
    rows = _load_rows()
    result = _run(rows, seed_kb)
    joined = join_outcomes_with_labels(rows, result.outcomes)

    sliced = compute_sliced_report(
        joined.items,
        selector=slice_by_category,
        dimension="category",
    )
    categories_present = {row.category for row in rows}
    assert set(sliced.per_slice) == categories_present
    for cat in categories_present:
        # Each bucket has at least one pillar with support > 0.
        assert any(
            pa.support > 0 for pa in sliced.per_slice[cat].per_pillar.values()
        ), f"category {cat!r} should have at least one populated pillar"


def test_reviewer_analytics_activates_on_sample(
    all_pillars_registered: None, seed_kb: CompletenessKB
) -> None:
    rows = _load_rows()
    result = _run(rows, seed_kb)
    joined = join_outcomes_with_labels(rows, result.outcomes)

    assert has_reviewer_signal(joined.items)
    analytics = compute_reviewer_analytics(joined.items)
    assert analytics.has_data
    # All sample rows carry reviewer_name -> every reviewer in the
    # dataset is represented here.
    sample_reviewers = {r.reviewer_name for r in rows if r.reviewer_name}
    assert set(analytics.per_reviewer) == sample_reviewers
    # Each reviewer has non-zero sample_count and per-pillar stats.
    for reviewer in analytics.per_reviewer.values():
        assert reviewer.sample_count > 0
        assert set(reviewer.per_pillar) == set(PILLARS)
