"""Row-builder tests for the Stage 9 dashboard.

The Streamlit pages trust these helpers to return JSON-safe dict
lists with stable column sets. We unit-test:

- Agreement summary rows include every pillar + an optional overall.
- Category breakdown rows respect the ``pillar`` filter.
- Reviewer summary + per-pillar rows emit one row per reviewer /
  (reviewer, pillar) pair respectively.
- Disagreement rows populate the full column set from both rows and
  outcomes, and honour the optional filtered-items restriction.
"""

from __future__ import annotations

from typing import Any

from src.core.types import JudgeResult, NormalizedRow
from src.dashboard.tables import (
    build_agreement_summary_rows,
    build_category_breakdown_rows,
    build_disagreement_rows,
    build_reviewer_pair_rows,
    build_reviewer_pillar_rows,
    build_reviewer_summary_rows,
)
from src.evaluation.agreement import compute_agreement_report
from src.evaluation.join import ScoredItem
from src.evaluation.reviewer_analysis import compute_reviewer_analytics
from src.evaluation.slices import compute_sliced_report, slice_by_category
from src.judges.base import JudgeOutcome
from src.llm.base import LLMUsage

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _item(
    record_id: str,
    pillar: str,
    judge: int,
    human: int,
    *,
    category: str = "billing",
    reviewer_name: str | None = None,
) -> ScoredItem:
    return ScoredItem(
        record_id=record_id,
        pillar=pillar,
        judge_score=judge,
        human_score=human,
        category=category,
        reviewer_name=reviewer_name,
    )


def _result(pillar: str, score: int) -> JudgeResult:
    kwargs: dict[str, Any] = {
        "pillar": pillar,
        "score": score,
        "confidence": 0.7,
        "decision_summary": f"ok {score}",
        "rubric_anchor": score,
        "raw_model_name": "mock",
        "prompt_version": "p1",
        "rubric_version": "r1",
    }
    if score < 5:
        kwargs["why_not_higher"] = "room to grow"
        kwargs["failure_tags"] = ["tag"]
    if score > 1:
        kwargs["why_not_lower"] = "not worst"
    return JudgeResult(**kwargs)


def _outcome(
    pillar: str,
    record_id: str,
    *,
    score: int,
    latency_ms: float = 1.0,
) -> JudgeOutcome:
    return JudgeOutcome(
        pillar=pillar,
        record_id=record_id,
        latency_ms=latency_ms,
        attempts=1,
        usage=LLMUsage(),
        model_name="mock-v0",
        run_id="run-test",
        result=_result(pillar, score),
    )


def _row(
    record_id: str,
    *,
    category: str = "billing",
    reviewer_name: str | None = None,
    labels: dict[str, int] | None = None,
) -> NormalizedRow:
    kwargs: dict[str, Any] = {
        "record_id": record_id,
        "user_input": "user",
        "agent_output": "agent",
        "category": category,
        "reviewer_name": reviewer_name,
    }
    for pillar, value in (labels or {}).items():
        kwargs[f"label_{pillar}"] = value
    return NormalizedRow(**kwargs)


# ---------------------------------------------------------------------------
# Agreement summary
# ---------------------------------------------------------------------------


class TestAgreementSummaryRows:
    def test_one_row_per_pillar_plus_overall(self) -> None:
        items = [
            _item("r1", "factual_accuracy", 5, 5),
            _item("r1", "relevance", 4, 5),
            _item("r2", "relevance", 3, 5),
        ]
        report = compute_agreement_report(items, pillars=["factual_accuracy", "relevance"])
        rows = build_agreement_summary_rows(report)
        pillars = [r["pillar"] for r in rows]
        assert pillars[-1] == "__overall__"
        assert "factual_accuracy" in pillars and "relevance" in pillars

    def test_rates_rendered_as_percentages(self) -> None:
        items = [
            _item("r1", "factual_accuracy", 5, 5),
            _item("r2", "factual_accuracy", 4, 5),
        ]
        report = compute_agreement_report(items, pillars=["factual_accuracy"])
        rows = build_agreement_summary_rows(report, include_overall=False)
        fact = rows[0]
        # 1 exact match out of 2 pairs -> 50.00 %
        assert fact["exact_match_%"] == 50.00
        assert fact["within_1_%"] == 100.00

    def test_include_overall_toggle(self) -> None:
        items = [_item("r1", "factual_accuracy", 5, 5)]
        report = compute_agreement_report(items, pillars=["factual_accuracy"])
        rows = build_agreement_summary_rows(report, include_overall=False)
        assert all(r["pillar"] != "__overall__" for r in rows)


# ---------------------------------------------------------------------------
# Category breakdown
# ---------------------------------------------------------------------------


class TestCategoryBreakdownRows:
    def test_one_row_per_slice_pillar(self) -> None:
        items = [
            _item("r1", "factual_accuracy", 5, 5, category="billing"),
            _item("r2", "factual_accuracy", 4, 5, category="support"),
            _item("r1", "relevance", 3, 5, category="billing"),
        ]
        sliced = compute_sliced_report(
            items,
            selector=slice_by_category,
            dimension="category",
            pillars=["factual_accuracy", "relevance"],
            include_overall_per_slice=False,
        )
        rows = build_category_breakdown_rows(sliced)
        # 2 slices x 2 pillars = 4 rows
        assert len(rows) == 4
        for row in rows:
            assert "category" in row
            assert row["slice_total_items"] >= 0

    def test_pillar_filter(self) -> None:
        items = [
            _item("r1", "factual_accuracy", 5, 5, category="billing"),
            _item("r1", "relevance", 3, 5, category="billing"),
        ]
        sliced = compute_sliced_report(
            items,
            selector=slice_by_category,
            dimension="category",
            pillars=["factual_accuracy", "relevance"],
            include_overall_per_slice=False,
        )
        rows = build_category_breakdown_rows(sliced, pillar="relevance")
        assert [r["pillar"] for r in rows] == ["relevance"]


# ---------------------------------------------------------------------------
# Reviewer tables
# ---------------------------------------------------------------------------


class TestReviewerTables:
    def _reviewer_items(self) -> list[ScoredItem]:
        return [
            _item("r1", "factual_accuracy", 5, 5, reviewer_name="alex"),
            _item("r2", "factual_accuracy", 4, 5, reviewer_name="alex"),
            _item("r3", "factual_accuracy", 2, 5, reviewer_name="jamie"),
        ]

    def test_reviewer_summary_one_row_per_reviewer(self) -> None:
        analytics = compute_reviewer_analytics(self._reviewer_items())
        rows = build_reviewer_summary_rows(analytics)
        reviewers = [r["reviewer"] for r in rows]
        assert reviewers == ["alex", "jamie"]
        # alex has 2 samples, jamie has 1
        samples = {r["reviewer"]: r["samples"] for r in rows}
        assert samples == {"alex": 2, "jamie": 1}

    def test_reviewer_pillar_rows_populated(self) -> None:
        analytics = compute_reviewer_analytics(
            self._reviewer_items(),
            pillars=["factual_accuracy"],
        )
        rows = build_reviewer_pillar_rows(analytics)
        # one row per (reviewer, pillar); we restricted to one pillar
        assert len(rows) == 2
        for row in rows:
            assert row["pillar"] == "factual_accuracy"
            assert "disagreement_%" in row

    def test_reviewer_pair_rows_empty_without_overlap(self) -> None:
        analytics = compute_reviewer_analytics(self._reviewer_items())
        # No overlap on (record_id, pillar); pairs list is empty.
        assert build_reviewer_pair_rows(analytics) == []


# ---------------------------------------------------------------------------
# Disagreement rows
# ---------------------------------------------------------------------------


class TestDisagreementRows:
    def test_full_row_set_populated(self) -> None:
        rows = [
            _row(
                "r1",
                category="billing",
                reviewer_name="alex",
                labels={"factual_accuracy": 5},
            ),
            _row(
                "r2",
                category="support",
                reviewer_name="jamie",
                labels={"factual_accuracy": 4},
            ),
        ]
        outcomes = [
            _outcome("factual_accuracy", "r1", score=4, latency_ms=12.3),
            _outcome("factual_accuracy", "r2", score=4, latency_ms=9.9),
        ]
        table = build_disagreement_rows(rows, outcomes)
        assert len(table) == 2
        first = table[0]
        for key in (
            "record_id",
            "pillar",
            "category",
            "reviewer",
            "judge_score",
            "human_score",
            "distance",
            "confidence",
            "decision_summary",
            "failure_tags",
            "why_not_higher",
            "why_not_lower",
            "user_input",
            "agent_output",
            "latency_ms",
            "model",
        ):
            assert key in first, f"missing {key} in disagreement row"
        assert first["distance"] == 1
        assert first["reviewer"] == "alex"

    def test_filtered_items_restrict_output(self) -> None:
        rows = [
            _row("r1", labels={"factual_accuracy": 5}),
            _row("r2", labels={"factual_accuracy": 5}),
        ]
        outcomes = [
            _outcome("factual_accuracy", "r1", score=5),
            _outcome("factual_accuracy", "r2", score=3),
        ]
        # Only keep r2 via the items list.
        items = [
            ScoredItem(
                record_id="r2",
                pillar="factual_accuracy",
                judge_score=3,
                human_score=5,
                category="billing",
            )
        ]
        table = build_disagreement_rows(rows, outcomes, items=items)
        assert [r["record_id"] for r in table] == ["r2"]

    def test_missing_label_yields_none_distance(self) -> None:
        rows = [_row("r1")]  # no labels
        outcomes = [_outcome("factual_accuracy", "r1", score=3)]
        table = build_disagreement_rows(rows, outcomes)
        assert table[0]["human_score"] is None
        assert table[0]["distance"] is None

    def test_failed_outcome_skipped(self) -> None:
        rows = [_row("r1", labels={"factual_accuracy": 5})]
        outcomes = [
            JudgeOutcome(
                pillar="factual_accuracy",
                record_id="r1",
                latency_ms=0.0,
                attempts=1,
                usage=LLMUsage(),
                model_name="mock",
                run_id="run",
                error="provider down",
            )
        ]
        assert build_disagreement_rows(rows, outcomes) == []
