"""Join + agreement tests.

Covers:

- :func:`join_outcomes_with_labels` contract (label-missing, failure,
  orphan outcome, restricted pillar list).
- :func:`compute_pillar_agreement` / :func:`compute_agreement_report`
  shape + graceful handling of empty buckets.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.core.constants import PILLARS
from src.core.types import JudgeResult, NormalizedRow
from src.evaluation.agreement import (
    compute_agreement_report,
    compute_pillar_agreement,
)
from src.evaluation.join import (
    JoinedDataset,
    ScoredItem,
    join_outcomes_with_labels,
)
from src.judges.base import JudgeOutcome
from src.llm.base import LLMUsage

# ---------------------------------------------------------------------------
# Tiny factories (intentionally local; no conftest pollution)
# ---------------------------------------------------------------------------


def _row(
    record_id: str,
    *,
    category: str = "billing",
    labels: dict[str, int] | None = None,
    reviewer_name: str | None = None,
    intent: str | None = None,
    topic: str | None = None,
    **extra: Any,
) -> NormalizedRow:
    kwargs: dict[str, Any] = {
        "record_id": record_id,
        "user_input": "?",
        "agent_output": "!",
        "category": category,
        "reviewer_name": reviewer_name,
        "intent": intent,
        "topic": topic,
    }
    for pillar, value in (labels or {}).items():
        kwargs[f"label_{pillar}"] = value
    kwargs.update(extra)
    return NormalizedRow(**kwargs)


def _result(pillar: str, score: int) -> JudgeResult:
    kwargs: dict[str, Any] = {
        "pillar": pillar,
        "score": score,
        "confidence": 0.9,
        "decision_summary": "ok",
        "rubric_anchor": score,
        "raw_model_name": "mock-v0",
        "prompt_version": "p1",
        "rubric_version": "r1",
    }
    if score < 5:
        kwargs["why_not_higher"] = "room to grow"
        kwargs["failure_tags"] = ["tag"]
    if score > 1:
        kwargs["why_not_lower"] = "not the worst"
    return JudgeResult(**kwargs)


def _outcome(
    pillar: str,
    record_id: str,
    *,
    score: int | None = None,
    error: str | None = None,
) -> JudgeOutcome:
    return JudgeOutcome(
        pillar=pillar,
        record_id=record_id,
        latency_ms=1.0,
        attempts=1,
        usage=LLMUsage(),
        model_name="mock-v0",
        run_id="run-test",
        result=_result(pillar, score) if score is not None else None,
        error=error,
    )


# ---------------------------------------------------------------------------
# Join tests
# ---------------------------------------------------------------------------


class TestJoinOutcomesWithLabels:
    def test_happy_path_pairs_every_labelled_successful_outcome(self) -> None:
        rows = [
            _row("r1", labels={"factual_accuracy": 5, "relevance": 4}),
            _row("r2", labels={"factual_accuracy": 3}),
        ]
        outcomes = [
            _outcome("factual_accuracy", "r1", score=4),
            _outcome("factual_accuracy", "r2", score=3),
            _outcome("relevance", "r1", score=5),
        ]
        joined = join_outcomes_with_labels(rows, outcomes)
        assert isinstance(joined, JoinedDataset)
        assert joined.stats.paired == 3
        assert joined.stats.missing_labels == 0
        assert joined.stats.failed_outcomes == 0
        assert joined.stats.missing_outcomes == (2 * len(PILLARS) - 3)
        assert {it.pillar for it in joined.items} == {"factual_accuracy", "relevance"}

    def test_missing_label_drops_that_pair_only(self) -> None:
        rows = [_row("r1", labels={"relevance": 5})]  # no FA label
        outcomes = [
            _outcome("factual_accuracy", "r1", score=5),
            _outcome("relevance", "r1", score=4),
        ]
        joined = join_outcomes_with_labels(rows, outcomes)
        assert joined.stats.missing_labels == 1  # FA
        assert joined.stats.paired == 1
        assert joined.items[0].pillar == "relevance"

    def test_failed_outcome_is_tracked_and_dropped(self) -> None:
        rows = [_row("r1", labels={"factual_accuracy": 5})]
        outcomes = [_outcome("factual_accuracy", "r1", error="boom")]
        joined = join_outcomes_with_labels(rows, outcomes)
        assert joined.stats.failed_outcomes == 1
        assert joined.stats.paired == 0
        assert not joined.items

    def test_orphan_outcome_is_counted(self) -> None:
        rows = [_row("r1", labels={"factual_accuracy": 5})]
        # Outcome refers to a row id that doesn't exist in ``rows``.
        outcomes = [_outcome("factual_accuracy", "ghost", score=3)]
        joined = join_outcomes_with_labels(rows, outcomes)
        assert joined.stats.orphan_outcomes == 1
        assert joined.stats.paired == 0

    def test_pillar_restriction_ignores_other_outcomes(self) -> None:
        rows = [_row("r1", labels={"factual_accuracy": 5, "relevance": 5})]
        outcomes = [
            _outcome("factual_accuracy", "r1", score=5),
            _outcome("relevance", "r1", score=5),
        ]
        joined = join_outcomes_with_labels(
            rows,
            outcomes,
            pillars=["relevance"],
        )
        assert joined.stats.paired == 1
        assert joined.items[0].pillar == "relevance"
        # Factual outcome is ignored: doesn't land in any stats counter.
        assert joined.stats.failed_outcomes == 0
        assert joined.stats.missing_labels == 0

    def test_scored_item_carries_slice_metadata(self) -> None:
        rows = [
            _row(
                "r1",
                category="disputes",
                labels={"toxicity": 5},
                reviewer_name="alex",
                intent="dispute_initiation",
                topic="fraud",
            )
        ]
        outcomes = [_outcome("toxicity", "r1", score=5)]
        joined = join_outcomes_with_labels(rows, outcomes)
        item = joined.items[0]
        assert isinstance(item, ScoredItem)
        assert item.category == "disputes"
        assert item.reviewer_name == "alex"
        assert item.intent == "dispute_initiation"
        assert item.topic == "fraud"
        assert item.distance == 0

    def test_missing_outcomes_counted(self) -> None:
        # 1 row, 6 pillars, but only 1 outcome delivered -> 5 missing.
        rows = [_row("r1", labels={p: 3 for p in PILLARS})]
        outcomes = [_outcome("factual_accuracy", "r1", score=3)]
        joined = join_outcomes_with_labels(rows, outcomes)
        assert joined.stats.missing_outcomes == len(PILLARS) - 1

    def test_for_pillar_filter(self) -> None:
        rows = [
            _row("r1", labels={"factual_accuracy": 5, "relevance": 5}),
        ]
        outcomes = [
            _outcome("factual_accuracy", "r1", score=4),
            _outcome("relevance", "r1", score=5),
        ]
        joined = join_outcomes_with_labels(rows, outcomes)
        fa_items = joined.for_pillar("factual_accuracy")
        assert len(fa_items) == 1
        assert fa_items[0].judge_score == 4


# ---------------------------------------------------------------------------
# Agreement report tests
# ---------------------------------------------------------------------------


class TestComputePillarAgreement:
    def test_happy_path_shape(self) -> None:
        items = [
            ScoredItem(
                record_id=f"r{i}",
                pillar="factual_accuracy",
                judge_score=j,
                human_score=h,
                category="c",
            )
            for i, (j, h) in enumerate([(5, 5), (4, 5), (3, 5), (2, 5)])
        ]
        agreement = compute_pillar_agreement(items, pillar="factual_accuracy")
        assert agreement.support == 4
        assert agreement.exact_match_rate == pytest.approx(0.25)
        assert agreement.within_1_rate == pytest.approx(0.5)
        assert agreement.mean_absolute_error == pytest.approx((0 + 1 + 2 + 3) / 4)
        assert agreement.severity_aware_alignment == pytest.approx((1.0 + 0.75 + 0.40 + 0.10) / 4)
        # Confusion matrix row 4 (human=5) should sum to 4.
        assert sum(agreement.confusion_matrix[4]) == 4
        # Judge distribution covers the 5-bin range.
        assert set(agreement.judge_score_distribution) == {1, 2, 3, 4, 5}

    def test_empty_support_returns_defaults(self) -> None:
        agreement = compute_pillar_agreement([], pillar="relevance")
        assert agreement.support == 0
        assert not agreement.has_support
        assert agreement.exact_match_rate == 0.0
        assert agreement.weighted_kappa is None
        assert agreement.spearman_correlation is None


class TestComputeAgreementReport:
    def test_all_pillars_reported_even_with_zero_support(self) -> None:
        # Only factual_accuracy has data; the other 5 pillars get
        # zero-support placeholders so the dashboard can render a
        # consistent grid without special cases.
        items = [
            ScoredItem("r1", "factual_accuracy", 4, 5, "c"),
            ScoredItem("r2", "factual_accuracy", 5, 5, "c"),
        ]
        report = compute_agreement_report(items)
        assert set(report.per_pillar) == set(PILLARS)
        for pillar in PILLARS:
            if pillar == "factual_accuracy":
                assert report.per_pillar[pillar].support == 2
            else:
                assert report.per_pillar[pillar].support == 0

    def test_overall_aggregates_across_pillars(self) -> None:
        items = [
            ScoredItem("r1", "factual_accuracy", 3, 5, "c"),
            ScoredItem("r1", "relevance", 5, 5, "c"),
        ]
        report = compute_agreement_report(items)
        assert report.overall is not None
        assert report.overall.support == 2
        assert report.overall.pillar == "__overall__"

    def test_pillar_restriction_honoured(self) -> None:
        items = [ScoredItem("r1", "relevance", 5, 5, "c")]
        report = compute_agreement_report(items, pillars=["relevance"])
        assert list(report.per_pillar) == ["relevance"]

    def test_include_overall_flag(self) -> None:
        items = [ScoredItem("r1", "relevance", 5, 5, "c")]
        report = compute_agreement_report(items, include_overall=False)
        assert report.overall is None

    def test_pillars_method_sorted(self) -> None:
        items = [ScoredItem("r1", "toxicity", 5, 5, "c")]
        report = compute_agreement_report(items)
        assert report.pillars() == sorted(report.per_pillar)
