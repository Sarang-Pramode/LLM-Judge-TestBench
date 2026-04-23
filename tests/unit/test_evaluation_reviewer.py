"""Reviewer analytics tests.

Covers:

- :func:`has_reviewer_signal` gates the tab correctly.
- :func:`compute_reviewer_analytics` produces per-reviewer per-pillar
  stats, including human/judge averages and disagreement rates.
- Reviewer-pair disagreement is populated only when >=2 reviewers
  scored the same ``(record_id, pillar)``.
"""

from __future__ import annotations

import pytest

from src.evaluation.join import ScoredItem
from src.evaluation.reviewer_analysis import (
    compute_reviewer_analytics,
    has_reviewer_signal,
)


def _item(
    record_id: str,
    pillar: str,
    judge: int,
    human: int,
    *,
    reviewer_name: str | None = None,
    reviewer_id: str | None = None,
    category: str = "billing",
) -> ScoredItem:
    return ScoredItem(
        record_id=record_id,
        pillar=pillar,
        judge_score=judge,
        human_score=human,
        category=category,
        reviewer_name=reviewer_name,
        reviewer_id=reviewer_id,
    )


class TestHasReviewerSignal:
    def test_false_when_all_reviewers_none(self) -> None:
        items = [_item("a", "factual_accuracy", 5, 5)]
        assert not has_reviewer_signal(items)

    def test_true_when_reviewer_name_present(self) -> None:
        items = [_item("a", "factual_accuracy", 5, 5, reviewer_name="alex")]
        assert has_reviewer_signal(items)

    def test_true_when_only_reviewer_id_present(self) -> None:
        items = [_item("a", "factual_accuracy", 5, 5, reviewer_id="rid-1")]
        assert has_reviewer_signal(items)


class TestComputeReviewerAnalytics:
    def test_per_reviewer_sample_count_and_metrics(self) -> None:
        items = [
            _item("a", "factual_accuracy", 5, 5, reviewer_name="alex"),
            _item("b", "factual_accuracy", 4, 5, reviewer_name="alex"),
            _item("c", "factual_accuracy", 2, 5, reviewer_name="alex"),
            _item("d", "factual_accuracy", 5, 5, reviewer_name="blake"),
        ]
        analytics = compute_reviewer_analytics(items)
        assert analytics.has_data
        assert set(analytics.per_reviewer) == {"alex", "blake"}

        alex = analytics.per_reviewer["alex"]
        assert alex.sample_count == 3

        fa = alex.per_pillar["factual_accuracy"]
        assert fa.support == 3
        assert fa.avg_human_score == pytest.approx(5.0)
        assert fa.avg_judge_score == pytest.approx((5 + 4 + 2) / 3)
        # Exact match on 1/3 -> disagreement 2/3
        assert fa.disagreement_rate == pytest.approx(2 / 3)
        # Within-1 on 2/3 (distance 0 and 1)
        assert fa.within_1_agreement == pytest.approx(2 / 3)
        # Large-miss (>=3) on 1/3
        assert fa.large_miss_rate == pytest.approx(1 / 3)

    def test_unreviewed_items_excluded(self) -> None:
        items = [
            _item("a", "factual_accuracy", 5, 5, reviewer_name="alex"),
            _item("b", "factual_accuracy", 3, 5, reviewer_name=None),
        ]
        analytics = compute_reviewer_analytics(items)
        assert list(analytics.per_reviewer) == ["alex"]
        assert analytics.per_reviewer["alex"].sample_count == 1

    def test_reviewer_pairs_empty_without_overlap(self) -> None:
        items = [
            _item("a", "factual_accuracy", 5, 5, reviewer_name="alex"),
            _item("b", "factual_accuracy", 5, 5, reviewer_name="blake"),
        ]
        analytics = compute_reviewer_analytics(items)
        assert analytics.reviewer_pairs == []

    def test_reviewer_pairs_populated_on_overlap(self) -> None:
        items = [
            # Same record_id + pillar, two reviewers: this is the only
            # path that populates ``reviewer_pairs``.
            _item("a", "factual_accuracy", 5, 5, reviewer_name="alex"),
            _item("a", "factual_accuracy", 5, 4, reviewer_name="blake"),
            _item("b", "factual_accuracy", 5, 5, reviewer_name="alex"),
            _item("b", "factual_accuracy", 5, 5, reviewer_name="blake"),
        ]
        analytics = compute_reviewer_analytics(items)
        assert len(analytics.reviewer_pairs) == 1
        pair = analytics.reviewer_pairs[0]
        # Pair is alphabetised.
        assert (pair.reviewer_a, pair.reviewer_b) == ("alex", "blake")
        assert pair.overlap == 2
        # Scores: alex(5,5), blake(4,5) on 'a'; alex(5), blake(5) on 'b'.
        # Exact match on 1/2 -> 0.5
        assert pair.exact_match_rate == pytest.approx(0.5)
        assert pair.within_1_rate == pytest.approx(1.0)
        assert pair.large_miss_rate == pytest.approx(0.0)

    def test_prefers_name_over_id_when_both_present(self) -> None:
        items = [
            _item(
                "a",
                "factual_accuracy",
                5,
                5,
                reviewer_name="alex",
                reviewer_id="rid-999",
            )
        ]
        analytics = compute_reviewer_analytics(items)
        assert "alex" in analytics.per_reviewer
        assert "rid-999" not in analytics.per_reviewer

    def test_empty_input(self) -> None:
        analytics = compute_reviewer_analytics([])
        assert not analytics.has_data
        assert analytics.reviewers() == []
