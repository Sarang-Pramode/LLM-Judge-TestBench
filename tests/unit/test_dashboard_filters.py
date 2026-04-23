"""Disagreement-filter tests.

The filter primitive drives the Stage 9 disagreement explorer. We
cover:

- Severity bucket semantics (including the cumulative ``within_1``).
- Each dimension (pillar / category / reviewer / min_distance) in
  isolation.
- Combinations (AND semantics).
- Reviewer precedence (name preferred over id).
"""

from __future__ import annotations

from src.dashboard.filters import (
    DisagreementFilter,
    SeverityBucket,
    apply_filter,
    distinct_categories,
    distinct_pillars,
    distinct_reviewers,
    severity_bucket,
)
from src.evaluation.join import ScoredItem


def _item(
    record_id: str,
    pillar: str,
    judge: int,
    human: int,
    *,
    category: str = "billing",
    reviewer_name: str | None = None,
    reviewer_id: str | None = None,
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


SAMPLE = [
    _item("r0", "factual_accuracy", 5, 5, category="billing", reviewer_name="alex"),
    _item("r1", "factual_accuracy", 4, 5, category="billing", reviewer_name="jamie"),
    _item("r2", "factual_accuracy", 3, 5, category="support", reviewer_name="alex"),
    _item("r3", "factual_accuracy", 1, 5, category="support", reviewer_name="jamie"),
    _item("r4", "relevance", 3, 5, category="billing", reviewer_id="rev-7"),
    _item("r5", "relevance", 5, 5, category="support"),
]


class TestSeverityBucket:
    def test_mapping_is_exhaustive(self) -> None:
        assert severity_bucket(0) is SeverityBucket.EXACT
        assert severity_bucket(1) is SeverityBucket.WITHIN_1
        assert severity_bucket(2) is SeverityBucket.OFF_BY_2
        assert severity_bucket(3) is SeverityBucket.LARGE_MISS
        assert severity_bucket(4) is SeverityBucket.LARGE_MISS


class TestDisagreementFilter:
    def test_empty_filter_matches_everything(self) -> None:
        out = apply_filter(SAMPLE, DisagreementFilter())
        assert len(out) == len(SAMPLE)

    def test_pillar_restriction(self) -> None:
        filt = DisagreementFilter(pillars=frozenset({"relevance"}))
        out = apply_filter(SAMPLE, filt)
        assert {it.pillar for it in out} == {"relevance"}
        assert len(out) == 2

    def test_category_restriction(self) -> None:
        filt = DisagreementFilter(categories=frozenset({"support"}))
        out = apply_filter(SAMPLE, filt)
        assert all(it.category == "support" for it in out)
        assert len(out) == 3

    def test_reviewer_restriction_prefers_name_over_id(self) -> None:
        filt = DisagreementFilter(reviewers=frozenset({"alex"}))
        out = apply_filter(SAMPLE, filt)
        assert {it.record_id for it in out} == {"r0", "r2"}

    def test_reviewer_restriction_matches_reviewer_id_when_name_missing(self) -> None:
        filt = DisagreementFilter(reviewers=frozenset({"rev-7"}))
        out = apply_filter(SAMPLE, filt)
        assert [it.record_id for it in out] == ["r4"]

    def test_reviewer_restriction_excludes_items_with_no_reviewer(self) -> None:
        filt = DisagreementFilter(reviewers=frozenset({"alex", "jamie"}))
        out = apply_filter(SAMPLE, filt)
        assert "r5" not in {it.record_id for it in out}

    def test_severity_exact_only(self) -> None:
        filt = DisagreementFilter(severity=SeverityBucket.EXACT)
        out = apply_filter(SAMPLE, filt)
        assert {it.record_id for it in out} == {"r0", "r5"}

    def test_severity_within_1_is_cumulative(self) -> None:
        filt = DisagreementFilter(severity=SeverityBucket.WITHIN_1)
        out = apply_filter(SAMPLE, filt)
        # distance 0 and 1 both qualify
        assert {it.record_id for it in out} == {"r0", "r1", "r5"}

    def test_severity_large_miss(self) -> None:
        filt = DisagreementFilter(severity=SeverityBucket.LARGE_MISS)
        out = apply_filter(SAMPLE, filt)
        assert {it.record_id for it in out} == {"r3"}

    def test_min_distance_floor(self) -> None:
        filt = DisagreementFilter(min_distance=2)
        out = apply_filter(SAMPLE, filt)
        assert {it.record_id for it in out} == {"r2", "r3", "r4"}

    def test_combination_and_semantics(self) -> None:
        filt = DisagreementFilter(
            pillars=frozenset({"factual_accuracy"}),
            categories=frozenset({"support"}),
            severity=SeverityBucket.LARGE_MISS,
        )
        out = apply_filter(SAMPLE, filt)
        assert [it.record_id for it in out] == ["r3"]

    def test_preserves_input_order(self) -> None:
        filt = DisagreementFilter(pillars=frozenset({"factual_accuracy"}))
        out = apply_filter(SAMPLE, filt)
        assert [it.record_id for it in out] == ["r0", "r1", "r2", "r3"]

    def test_is_empty_flag(self) -> None:
        assert DisagreementFilter().is_empty() is True
        assert DisagreementFilter(pillars=frozenset({"x"})).is_empty() is False


class TestDistinctHelpers:
    def test_distinct_pillars_sorted(self) -> None:
        assert distinct_pillars(SAMPLE) == ["factual_accuracy", "relevance"]

    def test_distinct_categories_sorted(self) -> None:
        assert distinct_categories(SAMPLE) == ["billing", "support"]

    def test_distinct_reviewers_dedupes_name_over_id(self) -> None:
        assert distinct_reviewers(SAMPLE) == ["alex", "jamie", "rev-7"]

    def test_distinct_reviewers_empty_when_none_present(self) -> None:
        items = [_item("x", "relevance", 3, 3, category="billing")]
        assert distinct_reviewers(items) == []
