"""Slicing tests.

``slice_by`` + ``compute_sliced_report`` must:

- Group items by the selector output.
- Route ``None`` / empty keys to :data:`UNKNOWN_SLICE`.
- Produce per-slice :class:`AgreementReport` objects usable by the
  Stage 9 dashboard without further manipulation.
"""

from __future__ import annotations

import pytest

from src.evaluation.join import ScoredItem
from src.evaluation.slices import (
    UNKNOWN_SLICE,
    compute_sliced_report,
    slice_by,
    slice_by_category,
    slice_by_intent,
    slice_by_reviewer,
)


def _item(
    record_id: str,
    pillar: str,
    judge: int,
    human: int,
    *,
    category: str = "billing",
    reviewer_name: str | None = None,
    intent: str | None = None,
) -> ScoredItem:
    return ScoredItem(
        record_id=record_id,
        pillar=pillar,
        judge_score=judge,
        human_score=human,
        category=category,
        reviewer_name=reviewer_name,
        intent=intent,
    )


class TestSliceBy:
    def test_groups_preserve_input_order(self) -> None:
        items = [
            _item("a", "factual_accuracy", 5, 5, category="X"),
            _item("b", "factual_accuracy", 4, 5, category="Y"),
            _item("c", "factual_accuracy", 3, 5, category="X"),
        ]
        buckets = slice_by(items, slice_by_category)
        assert [it.record_id for it in buckets["X"]] == ["a", "c"]
        assert [it.record_id for it in buckets["Y"]] == ["b"]

    def test_none_key_routes_to_unknown(self) -> None:
        items = [
            _item("a", "relevance", 5, 5, reviewer_name="alex"),
            _item("b", "relevance", 4, 5, reviewer_name=None),
        ]
        buckets = slice_by(items, slice_by_reviewer)
        assert set(buckets) == {"alex", UNKNOWN_SLICE}

    def test_empty_string_treated_as_unknown(self) -> None:
        # Empty string should behave like None - users rarely mean
        # "the literal empty string" as a real slice value.
        items = [_item("a", "relevance", 5, 5, intent="")]
        buckets = slice_by(items, slice_by_intent)
        assert UNKNOWN_SLICE in buckets


class TestComputeSlicedReport:
    def test_per_slice_reports_have_expected_support(self) -> None:
        items = [
            _item("a", "factual_accuracy", 5, 5, category="X"),
            _item("b", "factual_accuracy", 3, 5, category="X"),
            _item("c", "factual_accuracy", 4, 5, category="Y"),
            _item("d", "relevance", 5, 5, category="Y"),
        ]
        report = compute_sliced_report(
            items,
            selector=slice_by_category,
            dimension="category",
        )
        assert report.dimension == "category"
        assert set(report.per_slice) == {"X", "Y"}
        assert report.slice_counts == {"X": 2, "Y": 2}
        assert report.per_slice["X"].per_pillar["factual_accuracy"].support == 2
        assert report.per_slice["Y"].per_pillar["factual_accuracy"].support == 1
        assert report.per_slice["Y"].per_pillar["relevance"].support == 1

    def test_pillar_restriction_propagated(self) -> None:
        items = [
            _item("a", "factual_accuracy", 5, 5, category="X"),
            _item("b", "relevance", 5, 5, category="X"),
        ]
        report = compute_sliced_report(
            items,
            selector=slice_by_category,
            dimension="category",
            pillars=["relevance"],
        )
        assert list(report.per_slice["X"].per_pillar) == ["relevance"]

    def test_slices_method_sorted(self) -> None:
        items = [
            _item("a", "factual_accuracy", 5, 5, category="Zebra"),
            _item("b", "factual_accuracy", 5, 5, category="Alpha"),
        ]
        report = compute_sliced_report(
            items,
            selector=slice_by_category,
            dimension="category",
        )
        assert report.slices() == ["Alpha", "Zebra"]

    def test_include_overall_flag_propagated(self) -> None:
        items = [_item("a", "factual_accuracy", 5, 5, category="X")]
        report = compute_sliced_report(
            items,
            selector=slice_by_category,
            dimension="category",
            include_overall_per_slice=False,
        )
        assert report.per_slice["X"].overall is None

    def test_empty_input_yields_empty_report(self) -> None:
        report = compute_sliced_report(
            [],
            selector=slice_by_category,
            dimension="category",
        )
        assert report.per_slice == {}
        assert report.slice_counts == {}

    def test_per_slice_metrics_match_direct_compute(self) -> None:
        # Spot-check: exact-match rate computed via the sliced report
        # must equal the same rate computed directly.
        items = [
            _item("a", "factual_accuracy", 5, 5, category="X"),
            _item("b", "factual_accuracy", 3, 5, category="X"),
            _item("c", "factual_accuracy", 4, 5, category="X"),
        ]
        report = compute_sliced_report(
            items,
            selector=slice_by_category,
            dimension="category",
        )
        fa = report.per_slice["X"].per_pillar["factual_accuracy"]
        assert fa.exact_match_rate == pytest.approx(1 / 3)
        assert fa.mean_absolute_error == pytest.approx((0 + 2 + 1) / 3)
