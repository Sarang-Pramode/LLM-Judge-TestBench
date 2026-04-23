"""Contract tests for Stage 9 Altair chart builders.

We don't render pixels - Altair charts serialise to JSON specs we can
inspect. Contract-level checks that would break the dashboard if a
chart builder regressed:

- Returns an ``alt.Chart`` / ``alt.LayerChart`` (serialisable).
- Empty / zero-support inputs yield the "no data" placeholder rather
  than an exception.
- Encoded fields match the expected columns (so ``st.altair_chart``
  has something to render and tooltips don't vanish).
"""

from __future__ import annotations

import altair as alt
import pytest

from src.dashboard.charts import (
    build_category_pillar_heatmap,
    build_confusion_matrix_heatmap,
    build_large_miss_by_category_chart,
    build_no_data_chart,
    build_pillar_agreement_bar,
    build_reviewer_agreement_bar,
    build_score_distribution_bar,
)
from src.evaluation.agreement import AgreementReport, compute_agreement_report
from src.evaluation.join import ScoredItem
from src.evaluation.reviewer_analysis import (
    ReviewerAnalytics,
    compute_reviewer_analytics,
)
from src.evaluation.slices import SliceReport, compute_sliced_report, slice_by_category


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


def _report_with_data() -> AgreementReport:
    items = [
        _item("r1", "factual_accuracy", 5, 5),
        _item("r2", "factual_accuracy", 4, 5),
        _item("r3", "factual_accuracy", 2, 5),
        _item("r1", "relevance", 3, 4),
        _item("r2", "relevance", 4, 4),
    ]
    return compute_agreement_report(items, pillars=["factual_accuracy", "relevance"])


def _sliced_with_data() -> SliceReport:
    items = [
        _item("r1", "factual_accuracy", 5, 5, category="billing"),
        _item("r2", "factual_accuracy", 3, 5, category="billing"),
        _item("r3", "factual_accuracy", 4, 5, category="support"),
        _item("r4", "relevance", 2, 5, category="support"),
    ]
    return compute_sliced_report(
        items,
        selector=slice_by_category,
        dimension="category",
        pillars=["factual_accuracy", "relevance"],
        include_overall_per_slice=False,
    )


# ---------------------------------------------------------------------------
# No-data chart
# ---------------------------------------------------------------------------


class TestNoDataChart:
    def test_returns_altair_chart(self) -> None:
        chart = build_no_data_chart("nothing here")
        spec = chart.to_dict()
        assert spec["mark"]["type"] == "text"


# ---------------------------------------------------------------------------
# Pillar agreement bar
# ---------------------------------------------------------------------------


class TestPillarAgreementBar:
    def test_encoding_axes(self) -> None:
        report = _report_with_data()
        chart = build_pillar_agreement_bar(report, metric="severity_alignment")
        spec = chart.to_dict()
        enc = spec["encoding"]
        assert enc["x"]["field"] == "pillar"
        assert enc["y"]["field"] == "severity_alignment"

    def test_unknown_metric_raises(self) -> None:
        report = _report_with_data()
        with pytest.raises(ValueError, match="unknown metric"):
            build_pillar_agreement_bar(report, metric="nonsense")

    def test_empty_report_returns_no_data(self) -> None:
        empty = compute_agreement_report([], pillars=["relevance"])
        chart = build_pillar_agreement_bar(empty)
        spec = chart.to_dict()
        assert spec["mark"]["type"] == "text"

    def test_mae_metric_drops_bounded_scale(self) -> None:
        report = _report_with_data()
        chart = build_pillar_agreement_bar(report, metric="mae")
        spec = chart.to_dict()
        y = spec["encoding"]["y"]
        assert y["field"] == "mae"
        # Rate metrics force [0, 1]; mae does not.
        assert "scale" not in y or "domain" not in y.get("scale", {})


# ---------------------------------------------------------------------------
# Score distribution
# ---------------------------------------------------------------------------


class TestScoreDistributionBar:
    def test_encodes_source_split(self) -> None:
        report = _report_with_data()
        chart = build_score_distribution_bar(report.per_pillar["factual_accuracy"])
        spec = chart.to_dict()
        # Grouped bar via xOffset of the source channel.
        assert spec["encoding"]["xOffset"]["field"] == "source"
        assert spec["encoding"]["color"]["field"] == "source"

    def test_no_support_returns_placeholder(self) -> None:
        empty = compute_agreement_report([], pillars=["relevance"])
        chart = build_score_distribution_bar(empty.per_pillar["relevance"])
        spec = chart.to_dict()
        assert spec["mark"]["type"] == "text"


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


class TestCategoryPillarHeatmap:
    def test_encodes_category_on_y_and_pillar_on_x(self) -> None:
        sliced = _sliced_with_data()
        chart = build_category_pillar_heatmap(sliced, metric="exact_match")
        spec = chart.to_dict()
        enc = spec["encoding"]
        assert enc["y"]["field"] == "category"
        assert enc["x"]["field"] == "pillar"
        assert enc["color"]["field"] == "value"

    def test_empty_slice_report_is_placeholder(self) -> None:
        empty = compute_sliced_report(
            [],
            selector=slice_by_category,
            dimension="category",
            pillars=["relevance"],
            include_overall_per_slice=False,
        )
        chart = build_category_pillar_heatmap(empty)
        spec = chart.to_dict()
        assert spec["mark"]["type"] == "text"


# ---------------------------------------------------------------------------
# Large-miss chart
# ---------------------------------------------------------------------------


class TestLargeMissChart:
    def test_single_pillar_view(self) -> None:
        sliced = _sliced_with_data()
        chart = build_large_miss_by_category_chart(sliced, pillar="factual_accuracy")
        spec = chart.to_dict()
        assert spec["encoding"]["x"]["field"] == "category"
        assert spec["encoding"]["y"]["field"] == "large_miss_rate"

    def test_cross_pillar_view(self) -> None:
        sliced = _sliced_with_data()
        chart = build_large_miss_by_category_chart(sliced, pillar=None)
        spec = chart.to_dict()
        assert spec["encoding"]["y"]["field"] == "large_miss_rate"


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


class TestConfusionMatrixHeatmap:
    def test_layered_heatmap_and_labels(self) -> None:
        report = _report_with_data()
        chart = build_confusion_matrix_heatmap(report.per_pillar["factual_accuracy"])
        spec = chart.to_dict()
        # LayerChart has a ``layer`` list with both a rect and a text mark.
        layers = spec.get("layer", [])
        mark_types: set[str] = set()
        for layer in layers:
            mark = layer.get("mark")
            if isinstance(mark, dict):
                mark_types.add(mark.get("type", ""))
            elif isinstance(mark, str):
                mark_types.add(mark)
        assert {"rect", "text"}.issubset(mark_types)

    def test_no_support_placeholder(self) -> None:
        empty = compute_agreement_report([], pillars=["relevance"])
        chart = build_confusion_matrix_heatmap(empty.per_pillar["relevance"])
        spec = chart.to_dict()
        assert spec["mark"]["type"] == "text"


# ---------------------------------------------------------------------------
# Reviewer agreement
# ---------------------------------------------------------------------------


class TestReviewerAgreementBar:
    def _analytics(self) -> ReviewerAnalytics:
        items = [
            _item("r1", "factual_accuracy", 5, 5, reviewer_name="alex"),
            _item("r2", "factual_accuracy", 4, 5, reviewer_name="alex"),
            _item("r3", "factual_accuracy", 3, 5, reviewer_name="jamie"),
        ]
        return compute_reviewer_analytics(items)

    def test_overall_chart_encodings(self) -> None:
        chart = build_reviewer_agreement_bar(self._analytics())
        spec = chart.to_dict()
        assert spec["encoding"]["x"]["field"] == "reviewer"
        assert spec["encoding"]["y"]["field"] == "exact_match_rate"

    def test_per_pillar_chart(self) -> None:
        chart = build_reviewer_agreement_bar(self._analytics(), pillar="factual_accuracy")
        spec = chart.to_dict()
        assert spec["encoding"]["x"]["field"] == "reviewer"

    def test_no_reviewer_data_placeholder(self) -> None:
        empty = compute_reviewer_analytics([])
        chart = build_reviewer_agreement_bar(empty)
        spec = chart.to_dict()
        assert spec["mark"]["type"] == "text"


def test_all_chart_builders_produce_serialisable_specs() -> None:
    """Every chart serialises to a vega-lite compatible spec without
    raising. Catches future regressions where a builder returns a
    non-chart object."""
    report = _report_with_data()
    sliced = _sliced_with_data()
    charts = [
        build_pillar_agreement_bar(report),
        build_score_distribution_bar(report.per_pillar["factual_accuracy"]),
        build_category_pillar_heatmap(sliced),
        build_large_miss_by_category_chart(sliced),
        build_confusion_matrix_heatmap(report.per_pillar["factual_accuracy"]),
        build_no_data_chart(),
    ]
    for chart in charts:
        assert isinstance(chart, (alt.Chart, alt.LayerChart))
        spec = chart.to_dict()
        assert "$schema" in spec or "data" in spec or "mark" in spec
