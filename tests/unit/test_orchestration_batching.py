"""Unit tests for :mod:`src.orchestration.batching`."""

from __future__ import annotations

import pytest

from src.core.types import NormalizedRow
from src.orchestration.batching import TaskSpec, chunk, plan_tasks


def _row(rid: str) -> NormalizedRow:
    return NormalizedRow(
        record_id=rid,
        user_input="u",
        agent_output="a",
        category="c",
    )


class TestPlanTasks:
    def test_cross_product_is_row_major(self) -> None:
        rows = [_row("r1"), _row("r2")]
        tasks = plan_tasks(rows, ("p1", "p2", "p3"))
        assert [t.task_id for t in tasks] == [
            "r1::p1",
            "r1::p2",
            "r1::p3",
            "r2::p1",
            "r2::p2",
            "r2::p3",
        ]

    def test_empty_rows_returns_empty_list(self) -> None:
        assert plan_tasks([], ("p1",)) == []

    def test_empty_pillars_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one pillar"):
            plan_tasks([_row("r1")], ())

    def test_duplicate_pillars_raise(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            plan_tasks([_row("r1")], ("p1", "p1"))

    def test_row_index_preserved(self) -> None:
        rows = [_row("a"), _row("b"), _row("c")]
        tasks = plan_tasks(rows, ("p1",))
        assert [t.row_index for t in tasks] == [0, 1, 2]


class TestTaskSpec:
    def test_task_id_format(self) -> None:
        spec = TaskSpec(row_index=3, record_id="abc", pillar="relevance")
        assert spec.task_id == "abc::relevance"

    def test_frozen(self) -> None:
        spec = TaskSpec(row_index=0, record_id="r", pillar="p")
        with pytest.raises(Exception):  # FrozenInstanceError
            spec.pillar = "q"  # type: ignore[misc]


class TestChunk:
    def test_even_split(self) -> None:
        assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_uneven_split_keeps_tail(self) -> None:
        assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

    def test_empty_input(self) -> None:
        assert chunk([], 4) == []

    def test_size_larger_than_input(self) -> None:
        assert chunk([1, 2], 10) == [[1, 2]]

    def test_invalid_size_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk([1, 2], 0)
