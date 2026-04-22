"""Unit tests for the core Pydantic contracts.

Covers the validation rules declared in ``docs/JUDGE_OUTPUT_CONTRACT.md``
and ``dataset_contract.md``. These are the single stable surface the rest
of the system is built against, so we want the invariants pinned tightly.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from src.core.types import (
    Evidence,
    JudgeResult,
    NormalizedRow,
    RunContext,
    Turn,
)

# ---------------------------------------------------------------------------
# NormalizedRow
# ---------------------------------------------------------------------------


class TestNormalizedRow:
    def test_accepts_minimal_required_fields(
        self, normalized_row_kwargs: dict[str, Any]
    ) -> None:
        row = NormalizedRow(**normalized_row_kwargs)
        assert row.record_id == "r-001"
        assert row.category == "transactions"

    @pytest.mark.parametrize(
        "missing", ["record_id", "user_input", "agent_output", "category"]
    )
    def test_rejects_missing_required_field(
        self, normalized_row_kwargs: dict[str, Any], missing: str
    ) -> None:
        kwargs = dict(normalized_row_kwargs)
        kwargs.pop(missing)
        with pytest.raises(ValidationError):
            NormalizedRow(**kwargs)

    @pytest.mark.parametrize("field", ["record_id", "category"])
    def test_rejects_empty_required_strings(
        self, normalized_row_kwargs: dict[str, Any], field: str
    ) -> None:
        kwargs = dict(normalized_row_kwargs)
        kwargs[field] = ""
        with pytest.raises(ValidationError):
            NormalizedRow(**kwargs)

    def test_rejects_extra_fields(self, normalized_row_kwargs: dict[str, Any]) -> None:
        kwargs = dict(normalized_row_kwargs)
        kwargs["unknown_column"] = "oops"
        with pytest.raises(ValidationError):
            NormalizedRow(**kwargs)

    def test_defaults_have_all_flags_false(self, normalized_row: NormalizedRow) -> None:
        assert normalized_row.has_reviewer is False
        assert normalized_row.has_context is False
        assert normalized_row.has_history is False
        assert normalized_row.has_labels is False
        assert normalized_row.has_ground_truth is False

    def test_reviewer_flag_set_by_reviewer_name(
        self, normalized_row_kwargs: dict[str, Any]
    ) -> None:
        row = NormalizedRow(**normalized_row_kwargs, reviewer_name="Alex")
        assert row.has_reviewer is True

    def test_reviewer_flag_set_by_reviewer_id(
        self, normalized_row_kwargs: dict[str, Any]
    ) -> None:
        row = NormalizedRow(**normalized_row_kwargs, reviewer_id="rev-42")
        assert row.has_reviewer is True

    def test_context_flag_requires_non_empty_list(
        self, normalized_row_kwargs: dict[str, Any]
    ) -> None:
        empty = NormalizedRow(**normalized_row_kwargs, retrieved_context=[])
        assert empty.has_context is False
        populated = NormalizedRow(
            **normalized_row_kwargs, retrieved_context=["chunk-1", "chunk-2"]
        )
        assert populated.has_context is True

    def test_history_flag_requires_non_empty_list(
        self, normalized_row_kwargs: dict[str, Any]
    ) -> None:
        hist = NormalizedRow(
            **normalized_row_kwargs,
            chat_history=[Turn(role="user", content="hello")],
        )
        assert hist.has_history is True

    def test_labels_flag_triggered_by_any_pillar_label(
        self, normalized_row_kwargs: dict[str, Any]
    ) -> None:
        row = NormalizedRow(**normalized_row_kwargs, label_factual_accuracy=4)
        assert row.has_labels is True

    def test_ground_truth_flag(self, normalized_row_kwargs: dict[str, Any]) -> None:
        row = NormalizedRow(
            **normalized_row_kwargs, ground_truth_answer="A dispute takes 10 days."
        )
        assert row.has_ground_truth is True

    def test_derived_flags_overwrite_user_supplied_values(
        self, normalized_row_kwargs: dict[str, Any]
    ) -> None:
        # User lies about has_reviewer; validator must overwrite to match reality.
        row = NormalizedRow(**normalized_row_kwargs, has_reviewer=True)
        assert row.has_reviewer is False

    def test_source_extras_roundtrip(
        self, normalized_row_kwargs: dict[str, Any]
    ) -> None:
        extras = {"orig_col_a": "x", "orig_col_b": 42, "nested": {"k": "v"}}
        row = NormalizedRow(**normalized_row_kwargs, source_extras=extras)
        assert row.source_extras == extras

    @pytest.mark.parametrize("bad_score", [0, 6, -1, 100])
    def test_rejects_label_out_of_range(
        self, normalized_row_kwargs: dict[str, Any], bad_score: int
    ) -> None:
        kwargs = dict(normalized_row_kwargs)
        kwargs["label_hallucination"] = bad_score
        with pytest.raises(ValidationError):
            NormalizedRow(**kwargs)


# ---------------------------------------------------------------------------
# Turn + Evidence
# ---------------------------------------------------------------------------


class TestTurn:
    def test_valid_roles(self) -> None:
        for role in ("user", "assistant", "system", "tool"):
            t = Turn(role=role, content="hi")  # type: ignore[arg-type]
            assert t.role == role

    def test_invalid_role(self) -> None:
        with pytest.raises(ValidationError):
            Turn(role="sme", content="hi")  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        t = Turn(role="user", content="hi")
        with pytest.raises(ValidationError):
            t.content = "changed"  # type: ignore[misc]


class TestEvidence:
    @pytest.mark.parametrize(
        "status", ["supported", "unsupported", "contradicted", "ungrounded", "n/a"]
    )
    def test_valid_status(self, status: str) -> None:
        e = Evidence(claim="x", status=status, support=None)  # type: ignore[arg-type]
        assert e.status == status

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            Evidence(claim="x", status="partial", support=None)  # type: ignore[arg-type]

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Evidence(claim="x", status="supported", support=None, severity="high")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# JudgeResult
# ---------------------------------------------------------------------------


class TestJudgeResult:
    def test_accepts_valid_example(self, judge_result_kwargs: dict[str, Any]) -> None:
        r = JudgeResult(**judge_result_kwargs)
        assert r.pillar == "factual_accuracy"
        assert r.score == 4
        assert len(r.evidence_for_score) == 2

    def test_rejects_extra_fields(self, judge_result_kwargs: dict[str, Any]) -> None:
        kwargs = dict(judge_result_kwargs)
        kwargs["mystery_field"] = "???"
        with pytest.raises(ValidationError):
            JudgeResult(**kwargs)

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -1.0])
    def test_rejects_confidence_out_of_range(
        self, judge_result_kwargs: dict[str, Any], bad: float
    ) -> None:
        kwargs = dict(judge_result_kwargs)
        kwargs["confidence"] = bad
        with pytest.raises(ValidationError):
            JudgeResult(**kwargs)

    @pytest.mark.parametrize("bad_score", [0, 6, -1])
    def test_rejects_score_out_of_literal(
        self, judge_result_kwargs: dict[str, Any], bad_score: int
    ) -> None:
        kwargs = dict(judge_result_kwargs)
        kwargs["score"] = bad_score
        with pytest.raises(ValidationError):
            JudgeResult(**kwargs)

    def test_score_5_requires_empty_failure_tags(
        self, judge_result_kwargs: dict[str, Any]
    ) -> None:
        kwargs = dict(judge_result_kwargs)
        kwargs["score"] = 5
        kwargs["rubric_anchor"] = 5
        kwargs["why_not_higher"] = None  # not required at max score
        # Leaving failure_tags non-empty -> must raise.
        with pytest.raises(ValidationError, match="empty failure_tags"):
            JudgeResult(**kwargs)

    def test_score_5_clean_succeeds(
        self, judge_result_kwargs: dict[str, Any]
    ) -> None:
        kwargs = dict(judge_result_kwargs)
        kwargs["score"] = 5
        kwargs["rubric_anchor"] = 5
        kwargs["failure_tags"] = []
        kwargs["why_not_higher"] = None  # score=5: no higher to reach
        r = JudgeResult(**kwargs)
        assert r.score == 5
        assert r.failure_tags == []

    def test_why_not_higher_required_below_max(
        self, judge_result_kwargs: dict[str, Any]
    ) -> None:
        kwargs = dict(judge_result_kwargs)
        kwargs["score"] = 3
        kwargs["rubric_anchor"] = 3
        kwargs["why_not_higher"] = None
        with pytest.raises(ValidationError, match="why_not_higher"):
            JudgeResult(**kwargs)

    def test_why_not_lower_required_above_min(
        self, judge_result_kwargs: dict[str, Any]
    ) -> None:
        kwargs = dict(judge_result_kwargs)
        kwargs["score"] = 3
        kwargs["rubric_anchor"] = 3
        kwargs["why_not_lower"] = None
        with pytest.raises(ValidationError, match="why_not_lower"):
            JudgeResult(**kwargs)

    def test_score_1_may_omit_why_not_lower(
        self, judge_result_kwargs: dict[str, Any]
    ) -> None:
        kwargs = dict(judge_result_kwargs)
        kwargs["score"] = 1
        kwargs["rubric_anchor"] = 1
        kwargs["failure_tags"] = ["bad"]
        kwargs["why_not_lower"] = None  # score=1: no lower to reach
        r = JudgeResult(**kwargs)
        assert r.score == 1

    @pytest.mark.parametrize("delta", [2, 3, 4, -2, -3])
    def test_rubric_anchor_within_one_of_score(
        self, judge_result_kwargs: dict[str, Any], delta: int
    ) -> None:
        kwargs = dict(judge_result_kwargs)
        score = kwargs["score"]
        anchor = score + delta
        if 1 <= anchor <= 5:
            kwargs["rubric_anchor"] = anchor
            with pytest.raises(ValidationError, match="rubric_anchor"):
                JudgeResult(**kwargs)

    def test_rubric_anchor_exactly_one_away_ok(
        self, judge_result_kwargs: dict[str, Any]
    ) -> None:
        kwargs = dict(judge_result_kwargs)
        kwargs["rubric_anchor"] = kwargs["score"] - 1
        r = JudgeResult(**kwargs)
        assert r.rubric_anchor == r.score - 1


# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------


class TestRunContext:
    def test_accepts_valid(self, run_context: RunContext) -> None:
        assert run_context.run_id == "run-test-0001"
        assert run_context.langfuse_span is None

    def test_rejects_empty_run_id(self) -> None:
        with pytest.raises(ValidationError):
            RunContext(run_id="", dataset_fingerprint="sha256:x")

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            RunContext(
                run_id="r", dataset_fingerprint="sha256:x", rogue="no"  # type: ignore[call-arg]
            )
