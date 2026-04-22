"""Tests for :mod:`src.judges.output_parser`."""

from __future__ import annotations

from typing import Any

import pytest

from src.core.exceptions import JudgeOutputParseError
from src.judges.base import JudgeCoreOutput
from src.judges.output_parser import validate_against_rubric
from src.rubrics.models import Rubric


def _good_output(**overrides: Any) -> JudgeCoreOutput:
    data: dict[str, Any] = {
        "pillar": "factual_accuracy",
        "score": 4,
        "confidence": 0.8,
        "decision_summary": "Mostly correct, one minor unsupported detail.",
        "evidence_for_score": [],
        "failure_tags": ["unsupported_claim"],
        "why_not_higher": "One minor unsupported detail.",
        "why_not_lower": "Core answer is still accurate.",
        "rubric_anchor": 4,
    }
    data.update(overrides)
    return JudgeCoreOutput(**data)


def test_passes_when_output_matches_rubric(factual_accuracy_rubric: Rubric) -> None:
    out = _good_output()
    validated = validate_against_rubric(out, rubric=factual_accuracy_rubric)
    assert validated is out


def test_rejects_pillar_mismatch(factual_accuracy_rubric: Rubric) -> None:
    out = _good_output(pillar="hallucination")
    with pytest.raises(JudgeOutputParseError) as exc:
        validate_against_rubric(out, rubric=factual_accuracy_rubric)
    reasons = {f.reason for f in exc.value.failures}
    assert "pillar_mismatch" in reasons


def test_rejects_unknown_failure_tag(factual_accuracy_rubric: Rubric) -> None:
    out = _good_output(failure_tags=["made_up_tag"])
    with pytest.raises(JudgeOutputParseError) as exc:
        validate_against_rubric(out, rubric=factual_accuracy_rubric)
    reasons = {f.reason for f in exc.value.failures}
    assert "unknown_failure_tag" in reasons
    # Detail lists the offending tag.
    unknown = next(f for f in exc.value.failures if f.reason == "unknown_failure_tag")
    assert "made_up_tag" in unknown.details["unknown_tags"]


def test_rejects_duplicate_failure_tag(factual_accuracy_rubric: Rubric) -> None:
    out = _good_output(
        failure_tags=["unsupported_claim", "unsupported_claim"],
    )
    with pytest.raises(JudgeOutputParseError) as exc:
        validate_against_rubric(out, rubric=factual_accuracy_rubric)
    reasons = {f.reason for f in exc.value.failures}
    assert "duplicate_failure_tag" in reasons


def test_rejects_tags_when_rubric_has_empty_taxonomy() -> None:
    from src.rubrics.models import ScoreAnchor

    rubric = Rubric(
        pillar="factual_accuracy",
        version="v1",
        description="x",
        anchors=[ScoreAnchor(score=s, name=str(s), description="d") for s in range(1, 6)],
        failure_tags=[],
    )
    out = _good_output(failure_tags=["whatever"])
    with pytest.raises(JudgeOutputParseError) as exc:
        validate_against_rubric(out, rubric=rubric)
    reasons = {f.reason for f in exc.value.failures}
    assert "failure_tags_without_taxonomy" in reasons


def test_accepts_empty_failure_tags_when_rubric_has_taxonomy(
    factual_accuracy_rubric: Rubric,
) -> None:
    out = _good_output(
        score=5,
        failure_tags=[],
        why_not_higher=None,
        rubric_anchor=5,
    )
    validated = validate_against_rubric(out, rubric=factual_accuracy_rubric)
    assert validated.failure_tags == []


def test_score_out_of_range_surfaces_when_rubric_uses_narrower_scale() -> None:
    from src.rubrics.models import ScoreAnchor

    # Custom rubric that uses a narrower 2-4 sub-scale for an experiment.
    rubric = Rubric(
        pillar="factual_accuracy",
        version="v1",
        description="x",
        score_min=2,
        score_max=4,
        anchors=[
            ScoreAnchor(score=2, name="a", description="a"),
            ScoreAnchor(score=3, name="b", description="b"),
            ScoreAnchor(score=4, name="c", description="c"),
        ],
    )
    # JudgeCoreOutput enforces the project-wide 1-5 scale at the schema
    # level. Rubric-level narrowing is what output_parser catches.
    out = _good_output(score=5, failure_tags=[], why_not_higher=None, rubric_anchor=4)
    with pytest.raises(JudgeOutputParseError) as exc:
        validate_against_rubric(out, rubric=rubric)
    reasons = {f.reason for f in exc.value.failures}
    assert "score_out_of_range" in reasons


def test_multiple_failures_are_reported_together(
    factual_accuracy_rubric: Rubric,
) -> None:
    out = _good_output(pillar="hallucination", failure_tags=["mystery"])
    with pytest.raises(JudgeOutputParseError) as exc:
        validate_against_rubric(out, rubric=factual_accuracy_rubric)
    reasons = {f.reason for f in exc.value.failures}
    assert "pillar_mismatch" in reasons
    assert "unknown_failure_tag" in reasons
