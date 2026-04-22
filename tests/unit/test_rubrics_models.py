"""Tests for :mod:`src.rubrics.models`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.rubrics.models import ALLOWED_RUBRIC_INPUTS, Rubric, ScoreAnchor, is_known_pillar


def _anchors() -> list[ScoreAnchor]:
    return [
        ScoreAnchor(score=1, name="worst", description="awful"),
        ScoreAnchor(score=2, name="poor", description="bad"),
        ScoreAnchor(score=3, name="mixed", description="mid"),
        ScoreAnchor(score=4, name="good", description="mostly ok"),
        ScoreAnchor(score=5, name="perfect", description="great"),
    ]


def test_valid_rubric_exposes_anchor_map_and_failure_set() -> None:
    r = Rubric(
        pillar="factual_accuracy",
        version="v1.0",
        description="x",
        anchors=_anchors(),
        failure_tags=["a", "b"],
        required_inputs=["user_input", "agent_output"],
    )
    assert set(r.anchor_map.keys()) == {1, 2, 3, 4, 5}
    assert r.failure_tag_set == frozenset({"a", "b"})
    assert list(r.scores()) == [1, 2, 3, 4, 5]


def test_rubric_rejects_missing_anchor_score() -> None:
    anchors = [a for a in _anchors() if a.score != 3]
    with pytest.raises(ValidationError) as exc:
        Rubric(
            pillar="factual_accuracy",
            version="v1",
            description="x",
            anchors=anchors,
        )
    assert "missing anchors" in str(exc.value)


def test_rubric_rejects_duplicate_anchor_score() -> None:
    dupes = [*_anchors(), ScoreAnchor(score=3, name="dup", description="dup")]
    with pytest.raises(ValidationError, match="Duplicate anchor"):
        Rubric(
            pillar="factual_accuracy",
            version="v1",
            description="x",
            anchors=dupes,
        )


def test_rubric_rejects_anchor_outside_scale() -> None:
    weird = [
        ScoreAnchor(score=1, name="a", description="a"),
        ScoreAnchor(score=2, name="b", description="b"),
        ScoreAnchor(score=3, name="c", description="c"),
        ScoreAnchor(score=4, name="d", description="d"),
        ScoreAnchor(score=5, name="e", description="e"),
    ]
    with pytest.raises(ValidationError, match="outside rubric scale"):
        Rubric(
            pillar="factual_accuracy",
            version="v1",
            description="x",
            score_min=2,
            score_max=4,
            anchors=weird,  # score=1 and 5 are outside [2,4]
        )


def test_rubric_rejects_inverted_scale() -> None:
    with pytest.raises(ValidationError, match="score_min"):
        Rubric(
            pillar="factual_accuracy",
            version="v1",
            description="x",
            score_min=5,
            score_max=1,
            anchors=_anchors(),
        )


def test_rubric_rejects_uppercase_pillar() -> None:
    with pytest.raises(ValidationError, match="lowercase"):
        Rubric(
            pillar="Factual_Accuracy",
            version="v1",
            description="x",
            anchors=_anchors(),
        )


def test_rubric_rejects_unknown_required_input() -> None:
    with pytest.raises(ValidationError, match="Unknown rubric input"):
        Rubric(
            pillar="factual_accuracy",
            version="v1",
            description="x",
            anchors=_anchors(),
            required_inputs=["not_a_real_field"],
        )


def test_rubric_rejects_duplicate_required_input() -> None:
    with pytest.raises(ValidationError, match="Duplicate required_input"):
        Rubric(
            pillar="factual_accuracy",
            version="v1",
            description="x",
            anchors=_anchors(),
            required_inputs=["user_input", "user_input"],
        )


def test_rubric_rejects_padded_failure_tag() -> None:
    with pytest.raises(ValidationError, match="must be non-empty and not padded"):
        Rubric(
            pillar="factual_accuracy",
            version="v1",
            description="x",
            anchors=_anchors(),
            failure_tags=["  unsupported_claim"],
        )


def test_rubric_rejects_duplicate_failure_tag() -> None:
    with pytest.raises(ValidationError, match="Duplicate failure tag"):
        Rubric(
            pillar="factual_accuracy",
            version="v1",
            description="x",
            anchors=_anchors(),
            failure_tags=["a", "a"],
        )


def test_rubric_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Rubric.model_validate(
            {
                "pillar": "factual_accuracy",
                "version": "v1",
                "description": "x",
                "anchors": [a.model_dump() for a in _anchors()],
                "rogue_field": "nope",
            }
        )


def test_is_known_pillar_smoke() -> None:
    assert is_known_pillar("factual_accuracy")
    assert not is_known_pillar("some_custom_future_pillar")


def test_allowed_inputs_includes_core_fields() -> None:
    assert "user_input" in ALLOWED_RUBRIC_INPUTS
    assert "retrieved_context" in ALLOWED_RUBRIC_INPUTS
