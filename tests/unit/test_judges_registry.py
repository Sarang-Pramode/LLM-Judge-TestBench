"""Tests for :mod:`src.judges.registry`."""

from __future__ import annotations

import pytest

from src.core.exceptions import JudgeExecutionError
from src.judges.base import BaseJudge
from src.judges.config import JudgeBundle, JudgeConfig
from src.judges.registry import (
    build_judge,
    is_registered,
    list_registered_pillars,
    register_judge,
    registered_judges,
    resolve_judge,
)
from src.llm.mock_client import MockLLMClient
from src.rubrics.models import Rubric


class _FakeFAJudge(BaseJudge):
    pillar = "factual_accuracy"

    def _marker(self) -> None:  # pragma: no cover
        return None


class _FakeRelevanceJudge(BaseJudge):
    pillar = "relevance"

    def _marker(self) -> None:  # pragma: no cover
        return None


class _EmptyPillarJudge(BaseJudge):
    pillar = ""

    def _marker(self) -> None:  # pragma: no cover
        return None


def test_register_and_resolve_judge() -> None:
    register_judge(_FakeFAJudge)
    assert is_registered("factual_accuracy")
    assert resolve_judge("factual_accuracy") is _FakeFAJudge


def test_register_as_decorator_without_kwargs() -> None:
    @register_judge
    class _Dec(BaseJudge):
        pillar = "factual_accuracy"

        def _marker(self) -> None:  # pragma: no cover
            return None

    assert resolve_judge("factual_accuracy") is _Dec


def test_register_as_decorator_with_kwargs() -> None:
    register_judge(_FakeFAJudge)

    @register_judge(force=True)
    class _Override(BaseJudge):
        pillar = "factual_accuracy"

        def _marker(self) -> None:  # pragma: no cover
            return None

    assert resolve_judge("factual_accuracy") is _Override


def test_reregister_without_force_raises() -> None:
    register_judge(_FakeFAJudge)

    class _Other(BaseJudge):
        pillar = "factual_accuracy"

        def _marker(self) -> None:  # pragma: no cover
            return None

    with pytest.raises(JudgeExecutionError, match="already registered"):
        register_judge(_Other)


def test_reregister_same_class_is_idempotent() -> None:
    register_judge(_FakeFAJudge)
    register_judge(_FakeFAJudge)  # no error
    assert resolve_judge("factual_accuracy") is _FakeFAJudge


def test_register_rejects_empty_pillar() -> None:
    with pytest.raises(JudgeExecutionError, match="empty"):
        register_judge(_EmptyPillarJudge)


def test_register_rejects_non_subclass() -> None:
    class _NotAJudge:
        pillar = "factual_accuracy"

    with pytest.raises(JudgeExecutionError, match="BaseJudge subclass"):
        register_judge(_NotAJudge)  # type: ignore[arg-type]


def test_resolve_unknown_pillar_raises_with_context() -> None:
    register_judge(_FakeFAJudge)
    with pytest.raises(JudgeExecutionError, match="No judge registered"):
        resolve_judge("unknown_pillar")


def test_list_registered_pillars_is_sorted() -> None:
    register_judge(_FakeRelevanceJudge)
    register_judge(_FakeFAJudge)
    assert list_registered_pillars() == ["factual_accuracy", "relevance"]


def test_registered_judges_yields_sorted_pairs() -> None:
    register_judge(_FakeRelevanceJudge)
    register_judge(_FakeFAJudge)
    pairs = list(registered_judges())
    assert [p for p, _ in pairs] == ["factual_accuracy", "relevance"]


def test_build_judge_instantiates_via_bundle(
    factual_accuracy_rubric: Rubric,
    factual_accuracy_config: JudgeConfig,
) -> None:
    register_judge(_FakeFAJudge)
    bundle = JudgeBundle(config=factual_accuracy_config, rubric=factual_accuracy_rubric)
    judge = build_judge("factual_accuracy", bundle=bundle, llm=MockLLMClient(model_name="mock"))
    assert isinstance(judge, _FakeFAJudge)
    assert judge.pillar == "factual_accuracy"
    assert judge.rubric is factual_accuracy_rubric
    assert judge.config is factual_accuracy_config


def test_build_judge_rejects_pillar_bundle_mismatch(
    factual_accuracy_rubric: Rubric,
    factual_accuracy_config: JudgeConfig,
) -> None:
    register_judge(_FakeRelevanceJudge)
    bundle = JudgeBundle(config=factual_accuracy_config, rubric=factual_accuracy_rubric)
    with pytest.raises(JudgeExecutionError, match="declares pillar"):
        build_judge("relevance", bundle=bundle, llm=MockLLMClient(model_name="mock"))
