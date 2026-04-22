"""End-to-end tests for :class:`BaseJudge` using :class:`MockLLMClient`."""

from __future__ import annotations

from typing import Any

import pytest

from src.core.exceptions import (
    JudgeExecutionError,
    ProviderError,
    ProviderRateLimitError,
)
from src.core.types import NormalizedRow, RunContext
from src.judges.base import BaseJudge, JudgeCoreOutput
from src.judges.config import JudgeConfig
from src.llm.base import LLMUsage
from src.llm.mock_client import MockLLMClient
from src.rubrics.models import Rubric


class FAJudge(BaseJudge):
    """Concrete fake for the factual_accuracy pillar."""

    pillar = "factual_accuracy"

    def _marker(self) -> None:  # pragma: no cover
        return None


# Shortcut for building rows that satisfy factual_accuracy rubric reqs.
def _row(**overrides: Any) -> NormalizedRow:
    defaults: dict[str, Any] = {
        "record_id": "r-001",
        "user_input": "How do I dispute a transaction?",
        "agent_output": "Open the app, tap the charge, then choose Dispute.",
        "category": "transactions",
        "retrieved_context": [
            "To dispute a transaction, follow the in-app flow.",
        ],
    }
    defaults.update(overrides)
    return NormalizedRow(**defaults)


def _good_core(**overrides: Any) -> JudgeCoreOutput:
    data: dict[str, Any] = {
        "pillar": "factual_accuracy",
        "score": 4,
        "confidence": 0.82,
        "decision_summary": "Mostly correct with one small unsupported detail.",
        "evidence_for_score": [],
        "failure_tags": ["unsupported_claim"],
        "why_not_higher": "Contains one unsupported material detail.",
        "why_not_lower": "Core answer remains mostly correct and useful.",
        "rubric_anchor": 4,
    }
    data.update(overrides)
    return JudgeCoreOutput(**data)


# ---------------------------------------------------------------------------
# Construction / wiring validation
# ---------------------------------------------------------------------------


def test_base_judge_rejects_pillar_mismatch_in_config(
    factual_accuracy_rubric: Rubric,
) -> None:
    bad_cfg = JudgeConfig(
        pillar="relevance",
        prompt_version="v1",
        rubric_path="x",
        rubric_version="v1.0",
    )
    with pytest.raises(JudgeExecutionError, match="JudgeConfig pillar"):
        FAJudge(
            config=bad_cfg,
            rubric=factual_accuracy_rubric,
            llm=MockLLMClient(model_name="mock"),
        )


def test_base_judge_rejects_pillar_mismatch_in_rubric(
    factual_accuracy_config: JudgeConfig,
    factual_accuracy_rubric: Rubric,
) -> None:
    other = factual_accuracy_rubric.model_copy(update={"pillar": "relevance"})
    with pytest.raises(JudgeExecutionError, match="Rubric pillar"):
        FAJudge(
            config=factual_accuracy_config,
            rubric=other,
            llm=MockLLMClient(model_name="mock"),
        )


def test_base_judge_rejects_empty_pillar_class(
    factual_accuracy_rubric: Rubric,
    factual_accuracy_config: JudgeConfig,
) -> None:
    class _NoPillar(BaseJudge):
        pillar = ""

        def _marker(self) -> None:  # pragma: no cover
            return None

    with pytest.raises(JudgeExecutionError, match="pillar"):
        _NoPillar(
            config=factual_accuracy_config,
            rubric=factual_accuracy_rubric,
            llm=MockLLMClient(model_name="mock"),
        )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_happy_path_returns_judge_result(
    factual_accuracy_rubric: Rubric,
    factual_accuracy_config: JudgeConfig,
    run_context: RunContext,
) -> None:
    core = _good_core()
    llm = MockLLMClient(
        model_name="mock-model-v0",
        structured_script=[core],
        usage=LLMUsage(input_tokens=42, output_tokens=17, total_tokens=59),
    )
    judge = FAJudge(config=factual_accuracy_config, rubric=factual_accuracy_rubric, llm=llm)
    outcome = judge.run(_row(), run_context=run_context)

    assert outcome.ok
    assert outcome.error is None
    assert outcome.result is not None
    assert outcome.result.pillar == "factual_accuracy"
    assert outcome.result.score == 4
    # Metadata was injected by the judge, not the LLM.
    assert outcome.result.raw_model_name == "mock-model-v0"
    assert outcome.result.prompt_version == "factual_accuracy.v1"
    assert outcome.result.rubric_version == "v1.0"
    # Bookkeeping on the outcome itself.
    assert outcome.usage.total_tokens == 59
    assert outcome.model_name == "mock-model-v0"
    assert outcome.run_id == run_context.run_id
    assert outcome.prompt_versions == ("factual_accuracy.v1", "v1.0")


def test_run_forwards_prompt_and_request_fields(
    factual_accuracy_rubric: Rubric,
    factual_accuracy_config: JudgeConfig,
    run_context: RunContext,
) -> None:
    captured: dict[str, Any] = {}

    def _structured_fn(request: Any, schema: type[Any]) -> JudgeCoreOutput:
        captured["request"] = request
        captured["schema"] = schema
        return _good_core()

    llm = MockLLMClient(
        model_name="mock-model-v0",
        structured_fn=_structured_fn,
    )
    judge = FAJudge(config=factual_accuracy_config, rubric=factual_accuracy_rubric, llm=llm)
    outcome = judge.run(_row(), run_context=run_context)
    assert outcome.ok

    # The judge passed its rubric-aware prompt to the LLM.
    req = captured["request"]
    assert "factual_accuracy" in req.system_prompt
    assert "How do I dispute a transaction?" in req.user_prompt
    # Retry/temperature/timeout flowed from config.
    assert req.temperature == 0.0
    assert req.timeout_s == factual_accuracy_config.timeout_s
    assert req.retry.max_attempts == factual_accuracy_config.retry.max_attempts
    # Tags include pillar + record_id + run_id.
    assert req.tags["pillar"] == "factual_accuracy"
    assert req.tags["record_id"] == "r-001"
    assert req.tags["run_id"] == run_context.run_id
    # Schema forwarded for structured output.
    assert captured["schema"] is JudgeCoreOutput


# ---------------------------------------------------------------------------
# Required-input validation
# ---------------------------------------------------------------------------


def test_run_raises_when_rubric_required_input_missing(
    factual_accuracy_rubric: Rubric,
    factual_accuracy_config: JudgeConfig,
    run_context: RunContext,
) -> None:
    llm = MockLLMClient(model_name="mock", structured_script=[_good_core()])
    judge = FAJudge(config=factual_accuracy_config, rubric=factual_accuracy_rubric, llm=llm)
    row_without_ctx = _row(retrieved_context=None)
    with pytest.raises(JudgeExecutionError, match="rubric-required input"):
        judge.run(row_without_ctx, run_context=run_context)


# ---------------------------------------------------------------------------
# Failure modes captured into outcome
# ---------------------------------------------------------------------------


def test_provider_error_is_captured_as_outcome(
    factual_accuracy_rubric: Rubric,
    factual_accuracy_config: JudgeConfig,
    run_context: RunContext,
) -> None:
    llm = MockLLMClient(
        model_name="mock",
        structured_script=[
            ProviderRateLimitError("429"),
            ProviderRateLimitError("429"),
            ProviderRateLimitError("429"),
        ],
    )
    judge = FAJudge(config=factual_accuracy_config, rubric=factual_accuracy_rubric, llm=llm)
    outcome = judge.run(_row(), run_context=run_context)
    assert not outcome.ok
    assert outcome.result is None
    assert outcome.error_type == "ProviderRateLimitError"
    assert outcome.error is not None


def test_generic_provider_error_short_circuits(
    factual_accuracy_rubric: Rubric,
    factual_accuracy_config: JudgeConfig,
    run_context: RunContext,
) -> None:
    llm = MockLLMClient(
        model_name="mock",
        structured_script=[ProviderError("nope")],
    )
    judge = FAJudge(config=factual_accuracy_config, rubric=factual_accuracy_rubric, llm=llm)
    outcome = judge.run(_row(), run_context=run_context)
    assert not outcome.ok
    assert outcome.error_type == "ProviderError"


def test_rubric_parse_failure_is_captured(
    factual_accuracy_rubric: Rubric,
    factual_accuracy_config: JudgeConfig,
    run_context: RunContext,
) -> None:
    # Core output claims a failure tag the rubric doesn't allow.
    bogus = _good_core(failure_tags=["tag_not_in_rubric"])
    llm = MockLLMClient(model_name="mock", structured_script=[bogus])
    judge = FAJudge(config=factual_accuracy_config, rubric=factual_accuracy_rubric, llm=llm)
    outcome = judge.run(_row(), run_context=run_context)
    assert not outcome.ok
    assert outcome.error_type == "JudgeOutputParseError"


def test_outcome_as_dict_is_json_shaped(
    factual_accuracy_rubric: Rubric,
    factual_accuracy_config: JudgeConfig,
    run_context: RunContext,
) -> None:
    llm = MockLLMClient(
        model_name="mock",
        structured_script=[_good_core()],
        usage=LLMUsage(input_tokens=1, output_tokens=2, total_tokens=3),
    )
    judge = FAJudge(config=factual_accuracy_config, rubric=factual_accuracy_rubric, llm=llm)
    outcome = judge.run(_row(), run_context=run_context)
    d = outcome.as_dict()
    assert d["ok"] is True
    assert d["pillar"] == "factual_accuracy"
    assert d["usage"] == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    assert d["result"] is not None


# ---------------------------------------------------------------------------
# Extensibility
# ---------------------------------------------------------------------------


def test_subclass_can_override_extra_outcome_fields(
    factual_accuracy_rubric: Rubric,
    factual_accuracy_config: JudgeConfig,
    run_context: RunContext,
) -> None:
    class _WithExtras(BaseJudge):
        pillar = "factual_accuracy"

        def _marker(self) -> None:  # pragma: no cover
            return None

        def extra_outcome_fields(
            self, *, row: NormalizedRow, run_context: RunContext
        ) -> dict[str, str]:
            return {"category": row.category, "run_id": run_context.run_id}

    llm = MockLLMClient(model_name="mock", structured_script=[_good_core()])
    judge = _WithExtras(config=factual_accuracy_config, rubric=factual_accuracy_rubric, llm=llm)
    outcome = judge.run(_row(), run_context=run_context)
    assert outcome.ok
    assert outcome.extras["category"] == "transactions"
    assert outcome.extras["run_id"] == run_context.run_id
