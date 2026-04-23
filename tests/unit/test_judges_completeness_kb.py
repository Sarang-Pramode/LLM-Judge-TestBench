"""Unit tests for the Stage-6 KB-aware :class:`CompletenessJudge`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.completeness.kb_matcher import KBMatcher, MatchResult
from src.completeness.models import CompletenessEntry, CompletenessKB
from src.core.types import NormalizedRow, RunContext
from src.judges import (
    COMPLETENESS_MODE_EXTRA_KEY,
    COMPLETENESS_MODE_GENERIC_FALLBACK,
    COMPLETENESS_MODE_KB_INFORMED,
    CompletenessJudge,
    JudgeCoreOutput,
    build_judge,
    load_judge_bundle,
)
from src.llm.mock_client import MockLLMClient

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGES_CONFIG_DIR = REPO_ROOT / "configs" / "judges"
RUBRICS_CONFIG_DIR = REPO_ROOT / "configs" / "rubrics"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dispute_row() -> NormalizedRow:
    return NormalizedRow(
        record_id="R-DISPUTE",
        user_input="How do I dispute a transaction on my card?",
        agent_output=(
            "You can open a dispute from Card > Recent Transactions > Dispute. "
            "Most disputes resolve within 10 business days."
        ),
        category="disputes",
        intent="transaction_dispute",
        topic="disputes",
    )


@pytest.fixture
def unrelated_row() -> NormalizedRow:
    return NormalizedRow(
        record_id="R-WEATHER",
        user_input="What is the weather today?",
        agent_output="I do not have that information.",
        category="smalltalk",
        intent="weather_lookup",
        topic="weather",
    )


@pytest.fixture
def run_context() -> RunContext:
    return RunContext(
        run_id="run-kb-test",
        dataset_fingerprint="sha256:kb-test",
        kb_version="seed.v1",
    )


@pytest.fixture
def dispute_kb() -> CompletenessKB:
    return CompletenessKB(
        version="seed.test",
        entries=[
            CompletenessEntry.model_validate(
                {
                    "kb_id": "cmp_dispute",
                    "question_or_utterance_pattern": "How do I dispute a transaction?",
                    "topic_list": ["disputes", "transactions"],
                    "intent": "transaction_dispute",
                    "example_agent_response": "Dispute via the app.",
                    "completeness_notes": (
                        "Explain eligibility, steps, timeline, required info, and escalation path."
                    ),
                    "required_elements": [
                        "dispute initiation steps",
                        "expected timeline",
                        "required information to provide",
                        "escalation path",
                    ],
                    "optional_elements": ["caveat about pending charges"],
                    "forbidden_elements": ["promising guaranteed refund"],
                    "priority_level": "high",
                    "domain": "consumer_banking",
                }
            )
        ],
    )


# ---------------------------------------------------------------------------
# Mode resolution + prompt contents
# ---------------------------------------------------------------------------


def _build_completeness_judge(
    llm: MockLLMClient,
    kb: CompletenessKB | None = None,
) -> CompletenessJudge:
    bundle = load_judge_bundle(
        JUDGES_CONFIG_DIR / "completeness.yaml",
        rubric_root=RUBRICS_CONFIG_DIR,
    )
    judge = build_judge("completeness", bundle=bundle, llm=llm, kb=kb)
    assert isinstance(judge, CompletenessJudge)
    return judge


@pytest.fixture(autouse=True)
def _register_pillars(all_pillars_registered: None) -> None:
    """All tests in this module need the production judges registered."""
    return None


def test_kb_informed_mode_when_row_matches(
    dispute_row: NormalizedRow,
    dispute_kb: CompletenessKB,
    run_context: RunContext,
) -> None:
    core = _kb_completeness_core(
        present=["dispute initiation steps", "expected timeline"],
        missing=["required information to provide", "escalation path"],
    )
    llm = MockLLMClient(structured_script=[core])
    judge = _build_completeness_judge(llm, kb=dispute_kb)

    outcome = judge.run(dispute_row, run_context=run_context)

    assert outcome.ok, outcome.error
    assert outcome.extras[COMPLETENESS_MODE_EXTRA_KEY] == COMPLETENESS_MODE_KB_INFORMED
    assert outcome.extras["kb_match"] == "hit"
    assert outcome.extras["kb_id"] == "cmp_dispute"
    assert float(outcome.extras["kb_match_confidence"]) >= 0.5

    assert outcome.result is not None
    assert outcome.result.elements_present == [
        "dispute initiation steps",
        "expected timeline",
    ]
    assert outcome.result.elements_missing == [
        "required information to provide",
        "escalation path",
    ]


def test_fallback_mode_when_no_kb_wired(
    dispute_row: NormalizedRow,
    run_context: RunContext,
) -> None:
    llm = MockLLMClient(structured_script=[_fallback_completeness_core()])
    judge = _build_completeness_judge(llm, kb=None)

    outcome = judge.run(dispute_row, run_context=run_context)

    assert outcome.ok
    assert outcome.extras[COMPLETENESS_MODE_EXTRA_KEY] == COMPLETENESS_MODE_GENERIC_FALLBACK
    assert outcome.extras["kb_match"] == "none"
    assert "kb_id" not in outcome.extras
    # kb_match_confidence is still emitted so dashboards can show 0.0.
    assert float(outcome.extras["kb_match_confidence"]) == 0.0

    assert outcome.result is not None
    assert outcome.result.elements_present == []
    assert outcome.result.elements_missing == []


def test_fallback_mode_when_kb_wired_but_no_row_match(
    unrelated_row: NormalizedRow,
    dispute_kb: CompletenessKB,
    run_context: RunContext,
) -> None:
    llm = MockLLMClient(structured_script=[_fallback_completeness_core()])
    judge = _build_completeness_judge(llm, kb=dispute_kb)

    outcome = judge.run(unrelated_row, run_context=run_context)

    assert outcome.ok
    assert outcome.extras[COMPLETENESS_MODE_EXTRA_KEY] == COMPLETENESS_MODE_GENERIC_FALLBACK
    assert outcome.extras["kb_match"] == "none"
    assert "kb_id" not in outcome.extras


def test_prompt_contains_task_profile_block_in_kb_mode(
    dispute_row: NormalizedRow,
    dispute_kb: CompletenessKB,
    run_context: RunContext,
) -> None:
    llm = MockLLMClient(
        structured_script=[
            _kb_completeness_core(
                present=["dispute initiation steps"],
                missing=[
                    "expected timeline",
                    "required information to provide",
                    "escalation path",
                ],
            )
        ]
    )
    judge = _build_completeness_judge(llm, kb=dispute_kb)
    judge.run(dispute_row, run_context=run_context)

    assert len(llm.calls) == 1
    call = llm.calls[0]
    # User prompt should include the task profile block.
    assert "Task-specific completeness profile" in call.request.user_prompt
    assert "Source KB entry: cmp_dispute" in call.request.user_prompt
    assert "dispute initiation steps" in call.request.user_prompt
    # System prompt should tell the LLM to populate the element lists.
    system_prompt = call.request.system_prompt
    assert system_prompt is not None
    assert "KB_INFORMED" in system_prompt
    assert "elements_present" in system_prompt


def test_prompt_contains_fallback_addendum_when_no_match(
    unrelated_row: NormalizedRow,
    dispute_kb: CompletenessKB,
    run_context: RunContext,
) -> None:
    llm = MockLLMClient(structured_script=[_fallback_completeness_core()])
    judge = _build_completeness_judge(llm, kb=dispute_kb)
    judge.run(unrelated_row, run_context=run_context)

    call = llm.calls[0]
    assert "Task-specific completeness profile" not in call.request.user_prompt
    system_prompt = call.request.system_prompt
    assert system_prompt is not None
    assert "GENERIC_FALLBACK" in system_prompt


def test_match_cache_reused_between_prompt_and_extras(
    dispute_row: NormalizedRow,
    dispute_kb: CompletenessKB,
    run_context: RunContext,
) -> None:
    """Matcher.match should run exactly once per record_id per judge."""

    class CountingMatcher(KBMatcher):
        def __init__(self, kb: CompletenessKB) -> None:
            super().__init__(kb)
            self.call_count = 0

        def match(self, row: NormalizedRow) -> MatchResult:
            self.call_count += 1
            return super().match(row)

    matcher = CountingMatcher(dispute_kb)
    llm = MockLLMClient(
        structured_script=[
            _kb_completeness_core(
                present=["dispute initiation steps"],
                missing=[
                    "expected timeline",
                    "required information to provide",
                    "escalation path",
                ],
            )
        ]
    )
    bundle = load_judge_bundle(
        JUDGES_CONFIG_DIR / "completeness.yaml",
        rubric_root=RUBRICS_CONFIG_DIR,
    )
    judge = CompletenessJudge(
        config=bundle.config,
        rubric=bundle.rubric,
        llm=llm,
        kb=dispute_kb,
        matcher=matcher,
    )
    judge.run(dispute_row, run_context=run_context)

    assert matcher.call_count == 1


def test_build_judge_forwards_kb_only_to_completeness() -> None:
    """Non-completeness judges must not receive the KB kwarg."""
    from src.core.constants import PILLAR_FACTUAL_ACCURACY

    bundle = load_judge_bundle(
        JUDGES_CONFIG_DIR / f"{PILLAR_FACTUAL_ACCURACY}.yaml",
        rubric_root=RUBRICS_CONFIG_DIR,
    )
    llm = MockLLMClient()
    kb = CompletenessKB(version="v1", entries=[])
    judge = build_judge(PILLAR_FACTUAL_ACCURACY, bundle=bundle, llm=llm, kb=kb)
    # Would have raised TypeError if kb was forwarded to an unwilling class.
    assert judge.pillar == PILLAR_FACTUAL_ACCURACY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kb_completeness_core(
    *,
    present: list[str],
    missing: list[str],
) -> JudgeCoreOutput:
    failing = bool(missing)
    payload: dict[str, Any] = {
        "pillar": "completeness",
        "score": 3 if failing else 5,
        "confidence": 0.75,
        "decision_summary": "Partially complete against the KB task profile.",
        "evidence_for_score": [],
        "failure_tags": ["missing_required_step"] if failing else [],
        "why_not_higher": "Missing multiple required elements." if failing else None,
        "why_not_lower": "Key steps are still present.",
        "rubric_anchor": 3 if failing else 5,
        "elements_present": present,
        "elements_missing": missing,
    }
    return JudgeCoreOutput(**payload)


def _fallback_completeness_core() -> JudgeCoreOutput:
    return JudgeCoreOutput(
        pillar="completeness",
        score=4,
        confidence=0.7,
        decision_summary="Mostly complete against generic criteria.",
        evidence_for_score=[],
        failure_tags=["missing_timeline_expectation"],
        why_not_higher="Timeline expectation is a bit vague.",
        why_not_lower="Addresses the core question.",
        rubric_anchor=4,
    )
