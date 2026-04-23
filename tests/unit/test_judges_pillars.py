"""Stage 5: per-pillar judge smoke tests.

Each of the six pillars ships:

1. A rubric YAML (``configs/rubrics/<pillar>.yaml``).
2. A judge config YAML (``configs/judges/<pillar>.yaml``).
3. A :class:`BaseJudge` subclass registered via ``@register_judge``.

These tests verify - for every pillar - that:

* The shipped YAMLs load and produce a valid :class:`JudgeBundle`.
* The judge class is registered with the correct pillar key.
* The judge runs end-to-end against a :class:`MockLLMClient` and
  returns a successful :class:`JudgeOutcome` with a well-formed
  :class:`JudgeResult`.
* Rubric cross-file invariants hold (pillar + version match).

Pillar-specific contracts (e.g. completeness mode) get their own
focused tests below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.core.constants import PILLARS
from src.core.types import NormalizedRow, RunContext
from src.judges import (
    COMPLETENESS_MODE_EXTRA_KEY,
    COMPLETENESS_MODE_GENERIC_FALLBACK,
    BaseJudge,
    BiasDiscriminationJudge,
    CompletenessJudge,
    FactualAccuracyJudge,
    HallucinationJudge,
    JudgeBundle,
    JudgeCoreOutput,
    RelevanceJudge,
    ToxicityJudge,
    build_judge,
    is_registered,
    load_judge_bundle,
    resolve_judge,
)
from src.llm.base import LLMUsage
from src.llm.mock_client import MockLLMClient
from src.rubrics.models import Rubric

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGES_CONFIG_DIR = REPO_ROOT / "configs" / "judges"
RUBRICS_CONFIG_DIR = REPO_ROOT / "configs" / "rubrics"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rich_row() -> NormalizedRow:
    """A row with every rubric-required input populated for all pillars."""
    return NormalizedRow(
        record_id="r-stage5",
        user_input="How do I dispute a transaction?",
        agent_output=(
            "Open the app, tap the transaction, and choose Dispute. Most "
            "disputes resolve within 10 business days."
        ),
        category="disputes",
        retrieved_context=[
            "Disputes can be opened from the app for any posted transaction.",
            {
                "text": "Typical resolution time is 7-10 business days.",
                "doc_id": "policy_disputes_v3",
                "score": 0.87,
            },
        ],
    )


@pytest.fixture
def run_context() -> RunContext:
    return RunContext(
        run_id="run-stage5-smoke",
        model_alias="mock-judge",
        dataset_fingerprint="sha256:stage5-smoke",
    )


# ---------------------------------------------------------------------------
# Parametrised suite: each pillar in turn
# ---------------------------------------------------------------------------


PILLAR_CLASS_MAP: dict[str, type[BaseJudge]] = {
    "factual_accuracy": FactualAccuracyJudge,
    "hallucination": HallucinationJudge,
    "relevance": RelevanceJudge,
    "completeness": CompletenessJudge,
    "toxicity": ToxicityJudge,
    "bias_discrimination": BiasDiscriminationJudge,
}


def test_pillar_class_map_covers_all_v1_pillars() -> None:
    assert set(PILLAR_CLASS_MAP) == set(PILLARS)


@pytest.mark.parametrize("pillar", PILLARS)
def test_shipped_bundle_loads_cleanly(pillar: str) -> None:
    """Each shipped rubric+config pair loads and passes cross-file checks."""
    bundle = load_judge_bundle(
        JUDGES_CONFIG_DIR / f"{pillar}.yaml",
        rubric_root=RUBRICS_CONFIG_DIR,
    )
    assert isinstance(bundle, JudgeBundle)
    assert bundle.config.pillar == pillar
    assert bundle.rubric.pillar == pillar
    assert bundle.config.rubric_version == bundle.rubric.version


@pytest.mark.parametrize("pillar", PILLARS)
def test_pillar_judge_is_registered(
    all_pillars_registered: None,
    pillar: str,
) -> None:
    assert is_registered(pillar)
    cls = resolve_judge(pillar)
    assert cls is PILLAR_CLASS_MAP[pillar]
    assert cls.pillar == pillar


@pytest.mark.parametrize("pillar", PILLARS)
def test_pillar_runs_against_mock_llm_and_returns_valid_result(
    all_pillars_registered: None,
    rich_row: NormalizedRow,
    run_context: RunContext,
    pillar: str,
) -> None:
    """End-to-end: load bundle -> build judge -> run -> inspect result."""
    bundle = load_judge_bundle(
        JUDGES_CONFIG_DIR / f"{pillar}.yaml",
        rubric_root=RUBRICS_CONFIG_DIR,
    )
    llm = MockLLMClient(
        model_name=f"mock-{pillar}-v1",
        structured_script=[_scripted_core(pillar)],
        usage=LLMUsage(input_tokens=42, output_tokens=21),
    )
    judge = build_judge(pillar, bundle=bundle, llm=llm)
    outcome = judge.run(rich_row, run_context=run_context)

    assert outcome.ok, outcome.error
    assert outcome.pillar == pillar
    assert outcome.record_id == rich_row.record_id
    assert outcome.model_name == f"mock-{pillar}-v1"
    assert outcome.run_id == run_context.run_id

    result = outcome.result
    assert result is not None
    assert result.pillar == pillar
    assert 1 <= result.score <= 5
    assert 0.0 <= result.confidence <= 1.0
    assert result.decision_summary
    assert result.prompt_version == bundle.config.prompt_version
    assert result.rubric_version == bundle.rubric.version

    # Every successful call recorded in the mock, exactly once.
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call.kind == "structured"
    assert call.request.tags.get("pillar") == pillar


# ---------------------------------------------------------------------------
# Cross-pillar orchestration smoke test
# ---------------------------------------------------------------------------


def test_all_six_pillars_run_on_the_same_row(
    all_pillars_registered: None,
    rich_row: NormalizedRow,
    run_context: RunContext,
) -> None:
    """A single row can be scored by all six pillars through the registry.

    This is the contract orchestration (Stage 7) will rely on: the
    runner iterates ``PILLARS``, builds judges via the registry, and
    runs each against the same row. Here we validate that each judge
    is independently wired and produces a distinct result.
    """
    outcomes = {}
    for pillar in PILLARS:
        bundle = load_judge_bundle(
            JUDGES_CONFIG_DIR / f"{pillar}.yaml",
            rubric_root=RUBRICS_CONFIG_DIR,
        )
        llm = MockLLMClient(
            model_name="mock-judge-default",
            structured_script=[_scripted_core(pillar)],
        )
        judge = build_judge(pillar, bundle=bundle, llm=llm)
        outcomes[pillar] = judge.run(rich_row, run_context=run_context)

    assert set(outcomes) == set(PILLARS)
    for pillar, outcome in outcomes.items():
        assert outcome.ok, f"{pillar} failed: {outcome.error}"
        assert outcome.result is not None
        assert outcome.result.pillar == pillar


# ---------------------------------------------------------------------------
# Completeness mode contract (Stage 5 ships generic_fallback only)
# ---------------------------------------------------------------------------


def test_completeness_judge_reports_generic_fallback_mode(
    all_pillars_registered: None,
    rich_row: NormalizedRow,
    run_context: RunContext,
) -> None:
    """Until Stage 6 wires the KB, completeness always reports fallback."""
    bundle = load_judge_bundle(
        JUDGES_CONFIG_DIR / "completeness.yaml",
        rubric_root=RUBRICS_CONFIG_DIR,
    )
    llm = MockLLMClient(structured_script=[_scripted_core("completeness")])
    judge = build_judge("completeness", bundle=bundle, llm=llm)
    outcome = judge.run(rich_row, run_context=run_context)

    assert outcome.ok
    assert outcome.extras[COMPLETENESS_MODE_EXTRA_KEY] == COMPLETENESS_MODE_GENERIC_FALLBACK
    assert outcome.extras["kb_match"] == "none"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scripted_core(pillar: str, **overrides: Any) -> JudgeCoreOutput:
    """Build a plausible :class:`JudgeCoreOutput` for any pillar.

    The rubric-level output parser checks pillar + failure_tags
    membership, so every pillar needs a payload whose ``failure_tags``
    are valid for its own taxonomy. We pick a valid tag from each
    rubric or leave the list empty for the top-score case.
    """
    payload: dict[str, Any] = {
        "pillar": pillar,
        "score": 4,
        "confidence": 0.8,
        "decision_summary": f"Plausible {pillar} judgement for smoke tests.",
        "evidence_for_score": [],
        "failure_tags": _valid_failure_tags(pillar),
        "why_not_higher": f"Minor issue on the {pillar} axis.",
        "why_not_lower": f"Broadly acceptable on the {pillar} axis.",
        "rubric_anchor": 4,
    }
    payload.update(overrides)
    return JudgeCoreOutput(**payload)


def _valid_failure_tags(pillar: str) -> list[str]:
    """Pick a single valid failure tag from the pillar's rubric."""
    rubric = Rubric.model_validate(
        load_judge_bundle(
            JUDGES_CONFIG_DIR / f"{pillar}.yaml",
            rubric_root=RUBRICS_CONFIG_DIR,
        ).rubric.model_dump()
    )
    return [rubric.failure_tags[0]] if rubric.failure_tags else []
