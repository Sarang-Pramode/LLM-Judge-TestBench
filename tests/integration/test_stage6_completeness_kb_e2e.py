"""Stage 6 end-to-end: load the shipped seed KB and score every row of
``data/samples/full_schema_sample.csv`` with the KB-aware completeness
judge.

All four sample rows are authored to match an entry in ``seed.yaml``
by intent, so we expect every one to land in ``kb_informed`` mode. A
no-match row is also scored through the same judge instance to
confirm the matcher + mode switch are per-row, not per-judge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.completeness.kb_loader import load_kb
from src.completeness.models import CompletenessKB
from src.core.types import NormalizedRow, RunContext
from src.ingestion import auto_suggest_mapping, load_file, normalize_rows
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
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
JUDGES_CONFIG_DIR = REPO_ROOT / "configs" / "judges"
RUBRICS_CONFIG_DIR = REPO_ROOT / "configs" / "rubrics"
SEED_KB_PATH = REPO_ROOT / "configs" / "completeness_kb" / "seed.yaml"


@pytest.fixture
def seed_kb() -> CompletenessKB:
    return load_kb(SEED_KB_PATH)


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(
            f"Sample artifact missing at {path}; run "
            "`python scripts/generate_sample_data.py` to regenerate."
        )


def _load_full_schema_rows() -> list[NormalizedRow]:
    path = SAMPLES_DIR / "full_schema_sample.csv"
    _require(path)
    loaded = load_file(path)
    suggested = auto_suggest_mapping(loaded.source_columns)
    norm = normalize_rows(loaded.rows, mapping=suggested.mappings)
    assert norm.failure_count == 0, norm.failures
    return norm.rows


def test_seed_kb_fingerprint_is_deterministic(seed_kb: CompletenessKB) -> None:
    """Re-loading the KB must yield an identical fingerprint."""
    again = load_kb(SEED_KB_PATH)
    assert seed_kb.fingerprint() == again.fingerprint()


def test_all_sample_rows_resolve_to_kb_informed_mode(
    all_pillars_registered: None,
    seed_kb: CompletenessKB,
) -> None:
    """The sample dataset's intents are intentionally all covered by the
    shipped seed KB. Every row should therefore hit ``kb_informed``.
    """
    rows = _load_full_schema_rows()
    assert len(rows) == 4

    run_ctx = RunContext(
        run_id="stage6-e2e",
        dataset_fingerprint="sha256:full-schema-sample-v1",
        kb_version=seed_kb.fingerprint(),
        model_alias="mock-judge",
    )

    bundle = load_judge_bundle(
        JUDGES_CONFIG_DIR / "completeness.yaml",
        rubric_root=RUBRICS_CONFIG_DIR,
    )

    # One MockLLMClient with a script of four outputs, one per row. Each
    # row is matched to its intent-specific KB entry; we mirror the
    # entry's required_elements in the scripted output.
    scripted = [_kb_core_for_row(row, seed_kb) for row in rows]
    llm = MockLLMClient(model_name="mock-completeness", structured_script=scripted)

    judge = build_judge("completeness", bundle=bundle, llm=llm, kb=seed_kb)
    assert isinstance(judge, CompletenessJudge)

    expected_kb_ids = {
        "FS-0001": "cmp_transaction_dispute_001",
        "FS-0002": "cmp_apr_lookup_001",
        "FS-0003": "cmp_fraud_protection_001",
        "FS-0004": "cmp_policy_lookup_refunds_001",
    }

    for row in rows:
        outcome = judge.run(row, run_context=run_ctx)
        assert outcome.ok, f"row {row.record_id} failed: {outcome.error}"
        assert outcome.extras[COMPLETENESS_MODE_EXTRA_KEY] == COMPLETENESS_MODE_KB_INFORMED
        assert outcome.extras["kb_match"] == "hit"
        assert outcome.extras["kb_id"] == expected_kb_ids[row.record_id]
        assert float(outcome.extras["kb_match_confidence"]) >= 0.5

        assert outcome.result is not None
        # Either required elements were returned in present or missing.
        assert outcome.result.elements_present or outcome.result.elements_missing


def test_no_match_row_falls_back_on_same_judge_instance(
    all_pillars_registered: None,
    seed_kb: CompletenessKB,
) -> None:
    """Row whose intent doesn't match any KB entry must run in fallback,
    even if the same judge just served KB-informed calls.
    """
    bundle = load_judge_bundle(
        JUDGES_CONFIG_DIR / "completeness.yaml",
        rubric_root=RUBRICS_CONFIG_DIR,
    )

    dispute_row = NormalizedRow(
        record_id="R-DISPUTE",
        user_input="How do I dispute a transaction on my card?",
        agent_output="Open a dispute from Card > Recent Transactions > Dispute.",
        category="disputes",
        intent="transaction_dispute",
        topic="disputes",
    )
    weather_row = NormalizedRow(
        record_id="R-WEATHER",
        user_input="What is the weather today?",
        agent_output="I do not have that information.",
        category="smalltalk",
        intent="weather_lookup",
        topic="weather",
    )

    llm = MockLLMClient(
        structured_script=[
            _kb_core(
                present=["steps to initiate a dispute"],
                missing=[
                    "direct acknowledgement of the user's question",
                    "expected timeline or resolution window",
                    "information or documents to provide",
                    "escalation or help path",
                ],
            ),
            _fallback_core(),
        ]
    )
    judge = build_judge("completeness", bundle=bundle, llm=llm, kb=seed_kb)

    run_ctx = RunContext(
        run_id="stage6-mixed",
        dataset_fingerprint="sha256:stage6-mixed",
        kb_version=seed_kb.fingerprint(),
    )

    first = judge.run(dispute_row, run_context=run_ctx)
    second = judge.run(weather_row, run_context=run_ctx)

    assert first.ok and second.ok
    assert first.extras[COMPLETENESS_MODE_EXTRA_KEY] == COMPLETENESS_MODE_KB_INFORMED
    assert second.extras[COMPLETENESS_MODE_EXTRA_KEY] == COMPLETENESS_MODE_GENERIC_FALLBACK
    assert "kb_id" not in second.extras


def test_run_context_can_carry_kb_fingerprint(seed_kb: CompletenessKB) -> None:
    """`RunContext.kb_version` accepts the fingerprint string."""
    ctx = RunContext(
        run_id="x",
        dataset_fingerprint="sha256:x",
        kb_version=seed_kb.fingerprint(),
    )
    assert ctx.kb_version is not None
    assert ctx.kb_version.startswith(seed_kb.version + ":")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kb_core_for_row(row: NormalizedRow, kb: CompletenessKB) -> JudgeCoreOutput:
    """Build a plausible KB-informed core output that references the
    required_elements of the KB entry that *should* match ``row``.
    """
    matching_entry = next(
        (e for e in kb.entries if e.intent == row.intent),
        None,
    )
    if matching_entry is None:  # pragma: no cover - test sanity check
        raise AssertionError(f"No KB entry matches intent {row.intent!r}")
    required = list(matching_entry.required_elements)
    # Mark the first required element as present, the rest as missing so
    # the invariant "present ∩ missing = ∅" is trivially satisfied.
    present = required[:1]
    missing = required[1:]
    return _kb_core(present=present, missing=missing)


def _kb_core(*, present: list[str], missing: list[str]) -> JudgeCoreOutput:
    score = 3 if missing else 5
    payload: dict[str, Any] = {
        "pillar": "completeness",
        "score": score,
        "confidence": 0.8,
        "decision_summary": "KB-informed completeness judgement.",
        "evidence_for_score": [],
        "failure_tags": ["missing_required_step"] if missing else [],
        "why_not_higher": "Missing elements listed." if missing else None,
        "why_not_lower": "Key steps remain present.",
        "rubric_anchor": score,
        "elements_present": present,
        "elements_missing": missing,
    }
    return JudgeCoreOutput(**payload)


def _fallback_core() -> JudgeCoreOutput:
    return JudgeCoreOutput(
        pillar="completeness",
        score=4,
        confidence=0.7,
        decision_summary="Generic completeness assessment.",
        evidence_for_score=[],
        failure_tags=["missing_timeline_expectation"],
        why_not_higher="Timeline expectation is a bit vague.",
        why_not_lower="Addresses the core question.",
        rubric_anchor=4,
    )
