"""Stage 5 end-to-end: ingest the full-schema sample and score every
row with every pillar judge through a :class:`MockLLMClient`.

This is the narrowest possible "ingestion -> judge framework" slice
and is the canonical happy-path contract the Stage 7 runner will
widen. It deliberately avoids any real provider and any real
evaluation metrics; those arrive in later stages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.constants import PILLARS
from src.core.types import RunContext
from src.ingestion import auto_suggest_mapping, load_file, normalize_rows
from src.judges import (
    COMPLETENESS_MODE_EXTRA_KEY,
    COMPLETENESS_MODE_GENERIC_FALLBACK,
    JudgeCoreOutput,
    JudgeOutcome,
    build_judge,
    load_judge_bundle,
)
from src.llm.base import LLMUsage
from src.llm.mock_client import MockLLMClient
from src.rubrics.loader import load_rubric
from src.rubrics.models import Rubric

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
JUDGES_CONFIG_DIR = REPO_ROOT / "configs" / "judges"
RUBRICS_CONFIG_DIR = REPO_ROOT / "configs" / "rubrics"


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(
            f"Sample artifact missing at {path}; run "
            "`python scripts/generate_sample_data.py` to regenerate."
        )


def test_full_schema_sample_scored_by_all_six_pillars(
    all_pillars_registered: None,
) -> None:
    """Ingest full_schema_sample.csv -> run each pillar against each row."""
    path = SAMPLES_DIR / "full_schema_sample.csv"
    _require(path)

    loaded = load_file(path)
    suggested = auto_suggest_mapping(loaded.source_columns)
    norm = normalize_rows(loaded.rows, mapping=suggested.mappings)
    assert norm.failure_count == 0, norm.failures
    rows = norm.rows
    assert len(rows) == 4

    run_ctx = RunContext(
        run_id="stage5-e2e",
        model_alias="mock-judge-default",
        dataset_fingerprint="sha256:full-schema-sample-v1",
    )

    # Pre-load every pillar bundle once - Stage 7 will do the same.
    bundles = {
        pillar: load_judge_bundle(
            JUDGES_CONFIG_DIR / f"{pillar}.yaml",
            rubric_root=RUBRICS_CONFIG_DIR,
        )
        for pillar in PILLARS
    }
    rubrics = {pillar: load_rubric(RUBRICS_CONFIG_DIR / f"{pillar}.yaml") for pillar in PILLARS}

    outcomes_by_pillar_and_row: dict[tuple[str, str], JudgeOutcome] = {}
    for row in rows:
        for pillar in PILLARS:
            bundle = bundles[pillar]
            llm = MockLLMClient(
                model_name="mock-judge-default",
                structured_script=[_plausible_output(pillar, rubrics[pillar])],
                usage=LLMUsage(input_tokens=30, output_tokens=15),
            )
            judge = build_judge(pillar, bundle=bundle, llm=llm)
            outcome = judge.run(row, run_context=run_ctx)
            outcomes_by_pillar_and_row[(pillar, row.record_id)] = outcome
            assert outcome.ok, f"{pillar} failed on {row.record_id}: {outcome.error}"

    # 4 rows x 6 pillars = 24 successful outcomes.
    assert len(outcomes_by_pillar_and_row) == 24

    # Completeness always runs in generic_fallback mode in Stage 5.
    for (pillar, _record_id), outcome in outcomes_by_pillar_and_row.items():
        if pillar == "completeness":
            assert outcome.extras[COMPLETENESS_MODE_EXTRA_KEY] == COMPLETENESS_MODE_GENERIC_FALLBACK


def _plausible_output(pillar: str, rubric: Rubric) -> JudgeCoreOutput:
    """Score 4 with a valid failure tag for the pillar's taxonomy."""
    return JudgeCoreOutput(
        pillar=pillar,
        score=4,
        confidence=0.75,
        decision_summary=f"Plausible {pillar} judgement (e2e smoke test).",
        evidence_for_score=[],
        failure_tags=[rubric.failure_tags[0]] if rubric.failure_tags else [],
        why_not_higher=f"Minor {pillar} concern.",
        why_not_lower=f"Broadly acceptable on {pillar}.",
        rubric_anchor=4,
    )
