"""Stage 7 end-to-end: drive the shipped sample dataset through every
pillar via :class:`EvaluationRunner` under concurrent workers.

Goals validated here:

- Runner orchestrates all six pillars and both dataset rows without
  requiring callers to know anything about threading, rate limiting,
  or caching.
- Completeness pillar still picks up the KB when one is supplied.
- Deterministic output ordering regardless of thread interleaving.
- Cache hits make a re-run free (no new LLM calls).
- Token usage is aggregated in the run summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.completeness.kb_loader import load_kb
from src.completeness.models import CompletenessKB
from src.core.constants import PILLARS
from src.core.types import NormalizedRow, RunContext
from src.ingestion import auto_suggest_mapping, load_file, normalize_rows
from src.judges import (
    COMPLETENESS_MODE_EXTRA_KEY,
    COMPLETENESS_MODE_KB_INFORMED,
    JudgeCoreOutput,
    load_judge_bundle,
)
from src.llm.base import LLMUsage
from src.llm.mock_client import MockLLMClient
from src.orchestration import (
    ConcurrencyPolicy,
    EvaluationRunner,
    InMemoryOutcomeCache,
    RunPlan,
)
from src.rubrics.loader import load_rubric
from src.rubrics.models import Rubric

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
JUDGES_DIR = REPO_ROOT / "configs" / "judges"
RUBRICS_DIR = REPO_ROOT / "configs" / "rubrics"
SEED_KB = REPO_ROOT / "configs" / "completeness_kb" / "seed.yaml"


@pytest.fixture
def seed_kb() -> CompletenessKB:
    if not SEED_KB.exists():
        pytest.skip(f"Seed KB missing at {SEED_KB}")
    return load_kb(SEED_KB)


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Sample artifact missing at {path}; run scripts/generate_sample_data.py")


def _load_rows() -> list[NormalizedRow]:
    path = SAMPLES_DIR / "full_schema_sample.csv"
    _require(path)
    loaded = load_file(path)
    suggested = auto_suggest_mapping(loaded.source_columns)
    norm = normalize_rows(loaded.rows, mapping=suggested.mappings)
    assert norm.failure_count == 0, norm.failures
    return norm.rows


def _plausible_output(
    pillar: str, rubric: Rubric, row: NormalizedRow, kb: CompletenessKB
) -> JudgeCoreOutput:
    """Mirror Stage 5's pattern but tie completeness to the KB entry.

    For the completeness pillar we pretend the judge "read" the KB
    entry and reported one required_element as present, the rest as
    missing. That keeps the KB-informed branch honest without assuming
    anything about the row's real content.
    """
    payload: dict[str, Any] = {
        "pillar": pillar,
        "score": 4,
        "confidence": 0.75,
        "decision_summary": f"Plausible {pillar} judgement (stage 7 runner test).",
        "evidence_for_score": [],
        "failure_tags": [rubric.failure_tags[0]] if rubric.failure_tags else [],
        "why_not_higher": f"Minor {pillar} concern.",
        "why_not_lower": f"Broadly acceptable on {pillar}.",
        "rubric_anchor": 4,
    }
    if pillar == "completeness":
        entry = next((e for e in kb.entries if e.intent == row.intent), None)
        if entry is not None and entry.required_elements:
            payload["elements_present"] = entry.required_elements[:1]
            payload["elements_missing"] = entry.required_elements[1:]
    return JudgeCoreOutput(**payload)


def _build_scripts(
    rows: list[NormalizedRow],
    rubrics: dict[str, Rubric],
    kb: CompletenessKB,
) -> dict[str, list[JudgeCoreOutput]]:
    """Pre-script one response per (row, pillar) task.

    Because the runner may fan out across threads, the mock's internal
    lock guarantees each script entry is consumed exactly once. The
    script queue ends up in pop order equal to some permutation of
    ``(row, pillar)`` tasks, so entries must all be valid for the
    corresponding pillar. We therefore generate a script per pillar.
    """
    return {
        pillar: [_plausible_output(pillar, rubrics[pillar], row, kb) for row in rows]
        for pillar in PILLARS
    }


def test_runner_scores_all_six_pillars_in_parallel(
    all_pillars_registered: None,
    seed_kb: CompletenessKB,
) -> None:
    rows = _load_rows()
    assert len(rows) == 4
    rubrics = {p: load_rubric(RUBRICS_DIR / f"{p}.yaml") for p in PILLARS}
    bundles = {
        p: load_judge_bundle(JUDGES_DIR / f"{p}.yaml", rubric_root=RUBRICS_DIR) for p in PILLARS
    }
    scripts = _build_scripts(rows, rubrics, seed_kb)
    llms = {
        p: MockLLMClient(
            model_name=f"mock-{p}",
            structured_script=scripts[p],
            usage=LLMUsage(input_tokens=40, output_tokens=20),
        )
        for p in PILLARS
    }

    run_ctx = RunContext(
        run_id="stage7-e2e",
        dataset_fingerprint="sha256:full-schema-sample-v1",
        kb_version=seed_kb.fingerprint(),
        model_alias="mock-judge",
    )

    plan = RunPlan(
        rows=rows,
        pillars=list(PILLARS),
        bundles=bundles,
        llm_by_pillar=llms,
        run_context=run_ctx,
        kb=seed_kb,
        concurrency=ConcurrencyPolicy(max_workers=6),
    )

    result = EvaluationRunner().run(plan)

    expected_total = len(rows) * len(PILLARS)
    assert result.summary.total_tasks == expected_total
    assert result.summary.succeeded == expected_total
    assert result.summary.failed == 0
    # 6 pillars x 4 rows x 20 output tokens = 480.
    assert result.summary.total_output_tokens == expected_total * 20
    assert result.summary.total_input_tokens == expected_total * 40

    # Deterministic ordering: sort key is (row_index, pillar).
    expected_order = [(row.record_id, pillar) for row in rows for pillar in sorted(PILLARS)]
    got_order = [(o.record_id, o.pillar) for o in result.outcomes]
    # PILLARS in constants may not be alphabetical, so we re-sort locally:
    by_row: dict[int, list[tuple[str, str]]] = {}
    for idx, row in enumerate(rows):
        by_row[idx] = sorted([(row.record_id, p) for p in PILLARS], key=lambda pair: pair[1])
    flat = [pair for idx in sorted(by_row) for pair in by_row[idx]]
    assert got_order == flat == expected_order

    # Completeness outcomes all in KB-informed mode.
    completeness = [o for o in result.outcomes if o.pillar == "completeness"]
    assert len(completeness) == len(rows)
    for outcome in completeness:
        assert outcome.extras[COMPLETENESS_MODE_EXTRA_KEY] == COMPLETENESS_MODE_KB_INFORMED


def test_second_run_hits_the_cache_for_every_task(
    all_pillars_registered: None,
    seed_kb: CompletenessKB,
) -> None:
    """Re-running an identical plan through an in-memory cache should
    consume zero scripted LLM responses on the second pass.
    """
    rows = _load_rows()
    rubrics = {p: load_rubric(RUBRICS_DIR / f"{p}.yaml") for p in PILLARS}
    bundles = {
        p: load_judge_bundle(JUDGES_DIR / f"{p}.yaml", rubric_root=RUBRICS_DIR) for p in PILLARS
    }
    cache = InMemoryOutcomeCache()

    def _run(first: bool) -> None:
        scripts = _build_scripts(rows, rubrics, seed_kb)
        llms = {
            p: MockLLMClient(
                model_name=f"mock-{p}",
                # On the first run we need the full script. On the
                # second we don't - every task should cache-hit. Use an
                # empty script on pass 2 so any surprise miss raises.
                structured_script=scripts[p] if first else [],
            )
            for p in PILLARS
        }
        plan = RunPlan(
            rows=rows,
            pillars=list(PILLARS),
            bundles=bundles,
            llm_by_pillar=llms,
            run_context=RunContext(
                run_id=f"stage7-cache-{'a' if first else 'b'}",
                dataset_fingerprint="sha256:full-schema-sample-v1",
                kb_version=seed_kb.fingerprint(),
            ),
            kb=seed_kb,
            cache=cache,
            concurrency=ConcurrencyPolicy(max_workers=4),
        )
        result = EvaluationRunner().run(plan)
        expected_total = len(rows) * len(PILLARS)
        assert result.summary.succeeded == expected_total
        if not first:
            assert result.summary.cache_hits == expected_total

    _run(first=True)
    _run(first=False)
