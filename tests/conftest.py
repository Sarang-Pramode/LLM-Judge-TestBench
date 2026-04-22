"""Shared pytest fixtures.

Kept small and stage-1-scoped: a minimal ``NormalizedRow``, a minimal
valid ``JudgeResult``, and a helper to get fresh ``AppSettings`` that
bypass the ``lru_cache`` singleton.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from src.core.settings import AppSettings, get_settings
from src.core.types import Evidence, JudgeResult, NormalizedRow, RunContext
from src.judges.config import JudgeConfig
from src.judges.registry import reset_registry
from src.rubrics.models import Rubric, ScoreAnchor


@pytest.fixture
def normalized_row_kwargs() -> dict[str, Any]:
    """Minimal valid kwargs for ``NormalizedRow``. Tests can extend this."""
    return {
        "record_id": "r-001",
        "user_input": "How do I dispute a transaction?",
        "agent_output": "Open the app, tap the transaction, choose dispute...",
        "category": "transactions",
    }


@pytest.fixture
def normalized_row(normalized_row_kwargs: dict[str, Any]) -> NormalizedRow:
    return NormalizedRow(**normalized_row_kwargs)


@pytest.fixture
def judge_result_kwargs() -> dict[str, Any]:
    """Minimal valid kwargs for ``JudgeResult`` (non-boundary score).

    Score=4 exercises the invariants that ``why_not_higher`` and
    ``why_not_lower`` are both required.
    """
    return {
        "pillar": "factual_accuracy",
        "score": 4,
        "confidence": 0.82,
        "decision_summary": "Mostly correct with one small unsupported detail.",
        "evidence_for_score": [
            Evidence(
                claim="The fee is charged after 30 days.",
                status="supported",
                support="chunk_2",
            ),
            Evidence(
                claim="The fee is always waived for students.",
                status="unsupported",
                support=None,
            ),
        ],
        "failure_tags": ["unsupported_claim"],
        "why_not_higher": "Contains one unsupported material detail.",
        "why_not_lower": "Core answer remains mostly correct and useful.",
        "rubric_anchor": 4,
        "raw_model_name": "mock-model-v0",
        "prompt_version": "2.0.0",
        "rubric_version": "1.2.0",
    }


@pytest.fixture
def judge_result(judge_result_kwargs: dict[str, Any]) -> JudgeResult:
    return JudgeResult(**judge_result_kwargs)


@pytest.fixture
def run_context() -> RunContext:
    return RunContext(
        run_id="run-test-0001",
        dataset_fingerprint="sha256:deadbeef",
        kb_version="completeness-kb-1.0",
        model_alias="judge-default",
        run_config_hash="sha256:cafef00d",
    )


@pytest.fixture
def clean_settings_cache() -> Iterator[None]:
    """Clear the ``get_settings`` lru_cache around the test.

    Tests that tweak env vars should depend on this fixture so they see
    fresh settings and don't bleed into other tests.
    """
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest.fixture
def fresh_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_settings_cache: None,
) -> AppSettings:
    """Return a fresh ``AppSettings`` with paths pointed at ``tmp_path``.

    Handy for modules that read ``settings.configs_dir`` / ``data_dir``.
    """
    monkeypatch.setenv("JTB_CONFIGS_DIR", str(tmp_path / "configs"))
    monkeypatch.setenv("JTB_DATA_DIR", str(tmp_path / "data"))
    return AppSettings()


# ---------------------------------------------------------------------------
# Stage 4: rubric / judge-config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def factual_accuracy_rubric() -> Rubric:
    """A minimal but complete ``Rubric`` covering the full 1-5 scale."""
    return Rubric(
        pillar="factual_accuracy",
        version="v1.0",
        description="Whether agent claims are supported by the retrieved context.",
        required_inputs=["user_input", "agent_output", "retrieved_context"],
        anchors=[
            ScoreAnchor(score=1, name="Severely inaccurate", description="Fabricates."),
            ScoreAnchor(score=2, name="Mostly inaccurate", description="Many errors."),
            ScoreAnchor(score=3, name="Mixed", description="About half correct."),
            ScoreAnchor(score=4, name="Mostly accurate", description="One small miss."),
            ScoreAnchor(score=5, name="Fully accurate", description="All supported."),
        ],
        failure_tags=["unsupported_claim", "fabricated_citation", "contradicts_context"],
    )


@pytest.fixture
def factual_accuracy_config() -> JudgeConfig:
    return JudgeConfig(
        pillar="factual_accuracy",
        prompt_version="factual_accuracy.v1",
        rubric_path="rubrics/factual_accuracy.yaml",
        rubric_version="v1.0",
    )


@pytest.fixture(autouse=True)
def _reset_judge_registry() -> Iterator[None]:
    """Stage 4+ tests commonly register fake judges; keep the registry
    isolated between tests so accidental global state does not leak.
    """
    reset_registry()
    try:
        yield
    finally:
        reset_registry()
