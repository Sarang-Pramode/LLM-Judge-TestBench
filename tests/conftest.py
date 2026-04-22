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
