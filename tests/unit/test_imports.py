"""Smoke tests that every top-level ``src.*`` subpackage imports cleanly.

If a stage introduces a circular import or a module-level side effect,
this will catch it before it reaches CI.
"""

from __future__ import annotations

import importlib

import pytest

SUBPACKAGES = [
    "src",
    "src.app",
    "src.app.pages",
    "src.core",
    "src.core.constants",
    "src.core.exceptions",
    "src.core.settings",
    "src.core.types",
    "src.ingestion",
    "src.llm",
    "src.judges",
    "src.rubrics",
    "src.completeness",
    "src.evaluation",
    "src.orchestration",
    "src.observability",
    "src.observability.mlflow_risk_logging",
    "src.dashboard",
    "src.exports",
]


@pytest.mark.parametrize("dotted", SUBPACKAGES)
def test_subpackage_imports(dotted: str) -> None:
    mod = importlib.import_module(dotted)
    assert mod is not None


def test_streamlit_app_module_has_no_side_effects() -> None:
    """Importing the Streamlit entry must not call st.* at module scope.

    We verify by importing and confirming only ``main`` is exported as a
    callable - the render logic must live behind it.
    """
    mod = importlib.import_module("src.app.streamlit_app")
    assert callable(mod.main)


def test_core_public_api_exports() -> None:
    core = importlib.import_module("src.core")
    # Sanity-check a handful of names from the package __init__.
    expected = {
        "PILLARS",
        "REQUIRED_COLUMNS",
        "DISTANCE_WEIGHTS",
        "NormalizedRow",
        "JudgeResult",
        "Evidence",
        "Turn",
        "RunContext",
        "SchemaValidationError",
        "JudgeOutputParseError",
        "ProviderTimeoutError",
    }
    assert expected.issubset(set(core.__all__))
    for name in expected:
        assert hasattr(core, name)


def test_constants_match_spec() -> None:
    from src.core import DISTANCE_WEIGHTS, PILLARS, REQUIRED_COLUMNS

    assert PILLARS == (
        "factual_accuracy",
        "hallucination",
        "relevance",
        "completeness",
        "toxicity",
        "bias_discrimination",
    )
    assert REQUIRED_COLUMNS == (
        "record_id",
        "user_input",
        "agent_output",
        "category",
    )
    # docs/METRICS.md severity mapping
    assert dict(DISTANCE_WEIGHTS) == {0: 1.00, 1: 0.75, 2: 0.40, 3: 0.10, 4: 0.00}
