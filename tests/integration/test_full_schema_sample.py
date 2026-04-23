"""Integration test: full_schema_sample.* exercises every documented
normalized column and every accepted ``retrieved_context`` shape.

Loads the shipped ``data/samples/full_schema_sample.csv`` and
``.json``, ingests them end-to-end, and asserts:

1. Every documented column lands on the normalized row.
2. The four accepted ``retrieved_context`` shapes round-trip correctly
   (list[str], list[dict], single dict, free-text blob).
3. Judge prompts render dict chunks as JSON and string chunks
   verbatim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.constants import (
    OPTIONAL_LABEL_COLUMNS,
    OPTIONAL_METADATA_COLUMNS,
    OPTIONAL_RATIONALE_COLUMNS,
    OPTIONAL_REVIEWER_COLUMNS,
    RECOMMENDED_COLUMNS,
    REQUIRED_COLUMNS,
)
from src.ingestion import auto_suggest_mapping, load_file, normalize_rows
from src.judges.prompt_builder import render_row_block
from src.rubrics.models import Rubric, ScoreAnchor

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "data" / "samples"


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(
            f"Sample artifact missing at {path}; run "
            "`python scripts/generate_sample_data.py` to regenerate."
        )


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_full_schema_sample_normalizes_all_columns(fmt: str) -> None:
    path = SAMPLES_DIR / f"full_schema_sample.{fmt}"
    _require(path)
    loaded = load_file(path)
    assert len(loaded.rows) == 4

    # Column names already match the normalized schema -> auto-suggest
    # should produce an identity mapping for every documented field.
    suggested = auto_suggest_mapping(loaded.source_columns)
    mapping = suggested.mappings
    documented = (
        *REQUIRED_COLUMNS,
        *RECOMMENDED_COLUMNS,
        *OPTIONAL_REVIEWER_COLUMNS,
        *OPTIONAL_METADATA_COLUMNS,
        *OPTIONAL_LABEL_COLUMNS,
        *OPTIONAL_RATIONALE_COLUMNS,
    )
    for col in documented:
        assert col in mapping, f"auto-suggest dropped {col}"
        assert mapping[col] == col, f"auto-suggest mismapped {col}"

    result = normalize_rows(loaded.rows, mapping=mapping)
    assert result.failure_count == 0, result.failures
    assert result.success_count == 4

    # Spot-check a single rich row: every field populated.
    row = next(r for r in result.rows if r.record_id == "FS-0001")
    assert row.category == "disputes"
    assert row.reviewer_name == "alex.morgan"
    assert row.reviewer_id == "rev-001"
    assert row.intent == "transaction_dispute"
    assert row.topic == "disputes"
    assert row.conversation_id == "conv-20260101-0001"
    assert row.turn_index == 3
    assert row.ground_truth_answer is not None
    assert row.policy_reference == "policy_disputes_v3"
    assert row.label_factual_accuracy == 5
    assert row.label_completeness == 4
    assert row.rationale_completeness is not None
    assert row.metadata == {
        "device": "ios",
        "app_version": "3.2.1",
        "locale": "en_US",
    }


def test_retrieved_context_shape_list_of_strings() -> None:
    """FS-0001 ships retrieved_context as list[str]."""
    row = _load_row("FS-0001")
    assert row.retrieved_context is not None
    assert all(isinstance(x, str) for x in row.retrieved_context)
    assert "Disputes can be opened" in row.retrieved_context[0]


def test_retrieved_context_shape_list_of_dicts() -> None:
    """FS-0002 ships retrieved_context as list[dict] with per-chunk metadata."""
    row = _load_row("FS-0002")
    assert row.retrieved_context is not None
    assert len(row.retrieved_context) == 2
    first = row.retrieved_context[0]
    assert isinstance(first, dict)
    assert first["doc_id"] == "policy_cardholder_v4"
    assert first["chunk_id"] == "c-17"
    assert first["score"] == 0.91


def test_retrieved_context_shape_single_dict_wrapped() -> None:
    """FS-0003 ships a single-dict payload; ingestion wraps it into a list."""
    row = _load_row("FS-0003")
    assert row.retrieved_context is not None
    assert len(row.retrieved_context) == 1
    only = row.retrieved_context[0]
    assert isinstance(only, dict)
    assert only["doc_id"] == "fraud_policy_v2"
    assert only["title"] == "Fraud protection policy"


def test_retrieved_context_shape_free_text_blob() -> None:
    """FS-0004 ships retrieved_context as a free-text document blob."""
    row = _load_row("FS-0004")
    assert row.retrieved_context is not None
    assert len(row.retrieved_context) == 1
    chunk = row.retrieved_context[0]
    assert isinstance(chunk, str)
    assert "Refund Policy" in chunk
    assert "30 days" in chunk


def test_prompt_renders_dict_chunks_as_json() -> None:
    """A judge prompt for FS-0002 should contain the dict chunk as JSON."""
    row = _load_row("FS-0002")
    rubric = _ctx_rubric()
    block = render_row_block(row, rubric=rubric)
    # Dict fields rendered as pretty JSON:
    assert '"doc_id": "policy_cardholder_v4"' in block
    assert '"score": 0.91' in block
    # String chunks (from FS-0001 pattern) would appear verbatim:
    row1 = _load_row("FS-0001")
    block1 = render_row_block(row1, rubric=rubric)
    assert "Disputes can be opened from the app" in block1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_row(record_id: str):  # type: ignore[no-untyped-def]
    path = SAMPLES_DIR / "full_schema_sample.json"
    _require(path)
    loaded = load_file(path)
    suggested = auto_suggest_mapping(loaded.source_columns)
    result = normalize_rows(loaded.rows, mapping=suggested.mappings)
    return next(r for r in result.rows if r.record_id == record_id)


def _ctx_rubric() -> Rubric:
    """Minimal rubric that includes retrieved_context in required_inputs."""
    return Rubric(
        pillar="factual_accuracy",
        version="v1",
        description="test",
        required_inputs=["user_input", "agent_output", "retrieved_context"],
        anchors=[ScoreAnchor(score=s, name=str(s), description="d") for s in range(1, 6)],
    )
