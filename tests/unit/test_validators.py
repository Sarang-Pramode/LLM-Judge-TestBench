"""Tests for :mod:`src.ingestion.validators`."""

from __future__ import annotations

import pytest

from src.core.constants import REQUIRED_COLUMNS
from src.core.exceptions import SchemaValidationError
from src.ingestion.normalizer import NormalizationResult, RowFailure, normalize_rows
from src.ingestion.schema_mapper import ColumnMapping
from src.ingestion.validators import (
    ValidationReport,
    validate_mapping,
    validate_normalization,
)

# ---------------------------------------------------------------------------
# validate_mapping
# ---------------------------------------------------------------------------


def test_validate_mapping_ok_when_required_covered_and_sources_exist() -> None:
    mapping = ColumnMapping(
        mappings={
            "record_id": "id",
            "user_input": "prompt",
            "agent_output": "response",
            "category": "topic",
        }
    )
    report = validate_mapping(
        mapping, source_columns=["id", "prompt", "response", "topic", "extra"]
    )
    assert report.ok is True
    assert report.issues == []


def test_validate_mapping_flags_missing_required_columns() -> None:
    mapping = ColumnMapping(mappings={"record_id": "id"})
    report = validate_mapping(mapping, source_columns=["id"])
    assert report.ok is False
    issue_cols = {i.column for i in report.issues}
    assert {c for c in REQUIRED_COLUMNS if c != "record_id"} <= issue_cols
    for issue in report.issues:
        assert issue.issue == "missing_required_mapping"


def test_validate_mapping_flags_unknown_source_columns() -> None:
    mapping = ColumnMapping(
        mappings={
            "record_id": "id",
            "user_input": "prompt",
            "agent_output": "response",
            "category": "topic_does_not_exist",
        }
    )
    report = validate_mapping(mapping, source_columns=["id", "prompt", "response", "topic"])
    issues = [i for i in report.issues if i.issue == "unknown_source_column"]
    assert len(issues) == 1
    assert issues[0].column == "category"


def test_validation_report_raises_aggregated_error() -> None:
    mapping = ColumnMapping(mappings={"record_id": "id"})
    report = validate_mapping(mapping, source_columns=["id"])
    with pytest.raises(SchemaValidationError) as info:
        report.raise_if_errors("mapping is bad")
    assert "mapping is bad" in str(info.value)
    assert len(info.value.issues) == len(report.issues)


def test_validation_report_report_property_returns_plain_dicts() -> None:
    mapping = ColumnMapping(mappings={"record_id": "id"})
    report = validate_mapping(mapping, source_columns=["id"])
    with pytest.raises(SchemaValidationError) as info:
        report.raise_if_errors("bad")
    plain = info.value.report
    assert isinstance(plain, list)
    assert all(set(item.keys()) == {"column", "issue", "detail"} for item in plain)


# ---------------------------------------------------------------------------
# validate_normalization
# ---------------------------------------------------------------------------


def test_validate_normalization_ok_when_no_failures() -> None:
    result = NormalizationResult(rows=[], failures=[], total=0)
    assert validate_normalization(result).ok is True


def test_validate_normalization_aggregates_column_level_counts() -> None:
    failures = [
        RowFailure(
            row_index=0, record_id="r0", reason="parse", details={"retrieved_context": "oops"}
        ),
        RowFailure(
            row_index=1, record_id="r1", reason="parse", details={"retrieved_context": "oops"}
        ),
        RowFailure(row_index=2, record_id="r2", reason="parse", details={"label_toxicity": "oops"}),
    ]
    result = NormalizationResult(rows=[], failures=failures, total=3)
    report = validate_normalization(result)
    assert not report.ok
    counts = {i.column: i.detail for i in report.issues}
    assert "retrieved_context" in counts
    # Highest-count column should appear first.
    assert report.issues[0].column == "retrieved_context"


def test_validate_normalization_adds_catchall_for_detail_less_failures() -> None:
    failures = [RowFailure(row_index=0, record_id=None, reason="boom", details={})]
    result = NormalizationResult(rows=[], failures=failures, total=1)
    report = validate_normalization(result)
    assert any(i.column == "<row>" for i in report.issues)


# ---------------------------------------------------------------------------
# End-to-end (small) - mapping validation + normalization reporting
# ---------------------------------------------------------------------------


def test_mapping_validation_and_normalization_compose() -> None:
    """Mapping passes validation; normalization reports any per-row failures.

    Fails the second row via a boolean in a label column (labels reject
    booleans explicitly - see ``_coerce_int``). retrieved_context is
    deliberately *not* used to force a failure here because the
    contract accepts arbitrary free-text blobs as a single chunk.
    """
    source_rows = [
        {"id": "r1", "prompt": "hi", "response": "hello", "topic": "greetings", "lab": 4},
        {"id": "r2", "prompt": "?", "response": "?", "topic": "q", "lab": True},
    ]
    mapping = ColumnMapping(
        mappings={
            "record_id": "id",
            "user_input": "prompt",
            "agent_output": "response",
            "category": "topic",
            "label_toxicity": "lab",
        }
    )
    source_columns = ["id", "prompt", "response", "topic", "lab"]
    map_report: ValidationReport = validate_mapping(mapping, source_columns=source_columns)
    assert map_report.ok is True

    norm_result = normalize_rows(source_rows, mapping=mapping.mappings)
    norm_report = validate_normalization(norm_result)
    assert not norm_report.ok
    assert "label_toxicity" in {i.column for i in norm_report.issues}
