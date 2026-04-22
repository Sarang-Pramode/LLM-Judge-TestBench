"""Integration tests: upload file -> map columns -> validate -> normalize.

Exercises the full ingestion pipeline across every supported file format
using the checked-in sample dataset. Verifies that:

1. Each format loads and produces the same source columns.
2. The saved mapping preset is complete against the sample.
3. Normalization succeeds for every row.
4. Normalized rows carry the expected derived flags.
5. The malformed sample is blocked at the mapping-validation gate with a
   clear, column-level error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.constants import REQUIRED_COLUMNS
from src.core.exceptions import SchemaValidationError
from src.ingestion import (
    auto_suggest_mapping,
    load_file,
    load_mapping,
    normalize_rows,
    validate_mapping,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
MAPPING_PATH = REPO_ROOT / "configs" / "mappings" / "retail_support.yaml"


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(
            f"Sample artifact missing at {path}; run "
            "`python scripts/generate_sample_data.py` to regenerate."
        )


@pytest.mark.parametrize(
    "filename",
    [
        "retail_support.csv",
        "retail_support.xlsx",
        "retail_support.json",
        "retail_support.parquet",
    ],
)
def test_full_ingestion_pipeline_across_formats(filename: str) -> None:
    file_path = SAMPLES_DIR / filename
    _require(file_path)
    _require(MAPPING_PATH)

    loaded = load_file(file_path)
    mapping = load_mapping(MAPPING_PATH)

    report = validate_mapping(mapping, source_columns=loaded.source_columns)
    assert report.ok, [i.__dict__ for i in report.issues]

    result = normalize_rows(loaded.rows, mapping=mapping.mappings)
    assert result.success_count == 12, [f.__dict__ for f in result.failures]
    assert result.failure_count == 0

    # Every row should satisfy the required-column contract.
    for row in result.rows:
        assert row.record_id
        assert row.user_input
        assert row.agent_output
        assert row.category
        assert isinstance(row.source_extras, dict)
        assert "id" in row.source_extras  # original source column preserved

    # At least some rows should have reviewer / context / label coverage so
    # downstream reviewer + agreement analytics actually light up.
    assert any(row.has_reviewer for row in result.rows)
    assert any(row.has_context for row in result.rows)
    assert any(row.has_labels for row in result.rows)


def test_minimal_required_only_sample_normalizes() -> None:
    path = SAMPLES_DIR / "minimal_required.csv"
    _require(path)
    loaded = load_file(path)

    mapping = auto_suggest_mapping(loaded.source_columns)
    assert mapping.is_complete_for_evaluation()

    report = validate_mapping(mapping, source_columns=loaded.source_columns)
    assert report.ok

    result = normalize_rows(loaded.rows, mapping=mapping.mappings)
    assert result.success_count == len(loaded.rows)
    for row in result.rows:
        assert row.has_labels is False
        assert row.has_reviewer is False
        assert row.has_context is False


def test_malformed_sample_fails_at_mapping_validation() -> None:
    path = SAMPLES_DIR / "malformed_missing_category.csv"
    _require(path)
    loaded = load_file(path)

    # Auto-suggest on the malformed source should NOT produce a complete
    # mapping, because the required `category` has no source column.
    mapping = auto_suggest_mapping(loaded.source_columns)
    assert not mapping.is_complete_for_evaluation()
    assert "category" in mapping.missing_required()

    report = validate_mapping(mapping, source_columns=loaded.source_columns)
    assert not report.ok
    missing_cols = {
        issue.column for issue in report.issues if issue.issue == "missing_required_mapping"
    }
    assert "category" in missing_cols

    with pytest.raises(SchemaValidationError) as info:
        report.raise_if_errors("cannot run evaluation")
    assert "cannot run evaluation" in str(info.value)
    assert all(col in REQUIRED_COLUMNS or col == "category" for col in missing_cols)
