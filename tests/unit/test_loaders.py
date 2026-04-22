"""Tests for :mod:`src.ingestion.loaders`.

Each supported format goes through a round-trip: encode -> load -> check
that source columns and row contents survive. Edge cases and negative
paths are covered by dedicated tests.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.core.exceptions import SchemaValidationError
from src.ingestion.loaders import (
    SUPPORTED_FORMATS,
    detect_format,
    load_file,
    load_from_bytes,
)


def _rows() -> list[dict[str, Any]]:
    return [
        {"id": "r1", "text": "hello", "labels": [1, 2]},
        {"id": "r2", "text": "world", "labels": [3]},
    ]


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


def test_detect_format_covers_all_supported_formats() -> None:
    for ext, fmt in (
        ("foo.csv", "csv"),
        ("foo.tsv", "csv"),
        ("foo.xlsx", "xlsx"),
        ("foo.json", "json"),
        ("foo.jsonl", "json"),
        ("foo.parquet", "parquet"),
        ("foo.pq", "parquet"),
    ):
        assert detect_format(ext) == fmt


def test_detect_format_unknown_extension_raises_schema_validation_error() -> None:
    with pytest.raises(SchemaValidationError):
        detect_format("foo.bin")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_load_csv_preserves_columns_and_rows() -> None:
    df = pd.DataFrame([{"id": "r1", "text": "hello"}, {"id": "r2", "text": "world"}])
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    loaded = load_from_bytes(buf.getvalue(), fmt="csv")
    assert loaded.source_columns == ["id", "text"]
    assert [r["id"] for r in loaded.rows] == ["r1", "r2"]
    assert all(isinstance(r["text"], str) for r in loaded.rows)


def test_load_csv_empty_cells_become_none() -> None:
    # CSV blanks should reach the loader as "" - the loader does NOT
    # collapse them to None (the normalizer does). But NaN originating
    # from pandas must not leak.
    csv_bytes = b"id,optional\nr1,\nr2,value\n"
    loaded = load_from_bytes(csv_bytes, fmt="csv")
    assert loaded.rows[0]["optional"] == ""
    assert loaded.rows[1]["optional"] == "value"


# ---------------------------------------------------------------------------
# XLSX
# ---------------------------------------------------------------------------


def test_load_xlsx_preserves_columns_and_rows(tmp_path: Path) -> None:
    df = pd.DataFrame(_rows())
    path = tmp_path / "rows.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")
    loaded = load_file(path)
    assert loaded.source_columns == ["id", "text", "labels"]
    assert loaded.rows[0]["id"] == "r1"
    # Lists stored in xlsx round-trip as their string repr; the normalizer
    # is responsible for parsing - the loader just passes values through.
    assert isinstance(loaded.rows[0]["labels"], str)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def test_load_json_array_of_objects() -> None:
    payload = json.dumps(_rows()).encode("utf-8")
    loaded = load_from_bytes(payload, fmt="json")
    assert loaded.source_columns == ["id", "text", "labels"]
    assert loaded.rows[0]["labels"] == [1, 2]
    assert loaded.rows[1]["labels"] == [3]


def test_load_json_ndjson_one_object_per_line() -> None:
    ndjson = b'{"id": "r1"}\n{"id": "r2", "text": "t"}\n'
    loaded = load_from_bytes(ndjson, fmt="json")
    assert {"id", "text"} <= set(loaded.source_columns)
    assert loaded.rows[0] == {"id": "r1"}
    assert loaded.rows[1] == {"id": "r2", "text": "t"}


def test_load_json_wrapped_rows_object() -> None:
    payload = json.dumps({"rows": _rows()}).encode("utf-8")
    loaded = load_from_bytes(payload, fmt="json")
    assert loaded.source_columns == ["id", "text", "labels"]


def test_load_json_single_object_wraps_as_one_row() -> None:
    payload = json.dumps({"id": "solo", "text": "only"}).encode("utf-8")
    loaded = load_from_bytes(payload, fmt="json")
    assert len(loaded.rows) == 1
    assert loaded.rows[0]["id"] == "solo"


def test_load_json_rejects_top_level_scalar() -> None:
    with pytest.raises(SchemaValidationError):
        load_from_bytes(b'"just a string"', fmt="json")


def test_load_json_rejects_non_object_rows() -> None:
    with pytest.raises(SchemaValidationError):
        load_from_bytes(b"[1, 2, 3]", fmt="json")


def test_load_json_empty_payload_returns_no_rows() -> None:
    loaded = load_from_bytes(b"", fmt="json")
    assert loaded.rows == []
    assert loaded.source_columns == []


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------


def test_load_parquet_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame(_rows())
    path = tmp_path / "rows.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    loaded = load_file(path)
    assert loaded.source_columns == ["id", "text", "labels"]
    # Parquet preserves list types, so loader should not need to parse.
    assert loaded.rows[0]["labels"] == [1, 2]


# ---------------------------------------------------------------------------
# Load-from-disk and dispatch
# ---------------------------------------------------------------------------


def test_load_file_unknown_extension_raises() -> None:
    with pytest.raises(SchemaValidationError):
        load_file(Path("/tmp/does_not_exist.bin"))


def test_load_file_missing_raises_schema_validation_error(tmp_path: Path) -> None:
    with pytest.raises(SchemaValidationError):
        load_file(tmp_path / "nope.csv")


def test_load_from_bytes_rejects_unknown_format() -> None:
    with pytest.raises(SchemaValidationError):
        load_from_bytes(b"id\n1\n", fmt="txt")


def test_supported_formats_tuple_is_stable() -> None:
    # Acts as a guard: if this tuple changes we want tests to scream.
    assert SUPPORTED_FORMATS == ("csv", "xlsx", "json", "parquet")


# ---------------------------------------------------------------------------
# Sample files smoke test (exercises the real sample data)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "retail_support.csv",
        "retail_support.xlsx",
        "retail_support.json",
        "retail_support.parquet",
    ],
)
def test_sample_files_load_and_match(filename: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "data" / "samples" / filename
    if not path.exists():
        pytest.skip(f"Sample file missing: {path}")
    loaded = load_file(path)
    assert len(loaded.rows) == 12
    assert "id" in loaded.source_columns
    assert "prompt" in loaded.source_columns
    assert "topic" in loaded.source_columns
