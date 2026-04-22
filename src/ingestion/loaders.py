"""File loaders for CSV, XLSX, JSON, and Parquet.

Each loader returns a :class:`LoadedFile` - the raw source columns plus the
rows represented as plain ``dict[str, Any]`` values. Downstream code
(schema_mapper, validators, normalizer) never sees a ``DataFrame`` so that
the rest of the pipeline stays decoupled from pandas and easy to unit-test.

Design notes:
- JSON-like string fields are *not* parsed here. Parsing lives in
  :mod:`src.ingestion.normalizer` so a single, consistent policy applies
  across all formats (CSV always delivers strings; Parquet may deliver
  already-decoded lists; XLSX gives a mix).
- ``NaN`` values from pandas are normalised to ``None`` at this boundary
  so downstream code never has to worry about float NaN semantics.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.exceptions import SchemaValidationError

__all__ = [
    "SUPPORTED_FORMATS",
    "FileFormat",
    "LoadedFile",
    "detect_format",
    "load_file",
    "load_from_bytes",
]

FileFormat = str  # one of the SUPPORTED_FORMATS values

SUPPORTED_FORMATS: tuple[FileFormat, ...] = ("csv", "xlsx", "json", "parquet")

_EXTENSIONS: dict[str, FileFormat] = {
    ".csv": "csv",
    ".tsv": "csv",  # treated as CSV with tab separator
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
    ".parquet": "parquet",
    ".pq": "parquet",
}


@dataclass(frozen=True)
class LoadedFile:
    """The raw contents of an uploaded file.

    Attributes:
        source_columns: Column names in their original order, as seen in
            the source file. Preserved even if a column is entirely empty.
        rows: One dict per data row. Keys are source column names; values
            are Python scalars or JSON-decodable values (lists/dicts). NaN
            is normalized to ``None``.
        path: Optional path the file was loaded from (``None`` for
            in-memory loads).
        format: The detected file format.
    """

    source_columns: list[str]
    rows: list[dict[str, Any]]
    path: Path | None
    format: FileFormat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_format(path: Path | str) -> FileFormat:
    """Infer the file format from the extension.

    Raises :class:`SchemaValidationError` if the extension is unknown. We
    surface a ``SchemaValidationError`` rather than ``ValueError`` so the
    Streamlit UI can display a single class of "upload problem".
    """
    p = Path(path)
    fmt = _EXTENSIONS.get(p.suffix.lower())
    if fmt is None:
        raise SchemaValidationError(
            f"Unsupported file extension: {p.suffix!r}. " f"Supported: {sorted(set(_EXTENSIONS))}."
        )
    return fmt


def load_file(path: Path | str) -> LoadedFile:
    """Load a file from disk, dispatching on its extension."""
    p = Path(path)
    if not p.exists():
        raise SchemaValidationError(f"File not found: {p}")
    fmt = detect_format(p)
    with p.open("rb") as fh:
        return load_from_bytes(fh.read(), fmt=fmt, path=p)


def load_from_bytes(
    data: bytes,
    *,
    fmt: FileFormat,
    path: Path | None = None,
) -> LoadedFile:
    """Load a file from raw bytes (e.g. a Streamlit upload buffer)."""
    if fmt not in SUPPORTED_FORMATS:
        raise SchemaValidationError(f"Unsupported format: {fmt!r}. Supported: {SUPPORTED_FORMATS}.")
    loader = _LOADERS[fmt]
    try:
        columns, rows = loader(data)
    except SchemaValidationError:
        raise
    except Exception as exc:
        raise SchemaValidationError(f"Failed to parse {fmt.upper()} payload: {exc}") from exc
    return LoadedFile(source_columns=columns, rows=rows, path=path, format=fmt)


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------


def _load_csv(data: bytes) -> tuple[list[str], list[dict[str, Any]]]:
    import io

    # keep_default_na=False so "" / "null" / "NaN" reach the normalizer as
    # strings and get canonicalised in one place instead of pandas second-
    # guessing us. The normalizer decides what "empty" means.
    df = pd.read_csv(
        io.BytesIO(data),
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    columns = [str(c) for c in df.columns]
    rows = df.to_dict(orient="records")
    return columns, [_scrub_nans(r) for r in rows]


def _load_xlsx(data: bytes) -> tuple[list[str], list[dict[str, Any]]]:
    import io

    # First sheet only for now. A future change may expose sheet selection
    # through the upload UI.
    df = pd.read_excel(io.BytesIO(data), dtype=object, engine="openpyxl")
    columns = [str(c) for c in df.columns]
    rows = df.to_dict(orient="records")
    return columns, [_scrub_nans(r) for r in rows]


def _load_json(data: bytes) -> tuple[list[str], list[dict[str, Any]]]:
    text = data.decode("utf-8").strip()
    if not text:
        return [], []

    # Try a whole-document JSON parse first. If it fails because the text
    # is actually NDJSON (multiple objects, one per line), fall back to a
    # line-by-line parse. Single JSON scalars/objects/arrays are handled
    # by the first branch.
    rows_raw: list[Any]
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        rows_raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        if isinstance(parsed, list):
            rows_raw = parsed
        elif isinstance(parsed, dict):
            # Accept {"rows": [...]} or {"data": [...]} shapes, or a
            # single-row object (wrapped into a one-element list).
            if "rows" in parsed and isinstance(parsed["rows"], list):
                rows_raw = parsed["rows"]
            elif "data" in parsed and isinstance(parsed["data"], list):
                rows_raw = parsed["data"]
            else:
                rows_raw = [parsed]
        else:
            raise SchemaValidationError(
                "Top-level JSON value must be an array or object; got " f"{type(parsed).__name__}."
            )

    rows: list[dict[str, Any]] = []
    column_order: list[str] = []
    seen: set[str] = set()
    for raw in rows_raw:
        if not isinstance(raw, dict):
            raise SchemaValidationError(
                "JSON rows must be objects; got " f"{type(raw).__name__} instead."
            )
        rows.append(raw)
        for key in raw:
            if key not in seen:
                seen.add(key)
                column_order.append(str(key))
    return column_order, rows


def _load_parquet(data: bytes) -> tuple[list[str], list[dict[str, Any]]]:
    import io

    df = pd.read_parquet(io.BytesIO(data), engine="pyarrow")
    columns = [str(c) for c in df.columns]
    rows = df.to_dict(orient="records")
    return columns, [_scrub_nans(r) for r in rows]


_LOADERS: dict[FileFormat, Any] = {
    "csv": _load_csv,
    "xlsx": _load_xlsx,
    "json": _load_json,
    "parquet": _load_parquet,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scrub_nans(row: Mapping[Any, Any]) -> dict[str, Any]:
    """Normalize a row dict into plain Python types.

    Responsibilities at the loader boundary:
    - Replace float ``NaN`` with ``None`` (pandas emits NaN for empty
      cells even when ``dtype=str``).
    - Unwrap NumPy / pyarrow container types (``ndarray``,
      ``MaskedArray``) to Python lists so downstream code does not have
      to know the source format.
    """
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        cleaned[str(key)] = _to_python(value)
    return cleaned


def _to_python(value: Any) -> Any:
    """Convert NumPy/pyarrow containers and NaN into plain Python types."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    # Lazy import numpy so users without numpy installed don't pay for it.
    # pandas pulls numpy in anyway when the loaders run; this import is
    # cheap after that point.
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a transitive dep of pandas
        np = None  # type: ignore[assignment]

    if np is not None and isinstance(value, np.ndarray):
        return [_to_python(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    return value
