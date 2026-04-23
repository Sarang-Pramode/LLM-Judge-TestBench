"""Normalize raw source rows into :class:`NormalizedRow` instances.

Responsibilities (single source of truth for these policies):

1. **Empty-ish normalisation.** ``""``, ``"null"``, ``"None"``, ``"NaN"``,
   ``"na"``, ``"n/a"`` (case-insensitive) and pure whitespace all collapse
   to ``None``. This is applied uniformly regardless of source format.
2. **JSON-like parsing.** ``retrieved_context``, ``chat_history`` and
   ``metadata`` are parsed from strings via ``json.loads`` first and
   ``ast.literal_eval`` as a fallback. Parse failures are recorded as
   per-row :class:`RowFailure` entries rather than raising.
3. **Type coercion.** Integer label columns accept string representations
   (``"3"`` -> ``3``). Free-text fields are coerced to ``str`` when the
   source supplied another scalar type.
4. **Source preservation.** Every source column (mapped *or* unmapped)
   lands in ``NormalizedRow.source_extras`` so downstream debugging has
   access to the original payload verbatim.

The module deliberately does not enforce required-column presence; that
lives in :mod:`src.ingestion.validators` which operates on the mapping.
Normalisation assumes the mapping has already been validated.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from src.core.types import NormalizedRow, Turn

__all__ = [
    "EMPTY_LIKE_STRINGS",
    "NormalizationResult",
    "RowFailure",
    "is_empty_ish",
    "normalize_rows",
    "parse_chat_history",
    "parse_json_like",
    "parse_metadata",
    "parse_retrieved_context",
]

# Case-insensitive tokens treated as missing values.
EMPTY_LIKE_STRINGS: frozenset[str] = frozenset({"", "null", "none", "nan", "na", "n/a", "nil"})

# Int-typed normalized fields (coerce strings like "3" -> 3).
_INT_FIELDS: frozenset[str] = frozenset(
    {
        "label_factual_accuracy",
        "label_hallucination",
        "label_relevance",
        "label_completeness",
        "label_toxicity",
        "label_bias_discrimination",
        "turn_index",
    }
)

# Fields that must parse into a list[str]. Single strings are wrapped.
_CONTEXT_FIELD = "retrieved_context"
# Fields that must parse into list[Turn]-compatible dicts.
_HISTORY_FIELD = "chat_history"
# Fields that must parse into a dict[str, Any].
_METADATA_FIELD = "metadata"


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowFailure:
    """A single row that failed to normalize, with diagnostic info."""

    row_index: int
    record_id: str | None
    reason: str
    details: dict[str, str] = field(default_factory=dict)


@dataclass
class NormalizationResult:
    """Outcome of :func:`normalize_rows`.

    ``failures`` is recorded per-row rather than raising so the UI can
    show "9 of 10 rows normalized; row 4 has a malformed context".
    """

    rows: list[NormalizedRow]
    failures: list[RowFailure]
    total: int

    @property
    def success_count(self) -> int:
        return len(self.rows)

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def success_rate(self) -> float:
        return (len(self.rows) / self.total) if self.total else 0.0


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_empty_ish(value: Any) -> bool:
    """Return ``True`` for values that should collapse to ``None``."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in EMPTY_LIKE_STRINGS
    return False


def parse_json_like(value: Any) -> Any:
    """Best-effort parse of JSON-ish text into native Python types.

    Returns the parsed value on success, or raises ``ValueError`` if no
    parser succeeded. Non-string inputs are returned unchanged (the
    source file may have produced a real list/dict already, e.g. via
    Parquet or JSON).
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    # json first (stricter, more common)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # literal_eval second (handles single quotes / Python repr dumps)
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Could not parse JSON-like value: {exc}") from exc


def parse_retrieved_context(value: Any) -> list[str | dict[str, Any]] | None:
    """Parse ``retrieved_context`` into ``list[str | dict[str, Any]] | None``.

    Accepts (from most permissive to strictest):

    - ``None`` / empty-ish -> ``None``
    - ``str`` -> parsed as JSON if possible; otherwise treated as a
      single-chunk doc blob wrapped into ``[text]``.
    - ``list`` -> each element kept as-is if it's a string or a dict;
      any other scalar is stringified. ``None``/empty entries are
      dropped so downstream code can trust ``len(context) > 0``.
    - ``dict`` -> wrapped as a single structured chunk: ``[dict]``.

    Judge prompts pretty-print dict chunks as JSON, so the exact
    schema of dict items is a consumer-side concern - there is no
    required shape. This keeps the system compatible with arbitrary
    RAG payloads (per-chunk scores, doc IDs, metadata, etc.).
    """
    if is_empty_ish(value):
        return None
    if isinstance(value, str):
        try:
            parsed = parse_json_like(value)
        except ValueError:
            # Non-JSON free-form text blob - treat as a single chunk of
            # raw document text. This makes the pipeline tolerant of RAG
            # systems that dump raw doc text into a CSV cell.
            return [value]
    else:
        parsed = value
    if parsed is None:
        return None
    if isinstance(parsed, list):
        out: list[str | dict[str, Any]] = []
        for x in parsed:
            if is_empty_ish(x):
                continue
            if isinstance(x, dict):
                # Keys must be strings so downstream JSON round-trips.
                out.append({str(k): v for k, v in x.items()})
            elif isinstance(x, str):
                out.append(x)
            else:
                out.append(str(x))
        return out or None
    if isinstance(parsed, dict):
        return [{str(k): v for k, v in parsed.items()}]
    if isinstance(parsed, str):
        return [parsed]
    # Unknown scalar type - stringify as a single chunk. Previous
    # versions raised here; loosening keeps ingestion forgiving given
    # that retrieved_context is a strongly-recommended field, not a
    # required one, and users want arbitrary RAG payloads to flow
    # through without extra parsing code.
    return [str(parsed)]


def parse_chat_history(value: Any) -> list[Turn] | None:
    """Parse ``chat_history`` into a list of :class:`Turn`."""
    if is_empty_ish(value):
        return None
    parsed = parse_json_like(value) if isinstance(value, str) else value
    if parsed is None:
        return None
    if not isinstance(parsed, list):
        raise ValueError(f"chat_history must be a list of turns; got {type(parsed).__name__}.")
    turns: list[Turn] = []
    for i, raw in enumerate(parsed):
        if not isinstance(raw, dict):
            raise ValueError(f"chat_history[{i}] is not an object: {raw!r}.")
        try:
            turns.append(Turn.model_validate(raw))
        except ValidationError as exc:
            raise ValueError(f"chat_history[{i}] invalid Turn: {exc.errors()}") from exc
    return turns


def parse_metadata(value: Any) -> dict[str, Any] | None:
    """Parse ``metadata`` into ``dict[str, Any] | None``."""
    if is_empty_ish(value):
        return None
    parsed = parse_json_like(value) if isinstance(value, str) else value
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        raise ValueError(f"metadata must be an object/dict; got {type(parsed).__name__}.")
    return {str(k): v for k, v in parsed.items()}


# ---------------------------------------------------------------------------
# Normalization core
# ---------------------------------------------------------------------------


def normalize_rows(
    source_rows: list[dict[str, Any]],
    *,
    mapping: dict[str, str],
) -> NormalizationResult:
    """Transform raw source rows into :class:`NormalizedRow` models.

    Args:
        source_rows: Raw rows as produced by the file loaders. Keys are
            original source column names.
        mapping: ``{normalized_field_name: source_column_name}``. Only
            populated normalized fields appear; unmapped optional fields
            are left as their defaults. Should already be validated by
            :mod:`src.ingestion.validators` (required keys present,
            referenced source columns exist, etc.).

    Returns:
        A :class:`NormalizationResult` with successful rows and any
        per-row failures. Never raises for row-level issues; ingestion
        always returns a structured report.
    """
    successes: list[NormalizedRow] = []
    failures: list[RowFailure] = []

    for idx, source_row in enumerate(source_rows):
        try:
            row = _normalize_single(idx, source_row, mapping)
        except _RowError as exc:
            failures.append(
                RowFailure(
                    row_index=idx,
                    record_id=exc.record_id,
                    reason=exc.reason,
                    details=exc.details,
                )
            )
            continue
        successes.append(row)

    return NormalizationResult(rows=successes, failures=failures, total=len(source_rows))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


class _RowError(Exception):
    """Internal signal for a single-row normalization failure."""

    def __init__(
        self,
        reason: str,
        *,
        record_id: str | None = None,
        details: dict[str, str] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.record_id = record_id
        self.details = details or {}


def _normalize_single(
    row_index: int,
    source_row: dict[str, Any],
    mapping: dict[str, str],
) -> NormalizedRow:
    # Pull a best-effort record_id early so failures can point at a row.
    record_id_col = mapping.get("record_id")
    maybe_record_id: str | None = None
    if record_id_col and record_id_col in source_row:
        raw = source_row[record_id_col]
        if not is_empty_ish(raw):
            maybe_record_id = str(raw)

    kwargs: dict[str, Any] = {}
    parse_errors: dict[str, str] = {}

    for normalized_field, source_col in mapping.items():
        if source_col not in source_row:
            # Mapping references a column that's missing in this particular
            # row (can happen with JSON sources where keys are optional).
            continue
        raw_value = source_row[source_col]

        try:
            value = _coerce_value(normalized_field, raw_value)
        except ValueError as exc:
            parse_errors[normalized_field] = str(exc)
            continue

        if value is None:
            continue  # let Pydantic defaults/optionality handle it
        kwargs[normalized_field] = value

    kwargs["source_extras"] = {k: _json_safe(v) for k, v in source_row.items()}

    if parse_errors:
        # Surface parse errors but still try to build the row; required
        # fields may still be present. If they aren't, pydantic will raise
        # below and we'll merge both diagnostic sets.
        pass

    try:
        row = NormalizedRow(**kwargs)
    except ValidationError as exc:
        details: dict[str, str] = {**parse_errors}
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            details.setdefault(loc or "<row>", err.get("msg", "validation error"))
        raise _RowError(
            reason="Pydantic validation failed",
            record_id=maybe_record_id,
            details=details,
        ) from exc

    if parse_errors:
        raise _RowError(
            reason="Field parse errors",
            record_id=row.record_id,
            details=parse_errors,
        )

    return row


def _coerce_value(field_name: str, value: Any) -> Any:
    """Apply field-specific parsing/coercion.

    Returns ``None`` to indicate "treat as missing".
    """
    if field_name == _CONTEXT_FIELD:
        return parse_retrieved_context(value)
    if field_name == _HISTORY_FIELD:
        return parse_chat_history(value)
    if field_name == _METADATA_FIELD:
        return parse_metadata(value)

    if is_empty_ish(value):
        return None

    if field_name in _INT_FIELDS:
        return _coerce_int(field_name, value)

    # Default: coerce to trimmed string for text fields.
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return str(value)


def _coerce_int(field_name: str, value: Any) -> int:
    if isinstance(value, bool):
        # bool is subclass of int in Python but we reject it explicitly;
        # a True/False in a label column is almost certainly a data bug.
        raise ValueError(f"{field_name}: boolean is not a valid integer label.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field_name}: expected integer, got non-integer float {value!r}.")
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return int(stripped)
        except ValueError:
            # CSV exports often round-trip integer columns as "5.0" when
            # pandas widened the dtype to float (e.g. because of a NaN in
            # the column). Accept that cleanly by trying float() first.
            try:
                as_float = float(stripped)
            except ValueError as exc:
                raise ValueError(f"{field_name}: could not parse {value!r} as integer.") from exc
            if not as_float.is_integer():
                raise ValueError(
                    f"{field_name}: expected integer, got non-integer float {value!r}."
                ) from None
            return int(as_float)
    raise ValueError(f"{field_name}: unsupported type {type(value).__name__}.")


def _json_safe(value: Any) -> Any:
    """Best-effort conversion so ``source_extras`` is JSON-serialisable.

    ``NormalizedRow`` holds ``source_extras`` as ``dict[str, Any]`` which
    Pydantic accepts broadly, but we scrub NaNs and stringify obviously
    non-JSON types (e.g. ``datetime``) to keep downstream exports safe.
    """
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    # Fallback: stringify. (pandas Timestamp, numpy scalars, etc.)
    return str(value)
