"""Pre-normalization validators.

These checks run *before* :func:`src.ingestion.normalizer.normalize_rows`
and enforce the hard gate on required normalized columns. Failures here
raise :class:`SchemaValidationError` with a structured report that the
Streamlit upload page renders back to the user.

Scope boundaries:
- This module knows about the mapping and the raw source column set.
- It does NOT know about row values (that's the normalizer's job).
- It does NOT know about UI (that's ``src/app/pages/01_upload.py``).
"""

from __future__ import annotations

from src.core.constants import REQUIRED_COLUMNS
from src.core.exceptions import ColumnIssue, SchemaValidationError
from src.ingestion.normalizer import NormalizationResult
from src.ingestion.schema_mapper import ColumnMapping

__all__ = [
    "ValidationReport",
    "validate_mapping",
    "validate_normalization",
]


class ValidationReport:
    """Accumulates column-level issues; raises once at ``.raise_if_errors()``.

    Behaves like a lightweight mutable builder around
    :class:`SchemaValidationError`. Keeping it separate lets callers
    gather *all* problems (useful for the UI) instead of failing at the
    first one.
    """

    __slots__ = ("issues",)

    def __init__(self) -> None:
        self.issues: list[ColumnIssue] = []

    def add(self, column: str, issue: str, detail: str | None = None) -> None:
        self.issues.append(ColumnIssue(column=column, issue=issue, detail=detail))

    def extend(self, issues: list[ColumnIssue]) -> None:
        self.issues.extend(issues)

    @property
    def ok(self) -> bool:
        return not self.issues

    def raise_if_errors(self, message: str) -> None:
        if self.issues:
            raise SchemaValidationError(message, issues=self.issues)


# ---------------------------------------------------------------------------
# Mapping-level validation
# ---------------------------------------------------------------------------


def validate_mapping(
    mapping: ColumnMapping,
    *,
    source_columns: list[str],
) -> ValidationReport:
    """Validate a mapping against a set of source columns.

    Checks:
    1. Every required normalized column has a mapping entry.
    2. Every source column referenced by the mapping exists in the file.
    3. No duplicate normalized targets (already enforced by the model).

    Returns a :class:`ValidationReport`. Call ``raise_if_errors(...)`` on
    it when you want to hard-fail; the upload UI prefers to render the
    report inline instead.
    """
    report = ValidationReport()
    source_set = set(source_columns)

    for required in REQUIRED_COLUMNS:
        if required not in mapping.mappings:
            report.add(
                column=required,
                issue="missing_required_mapping",
                detail=(
                    f"Required normalized column {required!r} is not mapped. "
                    "Map it to one of your source columns before running evaluation."
                ),
            )

    for normalized_field, source_col in mapping.mappings.items():
        if source_col not in source_set:
            report.add(
                column=normalized_field,
                issue="unknown_source_column",
                detail=(
                    f"Mapping target {normalized_field!r} references source "
                    f"column {source_col!r}, which is not in the uploaded file."
                ),
            )

    return report


# ---------------------------------------------------------------------------
# Post-normalization validation
# ---------------------------------------------------------------------------


def validate_normalization(result: NormalizationResult) -> ValidationReport:
    """Convert a :class:`NormalizationResult` into a column-level report.

    Row failures are aggregated into a small set of column-level issues
    so the upload UI can display them next to the mapping table. The
    caller still has access to the full per-row failure list via
    ``result.failures``.
    """
    report = ValidationReport()
    if not result.failures:
        return report

    field_counts: dict[str, int] = {}
    for failure in result.failures:
        for column in failure.details:
            field_counts[column] = field_counts.get(column, 0) + 1

    for column, count in sorted(field_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        report.add(
            column=column,
            issue="row_parse_failure",
            detail=f"{count} row(s) failed to normalize at {column!r}.",
        )

    # Add a catch-all for failures with no column detail (e.g. missing
    # required-value rows).
    other = sum(1 for f in result.failures if not f.details)
    if other:
        report.add(
            column="<row>",
            issue="row_parse_failure",
            detail=f"{other} row(s) failed with no field-level detail.",
        )

    return report
