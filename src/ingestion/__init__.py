"""Dataset ingestion: loaders, mapping, normalization, validation.

Public entrypoints:
- :func:`load_file` / :func:`load_from_bytes` - read a source file.
- :class:`ColumnMapping` - captures source-column -> normalized-field mapping.
- :func:`validate_mapping` - required-column enforcement before evaluation.
- :func:`normalize_rows` - build :class:`NormalizedRow` instances.

All other modules should consume the normalized output only; source
schemas must not leak past this package boundary.
"""

from __future__ import annotations

from src.ingestion.loaders import (
    SUPPORTED_FORMATS,
    LoadedFile,
    detect_format,
    load_file,
    load_from_bytes,
)
from src.ingestion.normalizer import (
    NormalizationResult,
    RowFailure,
    is_empty_ish,
    normalize_rows,
    parse_chat_history,
    parse_json_like,
    parse_metadata,
    parse_retrieved_context,
)
from src.ingestion.schema_mapper import (
    ALLOWED_NORMALIZED_FIELDS,
    ColumnMapping,
    auto_suggest_mapping,
    load_mapping,
    save_mapping,
)
from src.ingestion.validators import (
    ValidationReport,
    validate_mapping,
    validate_normalization,
)

__all__ = [
    "ALLOWED_NORMALIZED_FIELDS",
    "SUPPORTED_FORMATS",
    "ColumnMapping",
    "LoadedFile",
    "NormalizationResult",
    "RowFailure",
    "ValidationReport",
    "auto_suggest_mapping",
    "detect_format",
    "is_empty_ish",
    "load_file",
    "load_from_bytes",
    "load_mapping",
    "normalize_rows",
    "parse_chat_history",
    "parse_json_like",
    "parse_metadata",
    "parse_retrieved_context",
    "save_mapping",
    "validate_mapping",
    "validate_normalization",
]
