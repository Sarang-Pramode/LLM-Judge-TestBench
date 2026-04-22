"""Column-mapping model for source-schema -> normalized-schema wiring.

The mapper captures "which source column should fill which normalized
field" for a given upload. It is intentionally a pure data model - file
IO (YAML read/write), UI glue, and validation live in adjacent modules
so that mapping objects stay trivially serialisable and testable.

Persistence format (YAML) matches ``configs/mappings/<name>.yaml``:

.. code-block:: yaml

    name: retail_support_v1
    version: "1"
    description: "Mapping for retail_support exports"
    mappings:
      record_id: id
      user_input: prompt
      agent_output: response
      category: topic
      label_relevance: sme_relevance
    source_format: csv       # optional, informational only
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.constants import (
    OPTIONAL_LABEL_COLUMNS,
    OPTIONAL_METADATA_COLUMNS,
    OPTIONAL_RATIONALE_COLUMNS,
    OPTIONAL_REVIEWER_COLUMNS,
    RECOMMENDED_COLUMNS,
    REQUIRED_COLUMNS,
)
from src.core.exceptions import ConfigLoadError

__all__ = [
    "ALLOWED_NORMALIZED_FIELDS",
    "ColumnMapping",
    "MappingSaveLoadError",
    "auto_suggest_mapping",
    "load_mapping",
    "save_mapping",
]


class MappingSaveLoadError(ConfigLoadError):
    """Raised when a mapping preset cannot be read or written."""


# Closed set of normalized field names a mapping is allowed to target.
# Anything else implies a typo or drift between the docs and code.
ALLOWED_NORMALIZED_FIELDS: frozenset[str] = frozenset(
    REQUIRED_COLUMNS
    + RECOMMENDED_COLUMNS
    + OPTIONAL_REVIEWER_COLUMNS
    + OPTIONAL_METADATA_COLUMNS
    + OPTIONAL_LABEL_COLUMNS
    + OPTIONAL_RATIONALE_COLUMNS
)


class ColumnMapping(BaseModel):
    """Maps normalized field names to source column names.

    The mapping is stored as ``{normalized_field: source_column}`` (not the
    other way around) because normalized field names form a small, closed
    vocabulary while source columns are arbitrary strings - this direction
    makes validation and "did the user cover all required fields?" checks
    trivial.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    version: str = "1"
    description: str | None = None
    source_format: str | None = None
    mappings: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_mapping_targets(self) -> ColumnMapping:
        unknown = sorted(set(self.mappings) - ALLOWED_NORMALIZED_FIELDS)
        if unknown:
            raise ValueError(
                "ColumnMapping references unknown normalized fields: "
                f"{unknown}. Allowed: {sorted(ALLOWED_NORMALIZED_FIELDS)}."
            )
        blanks = sorted(
            target for target, source in self.mappings.items() if not str(source).strip()
        )
        if blanks:
            raise ValueError(f"ColumnMapping has blank source columns for fields: {blanks}.")
        # Guard against the same source column being used twice. It is
        # possible but almost always unintentional; forbid it and make
        # the user duplicate the source column in their file if really
        # needed.
        seen: dict[str, str] = {}
        for target, source in self.mappings.items():
            if source in seen:
                raise ValueError(
                    f"Source column {source!r} is mapped to both "
                    f"{seen[source]!r} and {target!r}."
                )
            seen[source] = target
        return self

    # ---- ergonomics --------------------------------------------------------

    def covered_required(self) -> list[str]:
        """Required normalized columns that *are* mapped."""
        return [c for c in REQUIRED_COLUMNS if c in self.mappings]

    def missing_required(self) -> list[str]:
        """Required normalized columns that are *not* mapped yet."""
        return [c for c in REQUIRED_COLUMNS if c not in self.mappings]

    def is_complete_for_evaluation(self) -> bool:
        """True iff all required normalized columns are mapped."""
        return not self.missing_required()

    def source_columns_used(self) -> list[str]:
        """Distinct source columns referenced by this mapping, in order."""
        out: list[str] = []
        seen: set[str] = set()
        for source in self.mappings.values():
            if source not in seen:
                out.append(source)
                seen.add(source)
        return out


# ---------------------------------------------------------------------------
# YAML persistence
# ---------------------------------------------------------------------------


def save_mapping(mapping: ColumnMapping, path: Path | str) -> Path:
    """Persist a mapping to YAML. Returns the path written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = mapping.model_dump(exclude_none=True)
    try:
        p.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except OSError as exc:
        raise MappingSaveLoadError(f"Could not write mapping to {p}: {exc}") from exc
    return p


def load_mapping(path: Path | str) -> ColumnMapping:
    """Load a mapping from YAML, validating it into :class:`ColumnMapping`."""
    p = Path(path)
    if not p.exists():
        raise MappingSaveLoadError(f"Mapping file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MappingSaveLoadError(f"Invalid YAML in {p}: {exc}") from exc
    if raw is None:
        raise MappingSaveLoadError(f"Mapping file is empty: {p}")
    if not isinstance(raw, dict):
        raise MappingSaveLoadError(
            f"Mapping file {p} must contain a mapping at top level; got " f"{type(raw).__name__}."
        )
    try:
        return ColumnMapping.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError or similar
        raise MappingSaveLoadError(f"Mapping file {p} is invalid: {exc}") from exc


# ---------------------------------------------------------------------------
# Auto-suggestion: quick-start for the upload UI
# ---------------------------------------------------------------------------


def auto_suggest_mapping(source_columns: list[str]) -> ColumnMapping:
    """Propose a mapping by matching source columns to normalized names.

    Strategy (required columns always win to avoid blocking evaluation):

    1. Apply well-known synonyms for every required column first (e.g.
       ``id -> record_id``, ``topic -> category``).
    2. Then, for any remaining normalized field whose name appears as a
       source column (case-insensitive, ignoring separators), suggest a
       direct match - but only if the source column is not already
       claimed by step 1.

    The user is expected to review and refine in the upload UI.
    """
    normalized_sources = {_canon(col): col for col in source_columns}
    suggestions: dict[str, str] = {}

    # Step 1: required-column synonyms (highest priority).
    _maybe_add(suggestions, normalized_sources, "record_id", ("record_id", "id", "row_id", "uuid"))
    _maybe_add(
        suggestions,
        normalized_sources,
        "user_input",
        ("user_input", "prompt", "question", "query"),
    )
    _maybe_add(
        suggestions,
        normalized_sources,
        "agent_output",
        ("agent_output", "response", "answer", "reply", "assistant_response"),
    )
    _maybe_add(
        suggestions,
        normalized_sources,
        "category",
        ("category", "topic", "bucket", "tag"),
    )

    # Step 2: direct name matches for remaining fields.
    claimed_sources = set(suggestions.values())
    for target in ALLOWED_NORMALIZED_FIELDS:
        if target in suggestions:
            continue
        canon_target = _canon(target)
        if canon_target not in normalized_sources:
            continue
        source = normalized_sources[canon_target]
        if source in claimed_sources:
            continue
        suggestions[target] = source
        claimed_sources.add(source)

    return ColumnMapping(mappings=suggestions, name="auto_suggested")


def _canon(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _maybe_add(
    out: dict[str, str],
    candidates: dict[str, str],
    target: str,
    aliases: tuple[str, ...],
) -> None:
    if target in out:
        return
    for alias in aliases:
        canon_alias = _canon(alias)
        if canon_alias in candidates and candidates[canon_alias] not in out.values():
            out[target] = candidates[canon_alias]
            return
