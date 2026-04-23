"""Pydantic models for the completeness knowledge bank.

Each :class:`CompletenessEntry` captures what a SME considers a
"complete" answer for a particular kind of user question, keyed by
intent and topic(s). A :class:`CompletenessKB` is a versioned bundle
of entries - the loader and matcher operate on ``CompletenessKB``,
never on raw dicts.

Required entry fields match the spec in ``docs/PROJECT_CONTEXT.md`` /
``dataset_contract.md``. Recommended fields (``required_elements``,
``forbidden_elements``, policy refs, etc.) are optional so early KB
drafts can be thin but usable; the judge contract degrades gracefully
when a field is missing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ALLOWED_PRIORITY_LEVELS",
    "CompletenessEntry",
    "CompletenessKB",
]

#: Enum-like set of valid priority levels for KB entries. SMEs use these
#: to flag which completeness guarantees are most critical - priority is
#: surfaced verbatim to the judge prompt when present.
ALLOWED_PRIORITY_LEVELS: Final[frozenset[str]] = frozenset({"low", "medium", "high", "critical"})

# Identifier pattern mirrors pillar naming: lower snake_case. KB entries
# are referenced by ID in exports and in run metadata, so a strict
# pattern avoids accidental whitespace / casing drift.
_KB_ID_PATTERN: Final[str] = r"^[a-z][a-z0-9_]*$"


class CompletenessEntry(BaseModel):
    """A single SME-authored completeness guidance entry.

    An entry tells the completeness judge:

    * Which user questions / intents it applies to (matching signals).
    * What a SME considers a complete answer (scoring guidance).
    * Any forbidden content patterns (negative signals).

    The entry is immutable (``frozen=True``) so the KB can be cached
    and reused across many judge calls within a run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- Required ------------------------------------------------------
    kb_id: str = Field(..., min_length=1, pattern=_KB_ID_PATTERN)
    question_or_utterance_pattern: str = Field(..., min_length=1)
    topic_list: list[str] = Field(default_factory=list)
    intent: str = Field(..., min_length=1)
    example_agent_response: str = Field(..., min_length=1)
    completeness_notes: str = Field(..., min_length=1)

    # --- Recommended ---------------------------------------------------
    required_elements: list[str] = Field(default_factory=list)
    optional_elements: list[str] = Field(default_factory=list)
    forbidden_elements: list[str] = Field(default_factory=list)
    policy_refs: list[str] = Field(default_factory=list)
    priority_level: str | None = None
    domain: str | None = None
    version: str = Field(default="1.0", min_length=1)
    author: str | None = None
    last_updated: str | None = None

    @model_validator(mode="after")
    def _check_topic_list(self) -> CompletenessEntry:
        if not self.topic_list:
            raise ValueError(
                f"CompletenessEntry {self.kb_id!r}: topic_list must contain "
                "at least one topic for the matcher to have a signal to use."
            )
        seen: set[str] = set()
        for topic in self.topic_list:
            if not isinstance(topic, str) or not topic.strip():
                raise ValueError(
                    f"CompletenessEntry {self.kb_id!r}: topic_list entries "
                    "must be non-empty strings."
                )
            key = topic.strip().lower()
            if key in seen:
                raise ValueError(
                    f"CompletenessEntry {self.kb_id!r}: duplicate topic {topic!r} in topic_list."
                )
            seen.add(key)
        return self

    @model_validator(mode="after")
    def _check_priority(self) -> CompletenessEntry:
        if self.priority_level is None:
            return self
        if self.priority_level not in ALLOWED_PRIORITY_LEVELS:
            raise ValueError(
                f"CompletenessEntry {self.kb_id!r}: priority_level "
                f"{self.priority_level!r} is not one of "
                f"{sorted(ALLOWED_PRIORITY_LEVELS)}."
            )
        return self

    @model_validator(mode="after")
    def _check_element_lists_disjoint(self) -> CompletenessEntry:
        """``required`` and ``forbidden`` elements must not overlap.

        Checked case-insensitively and after whitespace normalisation
        so SMEs can't accidentally get a self-contradicting entry.
        """

        def _norm(items: list[str]) -> list[str]:
            return [" ".join(x.strip().lower().split()) for x in items]

        required_norm = set(_norm(self.required_elements))
        optional_norm = set(_norm(self.optional_elements))
        forbidden_norm = set(_norm(self.forbidden_elements))

        if required_norm & forbidden_norm:
            overlap = sorted(required_norm & forbidden_norm)
            raise ValueError(
                f"CompletenessEntry {self.kb_id!r}: required_elements and "
                f"forbidden_elements must not overlap; conflicts: {overlap}."
            )
        if required_norm & optional_norm:
            overlap = sorted(required_norm & optional_norm)
            raise ValueError(
                f"CompletenessEntry {self.kb_id!r}: required_elements and "
                f"optional_elements must not overlap; conflicts: {overlap}."
            )
        return self


class CompletenessKB(BaseModel):
    """A versioned collection of :class:`CompletenessEntry` records.

    The whole-KB ``version`` is the single string that should appear on
    :attr:`src.core.types.RunContext.kb_version` so runs are traceable
    back to the KB contents they used.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(..., min_length=1)
    entries: list[CompletenessEntry] = Field(default_factory=list)
    description: str | None = None

    @model_validator(mode="after")
    def _check_unique_ids(self) -> CompletenessKB:
        seen: dict[str, int] = {}
        for idx, entry in enumerate(self.entries):
            if entry.kb_id in seen:
                raise ValueError(
                    f"CompletenessKB: duplicate kb_id {entry.kb_id!r} "
                    f"at positions {seen[entry.kb_id]} and {idx}."
                )
            seen[entry.kb_id] = idx
        return self

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def by_id(self, kb_id: str) -> CompletenessEntry:
        """Return the entry with ``kb_id`` or raise :class:`KeyError`."""
        for entry in self.entries:
            if entry.kb_id == kb_id:
                return entry
        raise KeyError(f"No CompletenessEntry with kb_id={kb_id!r}")

    def __iter__(self) -> Iterator[CompletenessEntry]:  # type: ignore[override]
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------
    # Fingerprinting (for run metadata + observability)
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """Return a stable content hash of the KB.

        Format: ``"<version>:sha256:<hex[:12]>"``. The hash is derived
        from a deterministic JSON dump of every entry, so reordering
        entries does NOT change the fingerprint (we sort by kb_id
        first). Use this for ``RunContext.kb_version`` when persistence
        hasn't settled on a version string convention yet.
        """
        payload = sorted(
            (entry.model_dump(mode="json") for entry in self.entries),
            key=lambda d: d["kb_id"],
        )
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        return f"{self.version}:sha256:{digest[:12]}"
