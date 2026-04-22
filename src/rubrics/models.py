"""Rubric data model.

A rubric captures *how* a pillar is scored: the score scale, each
anchor's narrative description, the allowed failure-tag taxonomy, and
the input fields the rubric expects to reason over.

Rubrics are data (YAML under ``configs/rubrics/``) - not code - so SMEs
can tune them without code review. Pydantic validation catches drift at
load time (e.g. missing a score anchor, referencing an unknown input
column).

Invariants enforced here:

- The score scale must match the project-wide scale declared in
  :mod:`src.core.constants` (1-5). The field still lives on the model
  so future multi-scale pilots can override safely.
- Every score in the scale must have exactly one anchor.
- ``required_inputs`` must reference fields that exist on
  :class:`src.core.types.NormalizedRow` or a canonical shortcut name we
  recognise, otherwise the rubric is unusable.
- ``failure_tags`` must be unique non-empty strings.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.constants import PILLARS, SCORE_MAX, SCORE_MIN

__all__ = [
    "ALLOWED_RUBRIC_INPUTS",
    "Rubric",
    "ScoreAnchor",
    "is_known_pillar",
]

#: Allowed names for :attr:`Rubric.required_inputs`. These must be
#: derivable from a :class:`NormalizedRow`; the prompt builder uses
#: these names to extract fields.
ALLOWED_RUBRIC_INPUTS: frozenset[str] = frozenset(
    {
        "user_input",
        "agent_output",
        "category",
        "retrieved_context",
        "chat_history",
        "metadata",
        "intent",
        "topic",
        "ground_truth_answer",
        "policy_reference",
    }
)


class ScoreAnchor(BaseModel):
    """Narrative description for a single point on the score scale."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    score: int = Field(..., ge=SCORE_MIN, le=SCORE_MAX)
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


class Rubric(BaseModel):
    """SME-authored scoring guide for a single pillar."""

    model_config = ConfigDict(extra="forbid")

    pillar: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    score_min: int = Field(default=SCORE_MIN, ge=1, le=10)
    score_max: int = Field(default=SCORE_MAX, ge=1, le=10)
    required_inputs: list[str] = Field(default_factory=list)
    anchors: list[ScoreAnchor] = Field(..., min_length=1)
    failure_tags: list[str] = Field(default_factory=list)

    # --- Derived conveniences -----------------------------------------

    @property
    def anchor_map(self) -> dict[int, ScoreAnchor]:
        """``{score: anchor}`` for quick lookup in the prompt builder."""
        return {a.score: a for a in self.anchors}

    @property
    def failure_tag_set(self) -> frozenset[str]:
        """Immutable view for membership checks in output validation."""
        return frozenset(self.failure_tags)

    def scores(self) -> range:
        """Inclusive score range (Python ``range`` is end-exclusive)."""
        return range(self.score_min, self.score_max + 1)

    # --- Validators ---------------------------------------------------

    @model_validator(mode="after")
    def _check_scale(self) -> Rubric:
        if self.score_min >= self.score_max:
            raise ValueError(
                f"Rubric scale invalid: score_min ({self.score_min}) must be "
                f"< score_max ({self.score_max})."
            )
        return self

    @model_validator(mode="after")
    def _check_pillar_name(self) -> Rubric:
        # We deliberately allow custom / experimental pillars, but warn
        # the caller (via validation error) if the name is structurally
        # bad. Known v1 pillars live in ``PILLARS``.
        if self.pillar != self.pillar.lower().strip():
            raise ValueError(f"Rubric.pillar must be lowercase and untrimmed; got {self.pillar!r}.")
        return self

    @model_validator(mode="after")
    def _check_anchors(self) -> Rubric:
        seen: set[int] = set()
        for anchor in self.anchors:
            if not (self.score_min <= anchor.score <= self.score_max):
                raise ValueError(
                    f"Anchor score {anchor.score} outside rubric scale "
                    f"[{self.score_min}, {self.score_max}]."
                )
            if anchor.score in seen:
                raise ValueError(f"Duplicate anchor for score {anchor.score}.")
            seen.add(anchor.score)
        missing = [s for s in self.scores() if s not in seen]
        if missing:
            raise ValueError(
                f"Rubric is missing anchors for scores {missing}. Every "
                "score in the scale must have exactly one anchor."
            )
        return self

    @model_validator(mode="after")
    def _check_required_inputs(self) -> Rubric:
        seen: set[str] = set()
        for name in self.required_inputs:
            if name not in ALLOWED_RUBRIC_INPUTS:
                raise ValueError(
                    f"Unknown rubric input {name!r}. Allowed: " f"{sorted(ALLOWED_RUBRIC_INPUTS)}."
                )
            if name in seen:
                raise ValueError(f"Duplicate required_input {name!r}.")
            seen.add(name)
        return self

    @model_validator(mode="after")
    def _check_failure_tags(self) -> Rubric:
        seen: set[str] = set()
        for tag in self.failure_tags:
            if not tag or tag != tag.strip():
                raise ValueError(f"Failure tag {tag!r} must be non-empty and not padded.")
            if tag in seen:
                raise ValueError(f"Duplicate failure tag {tag!r}.")
            seen.add(tag)
        return self


def is_known_pillar(name: str) -> bool:
    """Return True if ``name`` is one of the v1 pillars.

    Helpful for tooling/UI; not used for validation so that custom
    pillars can be piloted without editing the constant list.
    """
    return name in PILLARS
