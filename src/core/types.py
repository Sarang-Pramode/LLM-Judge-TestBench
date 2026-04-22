"""Core Pydantic v2 contracts shared across the project.

Canonical specs:
- ``NormalizedRow`` matches ``dataset_contract.md`` at the repo root.
- ``JudgeResult`` matches ``docs/JUDGE_OUTPUT_CONTRACT.md``.

All models use ``extra="forbid"`` so unknown fields fail fast rather than
silently drifting across stages. Per-pillar extensions (e.g. hallucination
span lists) are declared in Stage 5 as subclasses of ``JudgeResult``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

TurnRole = Literal["user", "assistant", "system", "tool"]


class Turn(BaseModel):
    """A single chat turn used inside ``NormalizedRow.chat_history``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: TurnRole
    content: str


# ---------------------------------------------------------------------------
# Normalized dataset row
# ---------------------------------------------------------------------------


class NormalizedRow(BaseModel):
    """The internal representation every downstream module operates on.

    Ingestion is the single place that knows about source schemas; once a
    file has been mapped + normalized, judges / metrics / dashboards all
    consume this model. Original source columns are preserved in
    ``source_extras`` for debugging.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Required ---------------------------------------------------------
    record_id: str = Field(..., min_length=1)
    user_input: str
    agent_output: str
    category: str = Field(..., min_length=1)

    # --- Strongly recommended --------------------------------------------
    retrieved_context: list[str] | None = None
    chat_history: list[Turn] | None = None
    metadata: dict[str, Any] | None = None

    # --- Optional per-pillar labels (ordinal 1-5) ------------------------
    label_factual_accuracy: int | None = Field(default=None, ge=1, le=5)
    label_hallucination: int | None = Field(default=None, ge=1, le=5)
    label_relevance: int | None = Field(default=None, ge=1, le=5)
    label_completeness: int | None = Field(default=None, ge=1, le=5)
    label_toxicity: int | None = Field(default=None, ge=1, le=5)
    label_bias_discrimination: int | None = Field(default=None, ge=1, le=5)

    # --- Optional per-pillar rationales ----------------------------------
    rationale_factual_accuracy: str | None = None
    rationale_hallucination: str | None = None
    rationale_relevance: str | None = None
    rationale_completeness: str | None = None
    rationale_toxicity: str | None = None
    rationale_bias_discrimination: str | None = None

    # --- Optional reviewer + misc metadata -------------------------------
    reviewer_name: str | None = None
    reviewer_id: str | None = None
    intent: str | None = None
    topic: str | None = None
    model_name: str | None = None
    conversation_id: str | None = None
    turn_index: int | None = None
    ground_truth_answer: str | None = None
    policy_reference: str | None = None

    # --- Source preservation + derived flags -----------------------------
    source_extras: dict[str, Any] = Field(default_factory=dict)

    has_reviewer: bool = False
    has_context: bool = False
    has_labels: bool = False
    has_history: bool = False
    has_ground_truth: bool = False

    @model_validator(mode="after")
    def _derive_flags(self) -> NormalizedRow:
        """Populate ``has_*`` flags from the optional fields.

        Runs after field validation so callers can either leave the flags
        unset (preferred) or set them explicitly; explicit values are
        overwritten to stay consistent with actual content. This keeps the
        single source of truth inside the model rather than in ingestion
        code that might drift.
        """
        object.__setattr__(
            self, "has_reviewer", bool(self.reviewer_name or self.reviewer_id)
        )
        object.__setattr__(
            self,
            "has_context",
            self.retrieved_context is not None and len(self.retrieved_context) > 0,
        )
        object.__setattr__(
            self,
            "has_history",
            self.chat_history is not None and len(self.chat_history) > 0,
        )
        object.__setattr__(
            self,
            "has_ground_truth",
            self.ground_truth_answer is not None and self.ground_truth_answer != "",
        )
        object.__setattr__(
            self,
            "has_labels",
            any(
                getattr(self, f"label_{p}") is not None
                for p in (
                    "factual_accuracy",
                    "hallucination",
                    "relevance",
                    "completeness",
                    "toxicity",
                    "bias_discrimination",
                )
            ),
        )
        return self


# ---------------------------------------------------------------------------
# Judge output
# ---------------------------------------------------------------------------

EvidenceStatus = Literal["supported", "unsupported", "contradicted", "ungrounded", "n/a"]


class Evidence(BaseModel):
    """Structured evidence item inside a ``JudgeResult``.

    Each pillar narrows or extends ``status`` and ``support`` semantics in
    its pillar-specific subclass (Stage 5).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str
    status: EvidenceStatus
    support: str | None = None


Score = Literal[1, 2, 3, 4, 5]


class JudgeResult(BaseModel):
    """Base strict output contract for every judge pillar.

    See ``docs/JUDGE_OUTPUT_CONTRACT.md`` for the narrative spec. Per-pillar
    subclasses (``FactualAccuracyResult``, ``CompletenessResult``, ...) will
    add domain-specific fields in Stage 5 but must not relax this schema.
    """

    model_config = ConfigDict(extra="forbid")

    pillar: str = Field(..., min_length=1)
    score: Score
    confidence: float = Field(..., ge=0.0, le=1.0)

    decision_summary: str = Field(..., min_length=1)
    evidence_for_score: list[Evidence] = Field(default_factory=list)
    failure_tags: list[str] = Field(default_factory=list)

    why_not_higher: str | None = None
    why_not_lower: str | None = None

    rubric_anchor: Score

    raw_model_name: str = Field(..., min_length=1)
    prompt_version: str = Field(..., min_length=1)
    rubric_version: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_invariants(self) -> JudgeResult:
        # 5 = best, must have no failure tags. Any non-empty tag set implies
        # score < 5. These two clauses are equivalent but catching both
        # directions yields nicer error messages.
        if self.score == 5 and self.failure_tags:
            raise ValueError(
                "JudgeResult: score=5 must have an empty failure_tags list "
                f"(got {self.failure_tags!r})."
            )
        if self.failure_tags and self.score == 5:
            # Redundant with the above; kept for clarity.
            raise ValueError("JudgeResult: non-empty failure_tags require score < 5.")

        # Boundary explanations are required on the boundaries they describe.
        if self.score < 5 and (self.why_not_higher is None or self.why_not_higher == ""):
            raise ValueError(
                "JudgeResult: why_not_higher is required when score < 5."
            )
        if self.score > 1 and (self.why_not_lower is None or self.why_not_lower == ""):
            raise ValueError(
                "JudgeResult: why_not_lower is required when score > 1."
            )

        # Rubric anchor must be within +/- 1 of the declared score.
        if abs(self.rubric_anchor - self.score) > 1:
            raise ValueError(
                "JudgeResult: rubric_anchor must be within +/- 1 of score "
                f"(score={self.score}, rubric_anchor={self.rubric_anchor})."
            )

        return self


# ---------------------------------------------------------------------------
# Run context (passed from orchestration into every judge)
# ---------------------------------------------------------------------------


class RunContext(BaseModel):
    """Per-run context handed to every judge invocation.

    ``langfuse_span`` is typed ``Any | None`` here; Stage 10 replaces it
    with a real Langfuse span handle. Keeping it loose now avoids pulling
    Langfuse as a hard dep before that stage.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_id: str = Field(..., min_length=1)
    dataset_fingerprint: str = Field(..., min_length=1)
    kb_version: str | None = None
    model_alias: str | None = None
    run_config_hash: str | None = None

    langfuse_span: Any | None = None
