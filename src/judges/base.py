"""Judge framework core: output contracts, outcome wrapper, and the
:class:`BaseJudge` ABC that orchestrates prompt -> LLM -> parse ->
assemble.

Shape of a judge execution:

1. :meth:`BaseJudge.run` gets a :class:`NormalizedRow` + a
   :class:`RunContext`.
2. The judge builds a prompt via :meth:`BaseJudge.build_prompt` (which
   delegates to :mod:`prompt_builder` by default).
3. An :class:`LLMRequest` is constructed from the rubric-derived prompt
   plus the judge's :class:`JudgeConfig` (temperature, retry policy,
   timeout, tags).
4. ``llm.generate_structured(request, JudgeCoreOutput)`` returns a
   schema-validated :class:`JudgeCoreOutput`.
5. :mod:`output_parser` applies rubric-level constraints (failure tag
   taxonomy, pillar match).
6. The judge composes a :class:`src.core.types.JudgeResult` - which
   enforces its own invariants - and returns a :class:`JudgeOutcome`.

:class:`JudgeOutcome` always carries timing, usage, and attempt counts
so orchestration (Stage 7) can surface partial failures without having
to re-instrument every judge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.constants import MAX_RUBRIC_ANCHOR_DELTA, SCORE_MAX
from src.core.exceptions import (
    JudgeExecutionError,
    JudgeOutputParseError,
    ProviderError,
)
from src.core.types import Evidence, JudgeResult, NormalizedRow, RunContext
from src.judges.config import JudgeConfig
from src.judges.output_parser import validate_against_rubric
from src.judges.prompt_builder import PromptPair, build_default_prompt
from src.llm.base import LLMClient, LLMRequest, LLMUsage
from src.rubrics.models import Rubric

__all__ = [
    "BaseJudge",
    "JudgeCoreOutput",
    "JudgeOutcome",
]

Score = Literal[1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# LLM-facing output schema
# ---------------------------------------------------------------------------


class JudgeCoreOutput(BaseModel):
    """The slice of the judge output the LLM is asked to produce.

    Excludes metadata the system injects (``raw_model_name``,
    ``prompt_version``, ``rubric_version``). Structurally mirrors
    :class:`src.core.types.JudgeResult` so composing the final result
    is a straight field-copy plus three metadata additions.

    Invariants mirror :class:`JudgeResult` to catch bad LLM output at
    the provider boundary, before it touches the runner.
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

    @model_validator(mode="after")
    def _check_invariants(self) -> JudgeCoreOutput:
        if self.score == SCORE_MAX and self.failure_tags:
            raise ValueError(
                "JudgeCoreOutput: score=5 must have an empty failure_tags list "
                f"(got {self.failure_tags!r})."
            )
        if self.score < SCORE_MAX and (self.why_not_higher is None or self.why_not_higher == ""):
            raise ValueError("JudgeCoreOutput: why_not_higher is required when score < 5.")
        if self.score > 1 and (self.why_not_lower is None or self.why_not_lower == ""):
            raise ValueError("JudgeCoreOutput: why_not_lower is required when score > 1.")
        if abs(self.rubric_anchor - self.score) > MAX_RUBRIC_ANCHOR_DELTA:
            raise ValueError(
                "JudgeCoreOutput: rubric_anchor must be within +/- 1 of score "
                f"(score={self.score}, rubric_anchor={self.rubric_anchor})."
            )
        return self


# ---------------------------------------------------------------------------
# Outcome wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeOutcome:
    """Unified per-call outcome.

    Successful calls populate :attr:`result`. Failures populate
    :attr:`error` and :attr:`error_type` so orchestration can decide
    whether to retry, surface, or skip at the row level.

    This is deliberately a frozen dataclass (not a Pydantic model) so
    callers can cheaply build them in hot paths; it carries primitive
    fields + a Pydantic :class:`JudgeResult` when successful.
    """

    pillar: str
    record_id: str
    latency_ms: float
    attempts: int
    usage: LLMUsage
    model_name: str | None = None
    run_id: str | None = None
    result: JudgeResult | None = None
    error: str | None = None
    error_type: str | None = None
    prompt_versions: tuple[str, str] | None = None
    #: Extra key-value pairs (e.g. completeness mode flag). Stays a
    #: plain dict so judges don't have to subclass ``JudgeOutcome`` just
    #: to attach pillar-specific metadata.
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.result is not None and self.error is None

    def as_dict(self) -> dict[str, object]:
        """Serialisation helper for exports/logging - no Pydantic roundtrip."""
        return {
            "pillar": self.pillar,
            "record_id": self.record_id,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "model_name": self.model_name,
            "run_id": self.run_id,
            "error": self.error,
            "error_type": self.error_type,
            "extras": dict(self.extras),
            "result": (self.result.model_dump(mode="json") if self.result else None),
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


# ---------------------------------------------------------------------------
# Base judge
# ---------------------------------------------------------------------------


class BaseJudge(ABC):
    """Abstract single-pillar judge.

    Subclasses in Stage 5 provide:

    - :attr:`pillar` as a class constant (keeps it introspectable at
      class level for the registry).
    - An optional :attr:`output_model` if the pillar extends
      :class:`JudgeCoreOutput` (e.g. hallucination with spans).
    - Optional overrides of :meth:`build_prompt`,
      :meth:`validate_required_inputs`, or :meth:`_post_validate`.

    Constructor wiring:

    - ``config``: a :class:`JudgeConfig` (typically from YAML).
    - ``rubric``: the :class:`Rubric` cross-referenced from the config
      (typically loaded via :func:`load_judge_bundle`).
    - ``llm``: a concrete :class:`LLMClient` (or mock in tests).

    Invariants:

    - ``config.pillar == rubric.pillar == cls.pillar``. Checked at
      construction to avoid mis-wired judges at runtime.
    """

    #: Must be set by subclasses; identifies the pillar. Registry keys
    #: off this value, so two classes with the same pillar conflict.
    pillar: ClassVar[str] = ""

    #: Output schema actually sent to the LLM. Subclasses can override
    #: with a more specific model as long as it subclasses
    #: :class:`JudgeCoreOutput`, so the base post-validation still runs.
    output_model: ClassVar[type[JudgeCoreOutput]] = JudgeCoreOutput

    def __init__(
        self,
        *,
        config: JudgeConfig,
        rubric: Rubric,
        llm: LLMClient,
    ) -> None:
        if not self.pillar:
            raise JudgeExecutionError(
                f"{type(self).__name__}.pillar must be set as a class constant."
            )
        if config.pillar != self.pillar:
            raise JudgeExecutionError(
                f"JudgeConfig pillar {config.pillar!r} does not match "
                f"{type(self).__name__}.pillar ({self.pillar!r})."
            )
        if rubric.pillar != self.pillar:
            raise JudgeExecutionError(
                f"Rubric pillar {rubric.pillar!r} does not match "
                f"{type(self).__name__}.pillar ({self.pillar!r})."
            )
        self.config = config
        self.rubric = rubric
        self.llm = llm

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, row: NormalizedRow, *, run_context: RunContext) -> JudgeOutcome:
        """Run this judge against a single row and return a :class:`JudgeOutcome`.

        Never raises on *provider* or *parse* failures - those are
        captured into the outcome. Only fatal programming errors (bad
        config at construction time, or subclass contract breaks)
        propagate, because those indicate a bug rather than a run-time
        data issue.
        """
        self.validate_required_inputs(row)
        prompt = self.build_prompt(row, run_context=run_context)
        request = self._build_request(prompt, row=row, run_context=run_context)

        try:
            structured = self.llm.generate_structured(request, self.output_model)
        except ProviderError as exc:
            return JudgeOutcome(
                pillar=self.pillar,
                record_id=row.record_id,
                latency_ms=0.0,
                attempts=0,
                usage=LLMUsage(),
                model_name=self.llm.model_name,
                run_id=run_context.run_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )

        try:
            validated = self._post_validate(structured.parsed)
            result = self._assemble_result(validated, model_name=self.llm.model_name)
        except (JudgeOutputParseError, ValueError) as exc:
            return JudgeOutcome(
                pillar=self.pillar,
                record_id=row.record_id,
                latency_ms=structured.latency_ms,
                attempts=structured.attempts,
                usage=structured.usage,
                model_name=self.llm.model_name,
                run_id=run_context.run_id,
                error=str(exc),
                error_type=type(exc).__name__,
                prompt_versions=(self.config.prompt_version, self.rubric.version),
            )

        return JudgeOutcome(
            pillar=self.pillar,
            record_id=row.record_id,
            latency_ms=structured.latency_ms,
            attempts=structured.attempts,
            usage=structured.usage,
            model_name=self.llm.model_name,
            run_id=run_context.run_id,
            result=result,
            prompt_versions=(self.config.prompt_version, self.rubric.version),
            extras=self.extra_outcome_fields(row=row, run_context=run_context),
        )

    # ------------------------------------------------------------------
    # Subclass hooks (override as needed)
    # ------------------------------------------------------------------

    def build_prompt(self, row: NormalizedRow, *, run_context: RunContext) -> PromptPair:
        """Return (system_prompt, user_prompt).

        Default implementation calls the shared rubric-aware builder
        which is enough for all Stage 5 pillars except completeness
        (which overrides to inject the KB-informed task profile).
        """
        return build_default_prompt(
            rubric=self.rubric,
            row=row,
            prompt_version=self.config.prompt_version,
            run_context=run_context,
        )

    def validate_required_inputs(self, row: NormalizedRow) -> None:
        """Raise :class:`JudgeExecutionError` if the rubric requires a
        field the row does not carry.

        ``user_input``, ``agent_output``, ``category`` are always
        present on :class:`NormalizedRow` (required), so only
        ``retrieved_context``, ``chat_history``, and metadata-style
        inputs need run-time checks.
        """
        missing: list[str] = []
        for name in self.rubric.required_inputs:
            value = _extract_field(row, name)
            if value is None:
                missing.append(name)
        if missing:
            raise JudgeExecutionError(
                f"Row {row.record_id!r} is missing rubric-required input(s) "
                f"{missing} for pillar {self.pillar!r}."
            )

    def extra_outcome_fields(
        self,
        *,
        row: NormalizedRow,
        run_context: RunContext,
    ) -> dict[str, str]:
        """Override to attach pillar-specific flags (e.g. completeness
        mode) to the successful outcome. Default: no extras.
        """
        return {}

    # ------------------------------------------------------------------
    # Internal helpers (not expected to be overridden)
    # ------------------------------------------------------------------

    def _build_request(
        self,
        prompt: PromptPair,
        *,
        row: NormalizedRow,
        run_context: RunContext,
    ) -> LLMRequest:
        tags: dict[str, str] = {
            "run_id": run_context.run_id,
            "pillar": self.pillar,
            "record_id": row.record_id,
            "prompt_version": self.config.prompt_version,
            "rubric_version": self.rubric.version,
        }
        if run_context.model_alias:
            tags["model_alias"] = run_context.model_alias
        return LLMRequest(
            system_prompt=prompt.system,
            user_prompt=prompt.user,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            timeout_s=self.config.timeout_s,
            retry=self.config.retry,
            tags=tags,
        )

    def _post_validate(self, parsed: JudgeCoreOutput) -> JudgeCoreOutput:
        """Apply rubric-level checks on top of Pydantic validation."""
        return validate_against_rubric(parsed, rubric=self.rubric)

    def _assemble_result(self, core: JudgeCoreOutput, *, model_name: str) -> JudgeResult:
        """Compose a :class:`JudgeResult`, attaching system metadata."""
        return JudgeResult(
            pillar=core.pillar,
            score=core.score,
            confidence=core.confidence,
            decision_summary=core.decision_summary,
            evidence_for_score=list(core.evidence_for_score),
            failure_tags=list(core.failure_tags),
            why_not_higher=core.why_not_higher,
            why_not_lower=core.why_not_lower,
            rubric_anchor=core.rubric_anchor,
            raw_model_name=model_name,
            prompt_version=self.config.prompt_version,
            rubric_version=self.rubric.version,
        )

    # ------------------------------------------------------------------
    # Documented abstract hook - a marker that subclasses must exist.
    # Kept so static type checkers know ``BaseJudge`` is abstract even
    # though every concrete piece has a default. Stage 5 pillar classes
    # override ``pillar`` (class constant) which is the only contract.
    # ------------------------------------------------------------------

    @abstractmethod
    def _marker(self) -> None:  # pragma: no cover - structural only
        """Abstract marker; subclasses need not do anything here.

        The ABC mechanism requires at least one abstract method to
        prevent direct instantiation of :class:`BaseJudge`. The real
        contract is the ``pillar`` class constant.
        """


# ---------------------------------------------------------------------------
# Field extraction helper (shared by prompt builder + required-input check)
# ---------------------------------------------------------------------------


def _extract_field(row: NormalizedRow, name: str) -> object | None:
    """Return the rubric-visible value for ``name`` on ``row``.

    For ``metadata``, returns the whole dict (or ``None`` if empty).
    For container fields (``retrieved_context``, ``chat_history``),
    returns ``None`` when the list is empty so judges can treat
    "missing" and "empty list" uniformly.
    """
    if name == "metadata":
        meta = row.metadata
        return meta if meta else None
    if name == "retrieved_context":
        ctx = row.retrieved_context
        return ctx if ctx else None
    if name == "chat_history":
        hist = row.chat_history
        return hist if hist else None
    return getattr(row, name, None)
