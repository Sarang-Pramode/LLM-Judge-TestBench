"""Completeness judge.

Measures whether the agent's response covers the critical elements a
SME would expect for the question. Has two operating modes:

* ``kb_informed``: a :class:`~src.completeness.CompletenessKB` was
  wired in AND the row matched a KB entry above the configured
  threshold. The task profile's ``required_elements`` are injected
  into the prompt and the judge must populate
  ``elements_present`` / ``elements_missing`` against that list.
* ``generic_fallback``: no KB wired, or the best match was too weak.
  The judge evaluates against generic completeness criteria only.

The mode is reported via :class:`JudgeOutcome.extras` so dashboards
can filter on it (e.g. "show completeness calls that ran without KB
grounding"). When ``kb_informed``, extras also carry the matched
``kb_id`` and the match confidence.
"""

from __future__ import annotations

from typing import Final

from src.completeness.kb_matcher import KBMatcher, MatchResult
from src.completeness.models import CompletenessKB
from src.completeness.task_profile import TaskProfile, build_task_profile
from src.core.constants import PILLAR_COMPLETENESS
from src.core.types import NormalizedRow, RunContext
from src.judges.base import BaseJudge
from src.judges.config import JudgeConfig
from src.judges.prompt_builder import (
    PromptPair,
    build_default_prompt,
    render_task_profile_block,
)
from src.judges.registry import register_judge
from src.llm.base import LLMClient
from src.rubrics.models import Rubric

__all__ = [
    "COMPLETENESS_MODE_EXTRA_KEY",
    "COMPLETENESS_MODE_GENERIC_FALLBACK",
    "COMPLETENESS_MODE_KB_INFORMED",
    "COMPLETENESS_MODE_KEY",
    "CompletenessJudge",
]

#: Key used in :attr:`JudgeOutcome.extras` to report the mode the
#: completeness judge ran in. Dashboards / exports should key off this.
COMPLETENESS_MODE_EXTRA_KEY: Final[str] = "completeness_mode"

#: Alias kept for clarity in tests / callers.
COMPLETENESS_MODE_KEY: Final[str] = COMPLETENESS_MODE_EXTRA_KEY

#: The two valid modes for the completeness judge.
COMPLETENESS_MODE_KB_INFORMED: Final[str] = "kb_informed"
COMPLETENESS_MODE_GENERIC_FALLBACK: Final[str] = "generic_fallback"


@register_judge
class CompletenessJudge(BaseJudge):
    """Judge for the ``completeness`` pillar.

    Accepts an optional :class:`CompletenessKB`. Without one (or when
    no KB entry matches the row), the judge runs in
    ``generic_fallback`` mode and behaves like a regular
    :class:`BaseJudge`. With a matching KB entry, it injects a
    :class:`TaskProfile` block into the user prompt and expects the
    LLM to return ``elements_present`` / ``elements_missing`` that
    partition the profile's required elements.

    Match results are cached per ``record_id`` so ``build_prompt`` and
    :meth:`extra_outcome_fields` (called separately by
    :meth:`BaseJudge.run`) both observe the same match decision.
    """

    pillar = PILLAR_COMPLETENESS

    def __init__(
        self,
        *,
        config: JudgeConfig,
        rubric: Rubric,
        llm: LLMClient,
        kb: CompletenessKB | None = None,
        matcher: KBMatcher | None = None,
    ) -> None:
        super().__init__(config=config, rubric=rubric, llm=llm)
        self.kb = kb
        if matcher is not None:
            self._matcher: KBMatcher | None = matcher
        elif kb is not None:
            self._matcher = KBMatcher(kb)
        else:
            self._matcher = None
        # Per-row match cache. BaseJudge.run invokes build_prompt and
        # extra_outcome_fields independently, so we memoise the match
        # the first time we see each record_id to guarantee the prompt
        # and the reported mode agree.
        self._match_cache: dict[str, MatchResult] = {}

    def _marker(self) -> None:  # pragma: no cover - structural only
        return None

    # ------------------------------------------------------------------
    # KB-aware helpers
    # ------------------------------------------------------------------

    def _match_row(self, row: NormalizedRow) -> MatchResult:
        """Return the cached or freshly-computed match for ``row``."""
        if self._matcher is None:
            no_match = MatchResult(
                entry=None,
                confidence=0.0,
                match_reason="no_kb_configured",
                signals={},
            )
            self._match_cache.setdefault(row.record_id, no_match)
            return no_match
        cached = self._match_cache.get(row.record_id)
        if cached is not None:
            return cached
        match = self._matcher.match(row)
        self._match_cache[row.record_id] = match
        return match

    def _profile_for(self, row: NormalizedRow) -> TaskProfile | None:
        return build_task_profile(self._match_row(row))

    # ------------------------------------------------------------------
    # Mode resolution
    # ------------------------------------------------------------------

    def _resolve_mode(
        self,
        *,
        row: NormalizedRow,
        run_context: RunContext,
    ) -> str:
        """Return the operating mode for this row.

        ``kb_informed`` when the matcher returned a hit for ``row``;
        otherwise ``generic_fallback``.
        """
        del run_context  # reserved for future per-run overrides
        match = self._match_row(row)
        if match.is_hit:
            return COMPLETENESS_MODE_KB_INFORMED
        return COMPLETENESS_MODE_GENERIC_FALLBACK

    # ------------------------------------------------------------------
    # Prompt construction: inject task profile when in KB mode.
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        row: NormalizedRow,
        *,
        run_context: RunContext,
    ) -> PromptPair:
        base = build_default_prompt(
            rubric=self.rubric,
            row=row,
            prompt_version=self.config.prompt_version,
            run_context=run_context,
        )

        profile = self._profile_for(row)
        if profile is None:
            # Fallback mode: make the absence of a task profile explicit
            # in the system prompt so the LLM knows the output-contract
            # fields `elements_present` / `elements_missing` should stay
            # empty. Keeps behaviour identical to other pillars when no
            # KB is wired.
            system = base.system + _FALLBACK_ADDENDUM
            return PromptPair(system=system, user=base.user)

        profile_block = render_task_profile_block(profile)
        system = base.system + _KB_ADDENDUM
        user = f"{profile_block}\n\n{base.user}"
        return PromptPair(system=system, user=user)

    # ------------------------------------------------------------------
    # Hook into BaseJudge: attach mode + match metadata to outcome.
    # ------------------------------------------------------------------

    def extra_outcome_fields(
        self,
        *,
        row: NormalizedRow,
        run_context: RunContext,
    ) -> dict[str, str]:
        mode = self._resolve_mode(row=row, run_context=run_context)
        match = self._match_row(row)
        extras: dict[str, str] = {
            COMPLETENESS_MODE_EXTRA_KEY: mode,
            "kb_match": ("hit" if mode == COMPLETENESS_MODE_KB_INFORMED else "none"),
            "kb_match_confidence": f"{match.confidence:.4f}",
        }
        if match.entry is not None:
            extras["kb_id"] = match.entry.kb_id
            extras["kb_match_reason"] = match.match_reason
        return extras


# ---------------------------------------------------------------------------
# Prompt addenda appended to the system prompt based on mode.
# Kept as module constants so tests can assert the exact surface text.
# ---------------------------------------------------------------------------


_KB_ADDENDUM: Final[str] = (
    "\nCompleteness KB mode: KB_INFORMED.\n"
    "You will receive a task-specific completeness profile alongside the "
    "row. Score the agent's response ONLY against that profile's "
    "required_elements and forbidden_elements; do not invent new "
    "requirements. Populate `elements_present` with required items that "
    "ARE clearly addressed in the answer, and `elements_missing` with "
    "required items that are NOT addressed. Every required element must "
    "appear in exactly one of the two lists.\n"
)


_FALLBACK_ADDENDUM: Final[str] = (
    "\nCompleteness KB mode: GENERIC_FALLBACK.\n"
    "No task-specific completeness profile is available for this row. "
    "Score against generic completeness criteria: does the answer "
    "directly address the user's question, cover the key sub-questions, "
    "and provide enough actionable detail. Leave `elements_present` "
    "and `elements_missing` as empty lists in this mode.\n"
)
