"""Turn a KB match into the ``TaskProfile`` the judge prompt consumes.

The profile is a compact, judge-facing view of a matched KB entry.
It's deliberately separate from :class:`CompletenessEntry`:

- Entries have SME metadata (author, policy refs, version, domain)
  the judge doesn't need to see.
- Entries might hold example responses the judge should NOT see
  (risk of anchoring to the example text).
- A profile also carries the match confidence/reason, letting
  downstream code (dashboards, exports) filter on match strength
  without re-running the matcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.completeness.kb_matcher import MatchResult

__all__ = ["TaskProfile", "build_task_profile"]


@dataclass(frozen=True)
class TaskProfile:
    """A KB-derived, judge-facing completeness contract.

    Attributes:
        source_kb_id: ``kb_id`` of the entry this profile was derived
            from. Surfaced in ``JudgeOutcome.extras`` for traceability.
        required_elements: Elements a complete answer MUST contain.
            Drives the judge's ``elements_present`` / ``elements_missing``
            partition.
        optional_elements: Elements that are welcome but not required.
            Included in the prompt so the judge can reference them in
            ``why_not_higher`` without penalising their absence.
        forbidden_elements: Patterns that trigger failure tags when
            present in the agent's response.
        completeness_notes: The SME's narrative guidance.
        policy_refs: Optional list of policy identifiers the judge
            can cite by name (it does NOT open these files; they're
            display-only).
        priority_level: SME-assigned priority (``low``/``medium``/...
            or ``None`` if unset).
        domain: Free-form domain tag, included so the judge can be
            audited by domain later.
        match_confidence: Matcher's confidence in ``[0.0, 1.0]``.
        match_reason: Human-readable breakdown of why this entry won.
    """

    source_kb_id: str
    required_elements: list[str]
    optional_elements: list[str]
    forbidden_elements: list[str]
    completeness_notes: str
    policy_refs: list[str]
    priority_level: str | None
    domain: str | None
    match_confidence: float
    match_reason: str
    signals: dict[str, float] = field(default_factory=dict)


def build_task_profile(match: MatchResult) -> TaskProfile | None:
    """Return a :class:`TaskProfile` for ``match`` or ``None`` if no hit.

    Returning ``None`` rather than a sentinel profile forces callers
    to handle the fallback branch explicitly - which is exactly what
    the completeness judge must do to report the mode it ran in.
    """
    if match.entry is None:
        return None
    entry = match.entry
    return TaskProfile(
        source_kb_id=entry.kb_id,
        required_elements=list(entry.required_elements),
        optional_elements=list(entry.optional_elements),
        forbidden_elements=list(entry.forbidden_elements),
        completeness_notes=entry.completeness_notes,
        policy_refs=list(entry.policy_refs),
        priority_level=entry.priority_level,
        domain=entry.domain,
        match_confidence=match.confidence,
        match_reason=match.match_reason,
        signals=dict(match.signals),
    )
