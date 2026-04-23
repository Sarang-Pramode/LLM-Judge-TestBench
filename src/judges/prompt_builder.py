"""Rubric-aware prompt construction.

This module owns the *shape* of a judge prompt but is deliberately
free of LLM SDK concerns - it returns plain strings. The LLM layer
(:mod:`src.llm`) decides how to ship those to a vendor.

Design rules:

- Templates live in this module, not in YAML. SMEs tune rubric
  content; prompt structure is a code-level concern kept consistent
  across pillars. Stage 5 pillar judges override :func:`build_prompt`
  only when the pillar genuinely needs a different structure (e.g.
  completeness wants the KB task profile injected).
- Builder is pure and deterministic; identical inputs yield identical
  strings. This matters for caching (Stage 7) and for reproducible
  tests.
- No provider-specific formatting. The structured-output schema is
  referenced by name only - the provider layer is responsible for
  enforcing it via LangChain's ``with_structured_output``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.completeness.task_profile import TaskProfile
from src.core.types import NormalizedRow, RunContext, Turn
from src.rubrics.models import Rubric, ScoreAnchor

__all__ = [
    "PromptPair",
    "build_default_prompt",
    "render_row_block",
    "render_rubric_block",
    "render_task_profile_block",
]


@dataclass(frozen=True)
class PromptPair:
    """Return type of every prompt builder - ``(system, user)`` strings."""

    system: str
    user: str


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_default_prompt(
    *,
    rubric: Rubric,
    row: NormalizedRow,
    prompt_version: str,
    run_context: RunContext | None = None,
) -> PromptPair:
    """Build the default judge prompt for ``rubric`` scoring ``row``.

    The system prompt carries the rubric + output contract; the user
    prompt carries the data being judged.

    ``run_context`` is accepted but not interpolated into the prompt -
    exposing run IDs etc. to the LLM would just add noise and
    hallucination surface. It's included in the signature so prompt
    builder subclasses have the hook available.
    """
    del run_context  # retained for signature compatibility (see docstring)

    system = _build_system_prompt(rubric=rubric, prompt_version=prompt_version)
    user = _build_user_prompt(rubric=rubric, row=row)
    return PromptPair(system=system, user=user)


# ---------------------------------------------------------------------------
# Rubric -> system prompt
# ---------------------------------------------------------------------------


def render_rubric_block(rubric: Rubric) -> str:
    """Render the rubric anchors + failure taxonomy as prompt text."""
    lines: list[str] = []
    lines.append(
        f"Pillar: {rubric.pillar}\n"
        f"Rubric version: {rubric.version}\n"
        f"Score scale: {rubric.score_min} (worst) to {rubric.score_max} (best)\n"
    )
    lines.append("Score anchors:")
    for score in rubric.scores():
        anchor = rubric.anchor_map[score]
        lines.append(_format_anchor(anchor))
    if rubric.failure_tags:
        lines.append("")
        lines.append(
            "Allowed failure_tags (use ONLY these strings; empty list is valid "
            f"when score == {rubric.score_max}):"
        )
        for tag in rubric.failure_tags:
            lines.append(f"  - {tag}")
    return "\n".join(lines)


def _format_anchor(anchor: ScoreAnchor) -> str:
    return f"  {anchor.score} ({anchor.name}): {anchor.description}"


def _build_system_prompt(*, rubric: Rubric, prompt_version: str) -> str:
    rubric_block = render_rubric_block(rubric)
    return (
        "You are a strict, impartial evaluation judge. You score a single "
        f"chatbot response on one rubric pillar: {rubric.pillar}.\n"
        "\n"
        "Rubric (follow exactly):\n"
        f"{rubric_block}\n"
        "\n"
        "Output contract (your response is parsed as strict JSON; do not add "
        "fields, do not omit fields, do not add commentary outside the JSON):\n"
        "  - pillar: string, MUST equal the rubric pillar above.\n"
        f"  - score: integer in [{rubric.score_min}, {rubric.score_max}].\n"
        "  - confidence: float in [0.0, 1.0].\n"
        "  - decision_summary: short natural-language summary (1-3 sentences).\n"
        "  - evidence_for_score: list of {claim, status, support?}; may be empty.\n"
        "  - failure_tags: list of strings (use ONLY from the allowed list above).\n"
        f"  - why_not_higher: string, REQUIRED when score < {rubric.score_max}.\n"
        f"  - why_not_lower: string, REQUIRED when score > {rubric.score_min}.\n"
        "  - rubric_anchor: integer, the anchor closest to your score "
        "(within +/- 1 of score).\n"
        "\n"
        "Hard invariants (violating any of these will cause your response to "
        "be rejected):\n"
        f"  - If score == {rubric.score_max}, failure_tags MUST be empty.\n"
        "  - If failure_tags is non-empty, score MUST be less than "
        f"{rubric.score_max}.\n"
        "  - Every failure tag must come from the rubric's allowed list.\n"
        "  - pillar must exactly equal: " + rubric.pillar + "\n"
        "\n"
        f"Prompt version: {prompt_version}\n"
    )


# ---------------------------------------------------------------------------
# Row -> user prompt
# ---------------------------------------------------------------------------


def render_row_block(row: NormalizedRow, *, rubric: Rubric) -> str:
    """Render the slice of the row this pillar is expected to see.

    Only the rubric's ``required_inputs`` are rendered (plus the
    always-present user_input / agent_output) to keep prompt size
    focused and avoid leaking irrelevant columns.
    """
    parts: list[str] = []
    parts.append(f"record_id: {row.record_id}")
    parts.append(f"category: {row.category}")
    parts.append("")
    parts.append("User input:")
    parts.append(_indent_block(row.user_input))
    parts.append("")
    parts.append("Agent output (this is what you are scoring):")
    parts.append(_indent_block(row.agent_output))

    for name in rubric.required_inputs:
        if name in {"user_input", "agent_output", "category"}:
            # Already rendered above - skip duplicates.
            continue
        rendered = _render_optional(name, row)
        if rendered is None:
            continue
        parts.append("")
        parts.append(rendered)
    return "\n".join(parts)


def _build_user_prompt(*, rubric: Rubric, row: NormalizedRow) -> str:
    row_block = render_row_block(row, rubric=rubric)
    return (
        "Score the following row against the rubric you were given.\n"
        "\n"
        f"{row_block}\n"
        "\n"
        "Return ONLY the structured JSON object defined by the output "
        "contract. No preamble, no markdown, no trailing commentary."
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _indent_block(text: str, *, indent: str = "  ") -> str:
    """Indent every line of ``text`` for readability inside the prompt."""
    if text == "":
        return f"{indent}(empty)"
    return "\n".join(f"{indent}{line}" for line in text.splitlines()) or f"{indent}(empty)"


def _render_optional(name: str, row: NormalizedRow) -> str | None:
    """Render an optional rubric-visible field. Returns ``None`` when
    the row has no value; the caller skips that section.
    """
    if name == "retrieved_context":
        chunks = row.retrieved_context or []
        if not chunks:
            return None
        header = f"Retrieved context ({len(chunks)} chunk{'s' if len(chunks) != 1 else ''}):"
        rendered = "\n".join(
            f"  [{i}] {_indent_block(_stringify_chunk(chunk), indent='      ').lstrip()}"
            for i, chunk in enumerate(chunks, start=1)
        )
        return f"{header}\n{rendered}"

    if name == "chat_history":
        history = row.chat_history or []
        if not history:
            return None
        header = f"Chat history ({len(history)} turn{'s' if len(history) != 1 else ''}):"
        lines = [header]
        for turn in history:
            lines.append(_render_turn(turn))
        return "\n".join(lines)

    if name == "metadata":
        meta = row.metadata or {}
        if not meta:
            return None
        try:
            rendered = json.dumps(meta, indent=2, sort_keys=True, default=str)
        except TypeError:
            rendered = str(meta)
        return f"Row metadata:\n{_indent_block(rendered)}"

    if name == "intent":
        return None if not row.intent else f"Intent: {row.intent}"
    if name == "topic":
        return None if not row.topic else f"Topic: {row.topic}"
    if name == "ground_truth_answer":
        gt = row.ground_truth_answer
        if not gt:
            return None
        return f"Ground-truth answer (if available):\n{_indent_block(gt)}"
    if name == "policy_reference":
        ref = row.policy_reference
        return None if not ref else f"Policy reference: {ref}"

    # Unknown optional name - return None so the prompt keeps going.
    # Rubric validation already rejects unknown names at load time; this
    # is just a defensive guard for future fields.
    return None


def _render_turn(turn: Turn) -> str:
    content = _indent_block(turn.content, indent="      ").lstrip()
    return f"  [{turn.role}] {content}"


def render_task_profile_block(profile: TaskProfile) -> str:
    """Render a :class:`TaskProfile` as a compact, prompt-friendly block.

    The block intentionally lists elements as bullets so the LLM can
    mirror them verbatim in ``elements_present`` / ``elements_missing``.
    The matcher's example_agent_response is NOT included here, to avoid
    anchoring the judge to a specific phrasing of a good answer.
    """
    lines: list[str] = []
    lines.append("Task-specific completeness profile (KB-informed):")
    lines.append(f"  Source KB entry: {profile.source_kb_id}")
    lines.append(f"  Match confidence: {profile.match_confidence:.2f} ({profile.match_reason})")
    if profile.priority_level:
        lines.append(f"  Priority level: {profile.priority_level}")
    if profile.domain:
        lines.append(f"  Domain: {profile.domain}")

    lines.append("")
    lines.append(
        "SME completeness notes (treat as the authoritative definition of "
        "a complete answer for this row):"
    )
    lines.append(_indent_block(profile.completeness_notes))

    lines.append("")
    lines.append(
        "Required elements (every item MUST appear in the agent's answer "
        "for a 5. Classify each into `elements_present` or "
        "`elements_missing`; together they must cover this list):"
    )
    if profile.required_elements:
        for item in profile.required_elements:
            lines.append(f"  - {item}")
    else:
        lines.append("  (none declared)")

    if profile.optional_elements:
        lines.append("")
        lines.append(
            "Optional elements (nice-to-have; do NOT penalise their "
            "absence, but cite them in `why_not_higher` if relevant):"
        )
        for item in profile.optional_elements:
            lines.append(f"  - {item}")

    if profile.forbidden_elements:
        lines.append("")
        lines.append(
            "Forbidden elements (presence of any of these caps the score "
            "at 2 and must appear in `failure_tags`):"
        )
        for item in profile.forbidden_elements:
            lines.append(f"  - {item}")

    if profile.policy_refs:
        lines.append("")
        lines.append("Policy references (context only; do not invent content):")
        for ref in profile.policy_refs:
            lines.append(f"  - {ref}")

    return "\n".join(lines)


def _stringify_chunk(chunk: str | dict[str, object]) -> str:
    """Render a retrieved_context entry.

    Strings are passed through verbatim. Dict chunks (structured RAG
    payloads) are pretty-printed as JSON so the judge can see keys
    like ``doc_id``, ``score``, ``text`` without any parsing contract
    being baked into the judge code.
    """
    if isinstance(chunk, str):
        return chunk
    try:
        return json.dumps(chunk, indent=2, sort_keys=True, default=str)
    except TypeError:
        return str(chunk)
