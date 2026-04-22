"""Rubric-aware post-validation on :class:`JudgeCoreOutput`.

Pydantic already enforces the *structural* contract (field types,
required fields, numeric bounds, the invariants encoded on
:class:`src.judges.base.JudgeCoreOutput`). This module applies the
*semantic* checks that depend on the specific rubric in play and can't
live inside a static schema:

- ``pillar`` in the output must match the rubric's pillar.
- ``score`` and ``rubric_anchor`` must fall inside the rubric's scale.
- ``failure_tags`` must all come from the rubric's allowed taxonomy.

Failures produce :class:`JudgeOutputParseError` with a structured
:class:`ParseFailure` entry that the UI / observability layer can
surface verbatim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.exceptions import JudgeOutputParseError, ParseFailure
from src.rubrics.models import Rubric

if TYPE_CHECKING:
    from src.judges.base import JudgeCoreOutput

__all__ = ["validate_against_rubric"]


def validate_against_rubric(
    parsed: JudgeCoreOutput,
    *,
    rubric: Rubric,
) -> JudgeCoreOutput:
    """Validate ``parsed`` against ``rubric`` or raise.

    Returns the same object on success (no mutation) so callers can
    chain:

        validated = validate_against_rubric(parsed, rubric=r)
    """
    failures: list[ParseFailure] = []

    if parsed.pillar != rubric.pillar:
        failures.append(
            ParseFailure(
                attempt=1,
                raw_response=parsed.model_dump_json(),
                reason="pillar_mismatch",
                details={
                    "output_pillar": parsed.pillar,
                    "rubric_pillar": rubric.pillar,
                },
            )
        )

    if not (rubric.score_min <= parsed.score <= rubric.score_max):
        failures.append(
            ParseFailure(
                attempt=1,
                raw_response=parsed.model_dump_json(),
                reason="score_out_of_range",
                details={
                    "score": str(parsed.score),
                    "scale": f"[{rubric.score_min}, {rubric.score_max}]",
                },
            )
        )

    if not (rubric.score_min <= parsed.rubric_anchor <= rubric.score_max):
        failures.append(
            ParseFailure(
                attempt=1,
                raw_response=parsed.model_dump_json(),
                reason="rubric_anchor_out_of_range",
                details={
                    "rubric_anchor": str(parsed.rubric_anchor),
                    "scale": f"[{rubric.score_min}, {rubric.score_max}]",
                },
            )
        )

    allowed_tags = rubric.failure_tag_set
    if allowed_tags:
        unknown = [t for t in parsed.failure_tags if t not in allowed_tags]
        if unknown:
            failures.append(
                ParseFailure(
                    attempt=1,
                    raw_response=parsed.model_dump_json(),
                    reason="unknown_failure_tag",
                    details={
                        "unknown_tags": ", ".join(unknown),
                        "allowed": ", ".join(sorted(allowed_tags)),
                    },
                )
            )
    elif parsed.failure_tags:
        # Rubric declares no taxonomy, so any tag is "unknown".
        failures.append(
            ParseFailure(
                attempt=1,
                raw_response=parsed.model_dump_json(),
                reason="failure_tags_without_taxonomy",
                details={"tags": ", ".join(parsed.failure_tags)},
            )
        )

    # Check for duplicate failure tags (Pydantic allows duplicates
    # because ``list[str]`` has no uniqueness constraint; here we treat
    # duplicates as a data bug worth flagging).
    seen: set[str] = set()
    duplicates: list[str] = []
    for tag in parsed.failure_tags:
        if tag in seen:
            duplicates.append(tag)
        seen.add(tag)
    if duplicates:
        failures.append(
            ParseFailure(
                attempt=1,
                raw_response=parsed.model_dump_json(),
                reason="duplicate_failure_tag",
                details={"duplicates": ", ".join(sorted(set(duplicates)))},
            )
        )

    if failures:
        reasons = ", ".join(f.reason for f in failures)
        raise JudgeOutputParseError(
            f"Judge output for pillar {rubric.pillar!r} failed rubric " f"validation: {reasons}.",
            failures=failures,
        )
    return parsed
