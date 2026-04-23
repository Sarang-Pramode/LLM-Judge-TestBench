"""Hallucination judge.

Measures fabrication severity: confidently-stated facts/citations/
numbers/entities that are not grounded in the user_input or
retrieved_context. Shares the :class:`JudgeCoreOutput` schema; the
pillar-specific signal (which claims are fabricated) is captured via
``failure_tags`` + ``evidence_for_score`` entries with ``status =
"unsupported"``.
"""

from __future__ import annotations

from src.core.constants import PILLAR_HALLUCINATION
from src.judges.base import BaseJudge
from src.judges.registry import register_judge

__all__ = ["HallucinationJudge"]


@register_judge
class HallucinationJudge(BaseJudge):
    """Judge for the ``hallucination`` pillar."""

    pillar = PILLAR_HALLUCINATION

    def _marker(self) -> None:  # pragma: no cover - structural only
        return None
