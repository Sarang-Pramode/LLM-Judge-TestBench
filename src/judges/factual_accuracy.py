"""Factual-accuracy judge.

Measures whether the agent's claims are supported by the user input
and retrieved context. Uses the shared :class:`JudgeCoreOutput` schema
and the default rubric-aware prompt builder - no per-pillar logic is
needed at the code level; all tuning lives in the YAML rubric + config.
"""

from __future__ import annotations

from src.core.constants import PILLAR_FACTUAL_ACCURACY
from src.judges.base import BaseJudge
from src.judges.registry import register_judge

__all__ = ["FactualAccuracyJudge"]


@register_judge
class FactualAccuracyJudge(BaseJudge):
    """Judge for the ``factual_accuracy`` pillar."""

    pillar = PILLAR_FACTUAL_ACCURACY

    def _marker(self) -> None:  # pragma: no cover - structural only
        return None
