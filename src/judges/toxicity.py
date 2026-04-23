"""Toxicity judge.

Measures presence of toxic / abusive / demeaning language in the
agent_output. Intentionally independent of bias_discrimination - a
polite but biased response is NOT toxic; a rude but unbiased response
IS toxic. Scoring both pillars separately keeps the dashboard readable.
"""

from __future__ import annotations

from src.core.constants import PILLAR_TOXICITY
from src.judges.base import BaseJudge
from src.judges.registry import register_judge

__all__ = ["ToxicityJudge"]


@register_judge
class ToxicityJudge(BaseJudge):
    """Judge for the ``toxicity`` pillar."""

    pillar = PILLAR_TOXICITY

    def _marker(self) -> None:  # pragma: no cover - structural only
        return None
