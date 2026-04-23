"""Bias / discrimination judge.

Measures unwarranted generalisations, stereotypes, or differential
treatment based on protected attributes. Independent of toxicity.
"""

from __future__ import annotations

from src.core.constants import PILLAR_BIAS_DISCRIMINATION
from src.judges.base import BaseJudge
from src.judges.registry import register_judge

__all__ = ["BiasDiscriminationJudge"]


@register_judge
class BiasDiscriminationJudge(BaseJudge):
    """Judge for the ``bias_discrimination`` pillar."""

    pillar = PILLAR_BIAS_DISCRIMINATION

    def _marker(self) -> None:  # pragma: no cover - structural only
        return None
